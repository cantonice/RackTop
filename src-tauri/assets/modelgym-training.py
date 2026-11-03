#!/usr/bin/env python3
"""Private, per-SSH-user ModelGym queue. Requires Linux, Python 3 and NVIDIA SMI.

The desktop sends JSON requests; training commands are executed as argv, never
through a shell. Queue state is independent of the desktop/SSH connection.
"""
import base64
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

def stable_host_namespace(machine_identity=None, hostname=None):
    """Shared/NFS HOME directories must never share queues across machines."""
    if machine_identity is None:
        for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id", "/sys/class/dmi/id/product_uuid"):
            try:
                candidate_value = Path(candidate).read_text().strip()
                if len(candidate_value) >= 8:
                    machine_identity = candidate_value
                    break
            except OSError:
                pass
        if machine_identity:
            try:
                product_uuid = Path("/sys/class/dmi/id/product_uuid").read_text().strip()
                if product_uuid:
                    machine_identity += "\n" + product_uuid
            except OSError:
                pass
    hostname = hostname if hostname is not None else socket.gethostname()
    if not machine_identity:
        if os.name == "nt":  # Local test harness only; production runners use /proc.
            machine_identity = "windows-test:" + hostname
        else:
            raise ValueError("无法读取稳定的远端机器身份；需要 /etc/machine-id 或 DMI UUID，不能在共享 HOME 中安全建立训练队列")
    return hashlib.sha256((machine_identity + "\n" + hostname).encode()).hexdigest()[:32]


HOST_NAMESPACE = stable_host_namespace()
STATE_DIR = Path.home() / ".racktop" / "modelgym-training" / "hosts" / HOST_NAMESPACE
SCRIPT = Path(__file__).resolve()
STABLE_SECONDS = 30
POLL_SECONDS = 2
CANCEL_GRACE_SECONDS = 5
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)
ACTIVE = {"queued", "stabilizing", "running"}
TERMINAL = {"completed", "failed", "cancelled"}
SHELL_OPERATORS = {";", "&&", "||", "|", "&", ">", ">>", "<", "2>", "2>&1"}


def now():
    return time.time()


def bootstrap():
    os.umask(0o077)
    for directory in (STATE_DIR, STATE_DIR / "jobs"):
        if directory.is_symlink():
            raise ValueError("训练队列目录不能是符号链接")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)


@contextlib.contextmanager
def lock(name="state.lock", blocking=True):
    bootstrap()
    handle = open(STATE_DIR / name, "a+b")
    acquired = False
    try:
        if os.name == "nt":  # Enables deterministic unit tests on the Windows client.
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_id(value):
    if not isinstance(value, str):
        raise ValueError("任务标识必须为 UUID")
    try:
        if str(uuid.UUID(value)) != value.lower():
            raise ValueError()
    except (ValueError, AttributeError):
        raise ValueError("任务标识必须为标准 UUID")
    return value.lower()


def text(value, label, maximum=4096, allow_empty=False):
    if not isinstance(value, str) or len(value.encode("utf-8")) > maximum or any(c in value for c in "\0\r\n"):
        raise ValueError(label + "包含无效字符或过长")
    if not value.strip() and not allow_empty:
        raise ValueError(label + "不能为空")
    return value.strip()


def normalize_profile(value, require_gpus=False):
    if not isinstance(value, dict):
        raise ValueError("训练配置格式无效")
    profile = {
        "id": validate_id(value.get("id")),
        "name": text(value.get("name"), "配置名称", 240),
        "serverId": text(value.get("serverId"), "服务器标识", 200),
        "workingDirectory": text(value.get("workingDirectory"), "工作目录"),
        "interpreter": text(value.get("interpreter"), "启动前缀", 4096),
        "scriptPath": text(value.get("scriptPath"), "训练脚本路径"),
        "configPath": text(value.get("configPath"), "配置文件路径"),
        "configMode": value.get("configMode", "flag"),
        "configFlag": text(value.get("configFlag", "--config"), "配置参数名", 100, True),
        "extraArguments": text(value.get("extraArguments", ""), "附加参数", 16384, True),
    }
    if profile["configMode"] not in ("flag", "positional"):
        raise ValueError("配置文件传参方式无效")
    if profile["configMode"] == "flag" and not re.fullmatch(r"--?[A-Za-z0-9][A-Za-z0-9_-]*", profile["configFlag"]):
        raise ValueError("配置参数名必须为单个 -x 或 --config 形式的标记")
    devices = value.get("gpuUuids")
    if not isinstance(devices, list) or not (1 if require_gpus else 0) <= len(devices) <= 16 or any(not isinstance(item, str) for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("请选择同一服务器上 1 到 16 张不同的 GPU")
    if any(not isinstance(item, str) or not re.fullmatch(r"GPU-[A-Za-z0-9_-]+", item) for item in devices):
        raise ValueError("GPU UUID 无效；目前仅支持 NVIDIA 物理 GPU")
    profile["gpuUuids"] = devices
    minimum = value.get("minimumFreeMemoryGb", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or not 0 <= minimum <= 100000:
        raise ValueError("每卡最低空闲显存必须为有效非负数")
    profile["minimumFreeMemoryGb"] = minimum
    for field in ("createdAt", "updatedAt"):
        timestamp = value.get(field)
        if timestamp is not None and (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError("训练配置时间无效")
        profile[field] = timestamp if timestamp is not None else int(now())
    command_argv(profile)  # Validate quoting and unsupported shell operations early.
    return profile


def resolved_paths(profile):
    workdir = Path(os.path.expanduser(profile["workingDirectory"]))
    if not workdir.is_absolute():
        workdir = Path.home() / workdir
    workdir = workdir.resolve()
    def resolve(value):
        path = Path(os.path.expanduser(value))
        return str((path if path.is_absolute() else workdir / path).resolve())
    return str(workdir), resolve(profile["scriptPath"]), resolve(profile["configPath"])


def split_arguments(value, label):
    try:
        arguments = shlex.split(value, posix=True)
    except ValueError as error:
        raise ValueError(label + "引号未闭合：" + str(error))
    if any(argument in SHELL_OPERATORS for argument in arguments):
        raise ValueError(label + "不支持 shell 运算符；请直接填写命令参数")
    return arguments


def command_argv(profile):
    prefix = split_arguments(profile["interpreter"], "启动前缀")
    if not prefix:
        raise ValueError("启动前缀不能为空")
    if Path(prefix[0]).name in ("sh", "bash", "zsh", "dash", "fish", "nohup", "setsid"):
        raise ValueError("请使用 Python、conda run 或 torchrun 前缀；不支持后台包装或 shell 脚本前缀")
    prefix[0] = os.path.expanduser(prefix[0])
    workdir, script, config = resolved_paths(profile)
    # Popen resolves relative executable paths after changing cwd. Resolve them
    # here too, so preflight and the preview use that same project environment.
    if os.sep in prefix[0] or (os.altsep and os.altsep in prefix[0]):
        executable = Path(prefix[0])
        if not executable.is_absolute():
            prefix[0] = str((Path(workdir) / executable).resolve())
    return prefix + [script] + ([profile["configFlag"]] if profile["configMode"] == "flag" else []) + [config] + split_arguments(profile["extraArguments"], "附加参数")


def command_preview(arguments):
    return " ".join(shlex.quote(argument) for argument in arguments)


def probe_gpus(minimum=0):
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], "远端缺少 nvidia-smi；请先安装 NVIDIA 驱动"
    try:
        def query(fields):
            return subprocess.run([executable, fields, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True).stdout
        metrics = query("--query-gpu=uuid,index,name,utilization.gpu,memory.total,memory.used")
        applications = query("--query-compute-apps=gpu_uuid,pid")
        devices = []
        for row in csv.reader(io.StringIO(metrics)):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError("GPU 指标列数不完整")
            values = [item.strip() for item in row]
            utilization, total, used = map(float, values[3:])
            if not all(math.isfinite(item) for item in (utilization, total, used)) or total <= 0 or used < 0 or not 0 <= utilization <= 100:
                raise ValueError("GPU 指标不可读")
            index = int(values[1])
            if index < 0 or not re.fullmatch(r"GPU-[A-Za-z0-9_-]+", values[0]):
                raise ValueError("GPU 身份或卡号不可确认")
            devices.append({"uuid": values[0], "index": index, "name": values[2], "utilization": utilization, "memoryTotalMb": total, "memoryUsedMb": used, "computeProcessCount": 0})
        if not devices or len({gpu["uuid"] for gpu in devices}) != len(devices) or len({gpu["index"] for gpu in devices}) != len(devices):
            raise ValueError("未取得完整的 NVIDIA GPU 清单")
        by_uuid = {gpu["uuid"]: gpu for gpu in devices}
        for row in csv.reader(io.StringIO(applications)):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != 2 or row[0].strip() not in by_uuid or not row[1].strip().isdigit():
                raise ValueError("计算进程清单不可确认（包含不可映射的设备或 MIG 分区）")
            by_uuid[row[0].strip()]["computeProcessCount"] += 1
        for gpu in devices:
            gpu["reasons"] = []
            if gpu["computeProcessCount"]:
                gpu["reasons"].append("存在 %s 个计算进程" % gpu["computeProcessCount"])
            if gpu["utilization"] > 10:
                gpu["reasons"].append("利用率 %.1f%% 超过 10%%" % gpu["utilization"])
            free = (gpu["memoryTotalMb"] - gpu["memoryUsedMb"]) / 1024
            if free < minimum:
                gpu["reasons"].append("空闲显存 %.2f GB，要求 %.2f GB" % (free, minimum))
            gpu["eligible"] = not gpu["reasons"]
        return devices, None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return [], "无法确认 NVIDIA GPU 状态，继续等待：" + str(error)[:300]


def readiness(profile, devices, probe_error=None):
    if probe_error:
        return [probe_error]
    by_uuid = {gpu["uuid"]: gpu for gpu in devices}
    reasons = []
    for device in profile["gpuUuids"]:
        gpu = by_uuid.get(device)
        if gpu is None:
            reasons.append("所选 GPU 不可见：" + device)
            continue
        if gpu["computeProcessCount"]:
            reasons.append("GPU %s 存在计算进程" % gpu["index"])
        if gpu["utilization"] > 10:
            reasons.append("GPU %s 利用率超过 10%%" % gpu["index"])
        free = (gpu["memoryTotalMb"] - gpu["memoryUsedMb"]) / 1024
        if free < profile["minimumFreeMemoryGb"]:
            reasons.append("GPU %s 空闲显存 %.2f GB，要求 %.2f GB" % (gpu["index"], free, profile["minimumFreeMemoryGb"]))
    return reasons


def check_paths(profile):
    workdir, script, config = resolved_paths(profile)
    interpreter = command_argv(profile)[0]
    interpreter_available = bool(shutil.which(interpreter))
    result = {
        "workingDirectoryExists": os.path.isdir(workdir),
        "scriptExists": os.path.isfile(script) and os.access(script, os.R_OK),
        "configExists": os.path.isfile(config) and os.access(config, os.R_OK),
        "pythonAvailable": sys.version_info >= (3, 8),
        "nvidiaSmiAvailable": bool(shutil.which("nvidia-smi")),
        "interpreterAvailable": interpreter_available,
        "commandPreview": command_preview(command_argv(profile)),
    }
    issues = []
    for field, message in (("pythonAvailable", "远端需要 Python 3.8 或更新版本"), ("nvidiaSmiAvailable", "远端缺少 nvidia-smi"), ("workingDirectoryExists", "远端工作目录不存在"), ("scriptExists", "远端训练脚本不存在或不可读"), ("configExists", "远端配置文件不存在或不可读"), ("interpreterAvailable", "远端找不到启动程序；请使用 Python/Conda/torchrun 的绝对路径")):
        if not result[field]:
            issues.append(message)
    devices, probe_error = probe_gpus(profile["minimumFreeMemoryGb"]) if result["nvidiaSmiAvailable"] else ([], None)
    result.update({"gpus": devices, "ok": not issues and not probe_error, "message": "；".join(issues + ([probe_error] if probe_error else [])) or "脚本、配置、启动程序和 NVIDIA 驱动检查通过"})
    return result, issues, probe_error


def job_path(job_id):
    return STATE_DIR / "jobs" / validate_id(job_id) / "job.json"


def load_job(job_id):
    path = job_path(job_id)
    try:
        with open(path, encoding="utf-8") as handle:
            job = json.load(handle)
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError, OSError) as error:
        raise ValueError("训练任务状态损坏或不可读：%s；已暂停自动启动，请检查并恢复原记录，未覆盖任务数据" % path) from error
    try:
        if not isinstance(job, dict) or job.get("id") != validate_id(job_id):
            raise ValueError("任务标识与目录不一致")
        if job.get("status") not in ACTIVE | TERMINAL:
            raise ValueError("任务状态无效")
        profile = normalize_profile(job.get("profile"), require_gpus=True)
        if job.get("specHash") != spec_hash(profile) or job.get("gpuUuids") != profile["gpuUuids"] or job.get("serverId") != profile["serverId"]:
            raise ValueError("训练配置快照与任务不一致")
        for field in ("createdAt", "stableSince", "lastProbeAt", "startedAt", "endedAt", "cancelRequestedAt"):
            value = job.get(field)
            if field == "createdAt" and value is None:
                raise ValueError("任务创建时间缺失")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError("任务时间无效")
        if job.get("pid") is not None and (type(job["pid"]) is not int or job["pid"] <= 0):
            raise ValueError("任务 PID 无效")
        record = job.get("processIdentity")
        if record is not None and (not isinstance(record, dict) or record.get("pid") != job.get("pid") or not isinstance(record.get("startTicks"), str) or not record["startTicks"].isdigit() or not isinstance(record.get("bootId"), str)):
            raise ValueError("任务进程身份无效")
        return job
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("训练任务状态损坏：%s；已暂停自动启动，请检查并恢复原记录，未覆盖任务数据（%s）" % (path, error)) from error


def save_job(job):
    path = job_path(job["id"])
    path.parent.mkdir(mode=0o700, exist_ok=True)
    atomic_json(path, job)


def all_jobs(errors=None):
    jobs = []
    for path in (STATE_DIR / "jobs").glob("*/job.json"):
        try:
            job = load_job(path.parent.name)
        except ValueError as error:
            if errors is None:
                raise
            errors.append(str(error))
            continue
        if job is not None:
            jobs.append(job)
    return sorted(jobs, key=lambda job: (job["createdAt"], job["id"]))


def identity(pid):
    try:
        process = Path("/proc") / str(pid)
        content = (process / "stat").read_text()
        fields = content[content.rfind(")") + 2:].split()
        if process.stat().st_uid != os.getuid() or fields[0] == "Z":
            return None
        return {"pid": int(pid), "startTicks": fields[19], "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "hostNamespace": HOST_NAMESPACE}
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def identity_alive(expected):
    return bool(expected and identity(expected["pid"]) == expected)


def owned_group_members(job, exclude=None):
    members = []
    if not job.get("pid") or not Path("/proc").is_dir():
        return members
    marker = ("MODELGYM_JOB_ID=" + job["id"]).encode()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == exclude:
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].strip().split()
            if int(fields[2]) != job["pid"] or fields[0] == "Z":
                continue
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            record = identity(int(directory.name))
            if record:
                members.append(record)
        except (OSError, ValueError, IndexError):
            continue
    return members


def signal_job(job, value):
    if identity_alive(job.get("processIdentity")):
        try:
            os.killpg(job["pid"], value)
        except ProcessLookupError:
            pass
        return
    for member in owned_group_members(job):
        if identity_alive(member):
            try:
                os.kill(member["pid"], value)
            except ProcessLookupError:
                pass


def reconcile(job, timestamp):
    if job["status"] != "running":
        return job
    alive = identity_alive(job.get("processIdentity"))
    descendants = [] if alive else owned_group_members(job)
    if job.get("cancelRequestedAt") is not None and timestamp - job["cancelRequestedAt"] >= CANCEL_GRACE_SECONDS:
        signal_job(job, KILL_SIGNAL)
        alive = identity_alive(job.get("processIdentity"))
        descendants = owned_group_members(job) if not alive else []
    if not alive and not descendants:
        job.update({"status": "cancelled" if job.get("cancelRequestedAt") is not None else "failed", "endedAt": int(timestamp), "waitingReason": "任务已取消" if job.get("cancelRequestedAt") is not None else "训练进程中断或服务器重启；未自动重新运行"})
        save_job(job)
    return job


def spec_hash(profile):
    specification = {key: value for key, value in profile.items() if key not in ("createdAt", "updatedAt")}
    return hashlib.sha256(json.dumps(specification, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_daemon():
    record_path = STATE_DIR / "daemon.json"
    try:
        if identity_alive(json.loads(record_path.read_text())):
            return
    except (FileNotFoundError, ValueError, KeyError):
        pass
    with open(STATE_DIR / "daemon.log", "ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, str(SCRIPT), "daemon", HOST_NAMESPACE], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True)
    record = identity(process.pid)
    if not record:
        raise RuntimeError("远端队列守护进程启动失败；请检查 " + str(STATE_DIR / "daemon.log"))
    atomic_json(record_path, record)


def submit(job_id, profile, allow_queue):
    job_id = validate_id(job_id)
    with lock():
        existing = load_job(job_id)
        if existing:
            if existing["specHash"] != spec_hash(profile):
                raise ValueError("同一任务标识已绑定其他配置，不能修改后重复提交")
            if existing["status"] in ACTIVE:
                ensure_daemon()
            return {"job": public_job(existing), "requiresQueueConfirmation": False, "reasons": []}
        report, issues, probe_error = check_paths(profile)
        if issues:
            raise ValueError("；".join(issues))
        reasons = readiness(profile, report["gpus"], probe_error)
        jobs = all_jobs()
        for pending in jobs:
            reconcile(pending, now())
            if (pending["status"] in ACTIVE or identity_alive(pending.get("processIdentity"))) and set(pending["gpuUuids"]) & set(profile["gpuUuids"]):
                reasons.append("所选 GPU 已有 ModelGym 训练或更早提交的排队任务")
                break
        if reasons and not allow_queue:
            return {"job": None, "requiresQueueConfirmation": True, "reasons": reasons}
        if sum(job["status"] in ACTIVE for job in jobs) >= 200:
            raise ValueError("每台服务器最多保留 200 个未完成训练任务")
        timestamp = now()
        job = {"id": job_id, "profileId": profile["id"], "name": profile["name"], "serverId": profile["serverId"], "profile": profile, "gpuUuids": profile["gpuUuids"], "gpuIndices": [], "command": report["commandPreview"], "status": "queued" if reasons else "stabilizing", "waitingReason": "；".join(reasons) if reasons else "正在确认所有 GPU 连续空闲 30 秒", "createdAt": timestamp, "startedAt": None, "endedAt": None, "pid": None, "exitCode": None, "logPath": "~/.racktop/modelgym-training/hosts/" + HOST_NAMESPACE + "/jobs/" + job_id + "/output.log", "specHash": spec_hash(profile), "stableSince": None if reasons else timestamp, "lastProbeAt": timestamp, "processIdentity": None}
        save_job(job)
        # A lost SSH reply may leave the accepted job here. Retrying the same id
        # returns this job instead of submitting another training process.
        ensure_daemon()
        return {"job": public_job(job), "requiresQueueConfirmation": False, "reasons": reasons}


def public_job(job):
    return {key: value for key, value in job.items() if key not in ("specHash", "stableSince", "lastProbeAt", "processIdentity", "cancelRequestedAt")}


def launch(job, devices, timestamp):
    mapping = {gpu["uuid"]: gpu["index"] for gpu in devices}
    job["gpuIndices"] = [mapping[item] for item in job["gpuUuids"]]
    environment = os.environ.copy()
    environment["MODELGYM_JOB_ID"] = job["id"]
    output_path = job_path(job["id"]).parent / "output.log"
    with open(output_path, "ab", buffering=0) as output:
        process = subprocess.Popen([sys.executable, str(SCRIPT), "run", job["id"], HOST_NAMESPACE], stdin=subprocess.DEVNULL, stdout=output, stderr=output, env=environment, start_new_session=True, close_fds=True)
    process_identity = identity(process.pid)
    if not process_identity:
        raise RuntimeError("训练运行器未能启动")
    job.update({"status": "running", "pid": process.pid, "processIdentity": process_identity, "startedAt": int(timestamp), "waitingReason": ""})
    save_job(job)


def tick():
    with lock():
        timestamp = now()
        errors = []
        jobs = [reconcile(job, timestamp) for job in all_jobs(errors)]
        if errors:
            # Keep cancellation/exit reconciliation working, but never start a
            # new job while another task's reservation cannot be established.
            raise ValueError("；".join(errors))
        pending = [job for job in jobs if job["status"] in ("queued", "stabilizing")]
        if not pending:
            return
        devices, error = probe_gpus()
        reserved = {device for job in jobs if job["status"] == "running" or identity_alive(job.get("processIdentity")) for device in job["gpuUuids"]}
        for job in pending:
            overlap = set(job["gpuUuids"]) & reserved
            reasons = readiness(job["profile"], devices, error)
            if overlap:
                reasons.insert(0, "等待使用同一 GPU 的更早训练任务")
            # Earlier tasks retain priority on their entire fixed GPU set;
            # independent GPU sets may stabilize and run concurrently.
            reserved.update(job["gpuUuids"])
            if reasons:
                job.update({"status": "queued", "stableSince": None, "waitingReason": "；".join(reasons), "lastProbeAt": timestamp})
                save_job(job)
                continue
            if job.get("stableSince") is None or timestamp - job.get("lastProbeAt", 0) > POLL_SECONDS * 3 or timestamp < job.get("lastProbeAt", 0):
                job["stableSince"] = timestamp
            job.update({"lastProbeAt": timestamp, "status": "stabilizing", "waitingReason": "所有 GPU 已满足条件，确认稳定 %s / 30 秒" % min(30, int(timestamp - job["stableSince"]))})
            if timestamp - job["stableSince"] >= STABLE_SECONDS:
                try:
                    # Fresh GPU checks and process creation happen under the same
                    # queue lock; no other ModelGym submit/cancel can race them.
                    fresh, fresh_error = probe_gpus()
                    latest = readiness(job["profile"], fresh, fresh_error)
                    if latest:
                        job.update({"status": "queued", "stableSince": None, "waitingReason": "；".join(latest)})
                    else:
                        launch(job, fresh, timestamp)
                except Exception as failure:
                    job.update({"status": "failed", "endedAt": int(now()), "waitingReason": "训练运行器启动失败：" + str(failure)[:500]})
            save_job(job)


def run_job(job_id):
    cancelled = [False]
    def on_terminate(_signum, _frame):
        cancelled[0] = True
    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGINT, on_terminate)
    with lock():
        job = load_job(job_id)
        if not job or job["status"] != "running" or job.get("pid") != os.getpid() or not identity_alive(job.get("processIdentity")):
            return
        if job.get("cancelRequestedAt") is not None:
            job.update({"status": "cancelled", "endedAt": int(now()), "waitingReason": "启动前已取消"})
            save_job(job)
            return
        working_directory, script, config = resolved_paths(job["profile"])
        if not os.path.isdir(working_directory) or not os.path.isfile(script) or not os.path.isfile(config):
            job.update({"status": "failed", "endedAt": int(now()), "waitingReason": "发车前检查失败：工作目录、脚本或配置已不存在"})
            save_job(job)
            return
        environment = os.environ.copy()
        # NVML/nvidia-smi indices can differ from CUDA's enumeration order.
        # Complete UUIDs bind training to the exact devices whose state we checked.
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(job["gpuUuids"])
        environment["PYTHONUNBUFFERED"] = "1"
        # Hold the cancellation lock through actual process creation.
        try:
            process = subprocess.Popen(command_argv(job["profile"]), cwd=working_directory, env=environment, stdin=subprocess.DEVNULL, close_fds=True)
        except OSError as error:
            job.update({"status": "failed", "endedAt": int(now()), "waitingReason": "无法启动训练程序：" + str(error)[:500]})
            save_job(job)
            return
    code = process.wait()
    # A foreground launcher may finish before its children; keep GPU reservations
    # until the remaining, job-tagged process group members actually exit.
    while owned_group_members(job, exclude=os.getpid()):
        time.sleep(0.25)
    with lock():
        current = load_job(job_id)
        if current and current["status"] == "running" and current.get("pid") == os.getpid():
            was_cancelled = cancelled[0] or current.get("cancelRequestedAt") is not None
            current.update({"status": "cancelled" if was_cancelled else "completed" if code == 0 else "failed", "exitCode": code, "endedAt": int(now()), "waitingReason": "训练已取消" if was_cancelled else "" if code == 0 else "训练程序退出，代码 %s；未自动重跑" % code})
            save_job(current)


def cancel(job_id):
    with lock():
        job = load_job(job_id)
        if job is None:
            raise ValueError("训练任务不存在")
        if job["status"] in ("queued", "stabilizing"):
            job.update({"status": "cancelled", "endedAt": int(now()), "waitingReason": "排队任务已取消"})
            save_job(job)
        elif job["status"] == "running":
            reconcile(job, now())
            if job["status"] == "running":
                job.setdefault("cancelRequestedAt", now())
                job["waitingReason"] = "正在停止训练及其子进程"
                save_job(job)
                signal_job(job, signal.SIGTERM)
        return public_job(job)


def read_log(job_id, lines):
    if load_job(job_id) is None:
        raise ValueError("训练任务不存在")
    path = job_path(job_id).parent / "output.log"
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 512 * 1024))
            data = handle.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-max(20, min(int(lines), 2000)):])
    except FileNotFoundError:
        return "尚未启动训练，暂无日志。"


def dispatch(request):
    action = request.get("action")
    if action == "check_paths":
        return check_paths(normalize_profile(request.get("profile")))[0]
    if action == "submit":
        return submit(request.get("jobId"), normalize_profile(request.get("profile"), require_gpus=True), request.get("queueIfBusy") is True)
    if action == "list":
        with lock():
            return [public_job(reconcile(job, now())) for job in all_jobs()]
    if action == "cancel":
        return cancel(validate_id(request.get("jobId")))
    if action == "read_log":
        return read_log(validate_id(request.get("jobId")), request.get("lines", 400))
    raise ValueError("未知训练队列操作")


def daemon():
    try:
        with lock("daemon.lock", blocking=False):
            with lock():
                atomic_json(STATE_DIR / "daemon.json", identity(os.getpid()))
            while True:
                try:
                    tick()
                except Exception as error:
                    print("ModelGym queue:", error, file=sys.stderr, flush=True)
                time.sleep(POLL_SECONDS)
    except (BlockingIOError, OSError):
        return


def main():
    bootstrap()
    if len(sys.argv) == 3 and sys.argv[1] == "daemon":
        if sys.argv[2] != HOST_NAMESPACE:
            raise ValueError("训练守护进程机器身份不匹配")
        daemon()
    elif len(sys.argv) == 4 and sys.argv[1] == "run":
        if sys.argv[3] != HOST_NAMESPACE:
            raise ValueError("训练运行器机器身份不匹配，操作已阻止")
        run_job(validate_id(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "rpc":
        try:
            request = json.loads(base64.b64decode(sys.argv[2], validate=True))
            if not isinstance(request, dict):
                raise ValueError("训练队列请求必须为 JSON 对象")
            result = dispatch(request)
            print(json.dumps({"result": result}, ensure_ascii=False, allow_nan=False))
        except Exception as error:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
            sys.exit(1)
    else:
        raise ValueError("训练队列命令格式无效")


if __name__ == "__main__":
    main()

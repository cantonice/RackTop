"""Behavioral tests with real persistence/locking and a simulated nvidia-smi.

Run with Python on Windows or Linux. Windows tests do not claim Linux /proc,
process-group or SSH integration coverage; those boundaries use explicit mocks.
"""
import concurrent.futures
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

WORKER_PATH = Path(__file__).resolve().parents[1] / "assets" / "modelgym-training.py"
SPEC = importlib.util.spec_from_file_location("modelgym_training", WORKER_PATH)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)
REAL_RUN = subprocess.run
REAL_START_DATALIST = queue.start_datalist
REAL_ENSURE_DAEMON = queue.ensure_daemon


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="modelgym-training-tests-")
        self.root = Path(self.temporary.name)
        self.clock = 1000.0
        self.launches = []
        self.datalist_launches = []
        self.patches = []
        self.previous_signals = {value: signal.getsignal(value) for value in (signal.SIGTERM, signal.SIGINT)}
        self.patch("STATE_DIR", self.root / "state")
        self.patch("now", lambda: self.clock)
        self.ensure_daemon = self.patch("ensure_daemon", mock.Mock())
        self.patch("owned_group_members", lambda _job, exclude=None: [])
        self.patch("identity_alive", lambda record: bool(record and record.get("bootId") == "test-boot"))
        self.patch("launch", self.fake_launch)
        self.patch("start_datalist", self.fake_start_datalist)
        self.workdir = self.root / "project with spaces"
        self.workdir.mkdir()
        self.script = self.workdir / "train.py"
        self.script.write_text("import sys\nprint('training', sys.argv)\n", encoding="utf-8")
        self.config = self.workdir / "config.yaml"
        self.config.write_text("epochs: 2\n", encoding="utf-8")
        self.gpu_state = self.root / "gpu.json"
        self.gpu_state.write_text(json.dumps({"busy": [], "utilization": {}, "used": {}, "error": False}), encoding="utf-8")
        self.fake_smi = self.root / "fake_nvidia_smi.py"
        self.fake_smi.write_text("""import json, sys
state=json.load(open(sys.argv[1], encoding='utf-8'))
if state.get('error'): sys.exit(3)
if sys.argv[2].startswith('--query-gpu='):
    for index, name in enumerate(['GPU-a','GPU-b','GPU-c']):
        print('%s,%s,Fake GPU %s,%s,24576,%s' % (name,index,index,state.get('utilization',{}).get(name,0),state.get('used',{}).get(name,0)))
else:
    for name in state.get('busy',[]): print('%s,12345' % name)
""", encoding="utf-8")
        original_which = queue.shutil.which
        self.patches.append(mock.patch.object(queue.shutil, "which", side_effect=lambda name: "fake-nvidia-smi" if name == "nvidia-smi" else original_which(name)))
        self.patches[-1].start()
        def fake_run(arguments, **options):
            if arguments[0] != "fake-nvidia-smi":
                return REAL_RUN(arguments, **options)
            return REAL_RUN([sys.executable, str(self.fake_smi), str(self.gpu_state), arguments[1]], **options)
        self.patches.append(mock.patch.object(queue.subprocess, "run", side_effect=fake_run))
        self.patches[-1].start()
        queue.bootstrap()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        for value, handler in self.previous_signals.items():
            signal.signal(value, handler)
        self.temporary.cleanup()

    def patch(self, name, value):
        patch = mock.patch.object(queue, name, value)
        patch.start()
        self.patches.append(patch)
        return value

    def profile(self, devices=None):
        return queue.normalize_profile({"id": str(uuid.uuid4()), "name": "GPU training", "serverId": "server-a", "workingDirectory": str(self.workdir), "interpreter": shlex.quote(sys.executable) + " -u", "scriptPath": self.script.name, "configPath": self.config.name, "configMode": "flag", "configFlag": "--config", "extraArguments": "--epochs 2", "gpuUuids": devices if devices is not None else ["GPU-a"], "minimumFreeMemoryGb": 8}, require_gpus=bool(devices is None or devices))

    def test_missing_config_mode_defaults_to_positional_and_preserves_explicit_flag(self):
        profile = self.profile()
        self.assertEqual(profile["configMode"], "flag")
        del profile["configMode"]
        defaulted = queue.normalize_profile(profile)
        self.assertEqual(defaulted["configMode"], "positional")
        self.assertNotIn("--config", queue.command_argv(defaulted))
        del profile["minimumFreeMemoryGb"]
        self.assertEqual(queue.normalize_profile(profile)["minimumFreeMemoryGb"], 10)

    def state(self, **updates):
        value = json.loads(self.gpu_state.read_text(encoding="utf-8"))
        value.update(updates)
        self.gpu_state.write_text(json.dumps(value), encoding="utf-8")

    def fake_launch(self, job, devices, timestamp):
        self.launches.append(job["id"])
        mapping = {gpu["uuid"]: gpu["index"] for gpu in devices}
        job.update({"status": "running", "gpuIndices": [mapping[item] for item in job["gpuUuids"]], "pid": 9000 + len(self.launches), "startedAt": timestamp, "processIdentity": {"pid": 9000 + len(self.launches), "startTicks": "10", "bootId": "test-boot"}, "waitingReason": ""})
        queue.save_job(job)

    def fake_start_datalist(self, job, timestamp):
        if job.get("datalistStatus") != "pending" or job["status"] not in ("queued", "stabilizing"):
            return
        self.datalist_launches.append(job["id"])
        pid = 12000 + len(self.datalist_launches)
        job.update({"datalistStatus": "running", "datalistPid": pid, "datalistProcessIdentity": {"pid": pid, "startTicks": "10", "bootId": "test-boot"}, "datalistStartedAt": timestamp})
        queue.save_job(job)

    def datalist_profile(self, body=None):
        script = self.workdir / "generate datalist.py"
        script.write_text(body or "import json, os, sys\nfrom pathlib import Path\nPath('prepared.json').write_text(json.dumps({'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'args':sys.argv[1:],'cwd':os.getcwd()}))\n", encoding="utf-8")
        return dict(self.profile(), datalistScriptPath=script.name, datalistArguments="--label 'a b $(literal)'")

    def finish_datalist(self, job_id):
        job = queue.load_job(job_id)
        job.update({"datalistPid": os.getpid(), "datalistProcessIdentity": {"pid": os.getpid(), "startTicks": "10", "bootId": "test-boot"}})
        queue.save_job(job)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            queue.run_job(job_id, datalist=True)
        return output.getvalue()

    def test_datalist_runs_once_immediately_while_gpus_are_busy_then_training_waits(self):
        self.state(busy=["GPU-a"])
        profile, job_id = self.datalist_profile(), str(uuid.uuid4())
        rejected = queue.submit(job_id, profile, False)
        self.assertTrue(rejected["requiresQueueConfirmation"])
        self.assertFalse(self.datalist_launches)
        accepted = queue.submit(job_id, profile, True)["job"]
        self.assertEqual(self.datalist_launches, [job_id])
        self.assertEqual(accepted["datalistStatus"], "running")
        self.assertIsNone(accepted["startedAt"])
        output = self.finish_datalist(job_id)
        received = json.loads((self.workdir / "prepared.json").read_text())
        self.assertEqual(received["cuda"], "")
        self.assertEqual(received["args"], ["--label", "a b $(literal)"])
        self.assertEqual(Path(received["cwd"]).resolve(), self.workdir.resolve())
        self.assertIn("datalist 生成开始", output)
        self.assertIn("datalist 生成结束，退出代码 0", output)
        self.tick_for(32)
        self.assertFalse(self.launches)
        queue.submit(job_id, profile, True)
        self.assertEqual(self.datalist_launches, [job_id])
        self.state(busy=[])
        self.tick_for(32)
        self.assertEqual(self.launches, [job_id])

    def test_gpu_stability_is_measured_during_datalist_but_training_cannot_start_early(self):
        profile, job_id = self.datalist_profile(), str(uuid.uuid4())
        queue.submit(job_id, profile, True)
        self.tick_for(40)
        self.assertFalse(self.launches)
        self.assertEqual(queue.load_job(job_id)["stableSince"], 1000)
        self.assertIn("生成 datalist", queue.load_job(job_id)["waitingReason"])
        self.finish_datalist(job_id)
        self.state(busy=["GPU-a"])
        self.tick_for(2)
        self.assertFalse(self.launches)
        self.state(busy=[])
        self.tick_for(32)
        self.assertEqual(self.launches, [job_id])

    def test_datalist_failure_stops_the_job_and_never_retries_or_trains(self):
        profile, job_id = self.datalist_profile("raise SystemExit(7)\n"), str(uuid.uuid4())
        queue.submit(job_id, profile, True)
        self.finish_datalist(job_id)
        failed = queue.load_job(job_id)
        self.assertEqual((failed["status"], failed["datalistStatus"], failed["datalistExitCode"]), ("failed", "failed", 7))
        self.assertIn("未启动训练", failed["waitingReason"])
        self.tick_for(40)
        queue.submit(job_id, profile, True)
        self.assertFalse(self.launches)
        self.assertEqual(self.datalist_launches, [job_id])

    def test_datalist_submit_retries_and_pending_recovery_do_not_duplicate_preparation(self):
        profile, job_id = self.datalist_profile(), str(uuid.uuid4())
        with mock.patch.object(queue, "start_datalist"):
            queue.submit(job_id, profile, True)  # A crash before process creation.
        self.assertEqual(queue.load_job(job_id)["datalistStatus"], "pending")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: queue.submit(job_id, profile, True), range(2)))
        self.tick_for(8)
        self.assertEqual(self.datalist_launches, [job_id])
        self.assertFalse(self.launches)

    def test_cancelling_datalist_waits_for_runner_and_descendants_before_finishing(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.datalist_profile(), True)
        with mock.patch.object(queue, "signal_job") as signal_job:
            result = queue.cancel(job_id)
            self.assertIn(result["status"], ("queued", "stabilizing"))
            self.assertIn("正在停止 datalist", result["waitingReason"])
            self.assertEqual(signal_job.call_args.args[0]["pid"], 12001)
            self.assertEqual(signal_job.call_args.args[1], signal.SIGTERM)
            self.clock += 8
            queue.tick()
            self.assertEqual(signal_job.call_args.args[1], queue.KILL_SIGNAL)
            with mock.patch.object(queue, "identity_alive", return_value=False), mock.patch.object(queue, "owned_group_members", return_value=[{"pid": 12002}]):
                queue.tick()
                self.assertIn(queue.load_job(job_id)["status"], ("queued", "stabilizing"))
            with mock.patch.object(queue, "identity_alive", return_value=False):
                queue.tick()
        self.assertEqual(queue.load_job(job_id)["status"], "cancelled")
        self.assertFalse(self.launches)

    def test_interrupted_datalist_fails_closed_instead_of_running_it_again(self):
        profile, job_id = self.datalist_profile(), str(uuid.uuid4())
        queue.submit(job_id, profile, True)
        with mock.patch.object(queue, "identity_alive", return_value=False):
            queue.tick()
        self.assertEqual(queue.load_job(job_id)["status"], "failed")
        queue.submit(job_id, profile, True)
        self.assertEqual(self.datalist_launches, [job_id])
        self.assertFalse(self.launches)

    def test_cancel_preserves_an_already_interrupted_datalist_failure(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.datalist_profile(), True)
        with mock.patch.object(queue, "identity_alive", return_value=False):
            result = queue.cancel(job_id)
        self.assertEqual(result["status"], "failed")
        self.assertIn("进程中断", result["waitingReason"])

    def test_datalist_launcher_has_an_independent_session_and_hides_gpus(self):
        profile, job_id = self.datalist_profile(), str(uuid.uuid4())
        with mock.patch.object(queue, "start_datalist"):
            queue.submit(job_id, profile, True)
        job = queue.load_job(job_id)
        record = {"pid": 12099, "startTicks": "10", "bootId": "test-boot"}
        with mock.patch.object(queue.subprocess, "Popen", return_value=mock.Mock(pid=12099)) as popen, mock.patch.object(queue, "identity", return_value=record):
            with queue.lock():
                REAL_START_DATALIST(job, self.clock)
                REAL_START_DATALIST(job, self.clock)
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0][2], "datalist")
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(queue.load_job(job_id)["datalistPid"], 12099)

    def test_old_job_without_optional_fields_keeps_its_original_spec_hash(self):
        profile, job_id = self.profile(), str(uuid.uuid4())
        queue.submit(job_id, profile, True)
        job = queue.load_job(job_id)
        job["profile"].pop("datalistScriptPath")
        job["profile"].pop("datalistArguments")
        job.pop("datalistStatus")
        queue.save_job(job)
        self.assertEqual(queue.load_job(job_id)["specHash"], queue.spec_hash(profile))
        self.assertEqual(queue.submit(job_id, profile, True)["job"]["id"], job_id)

    def test_datalist_path_check_and_arguments_validation(self):
        profile = self.datalist_profile()
        report, issues, _ = queue.check_paths(profile)
        self.assertTrue(report["datalistScriptExists"])
        self.assertFalse(issues)
        profile["datalistScriptPath"] = "missing.py"
        report, issues, _ = queue.check_paths(profile)
        self.assertFalse(report["datalistScriptExists"])
        self.assertTrue(any("datalist" in issue for issue in issues))
        profile["datalistScriptPath"] = ""
        with self.assertRaisesRegex(ValueError, "同时填写"):
            queue.normalize_profile(profile)
        profile["datalistScriptPath"] = "prepare.py"
        profile["datalistArguments"] = "--out list.txt && touch bad"
        with self.assertRaisesRegex(ValueError, "shell"):
            queue.normalize_profile(profile)

    def test_daemon_upgrade_replaces_only_the_verified_daemon_pid(self):
        previous = {"pid": 14001, "startTicks": "10", "bootId": "test-boot"}
        replacement = {"pid": 14002, "startTicks": "11", "bootId": "test-boot"}
        queue.atomic_json(queue.STATE_DIR / "daemon.json", previous)
        alive = {14001: True, 14002: True}
        with mock.patch.object(queue, "identity_alive", side_effect=lambda record: bool(record and alive.get(record["pid"]))), mock.patch.object(queue.os, "kill", side_effect=lambda pid, _signal: alive.update({pid: False})) as kill, mock.patch.object(queue.os, "killpg", create=True) as killpg, mock.patch.object(queue.subprocess, "Popen", return_value=mock.Mock(pid=14002)) as popen, mock.patch.object(queue, "identity", return_value=replacement):
            with queue.lock():
                REAL_ENSURE_DAEMON()
                REAL_ENSURE_DAEMON()
        kill.assert_called_once_with(14001, signal.SIGTERM)
        killpg.assert_not_called()
        popen.assert_called_once()
        self.assertEqual(json.loads((queue.STATE_DIR / "daemon.json").read_text()), replacement)
        self.assertEqual(json.loads((queue.STATE_DIR / "daemon-version.json").read_text())["workerSha256"], queue.WORKER_VERSION)

    def test_daemon_upgrade_does_not_start_a_second_daemon_when_the_old_one_cannot_exit(self):
        previous = {"pid": 14001, "startTicks": "10", "bootId": "test-boot"}
        queue.atomic_json(queue.STATE_DIR / "daemon.json", previous)
        with mock.patch.object(queue, "identity_alive", return_value=True), mock.patch.object(queue.os, "kill") as kill, mock.patch.object(queue.time, "monotonic", side_effect=[0, 3]), mock.patch.object(queue.subprocess, "Popen") as popen:
            with queue.lock(), self.assertRaisesRegex(RuntimeError, "尚未退出"):
                REAL_ENSURE_DAEMON()
        kill.assert_called_once_with(14001, signal.SIGTERM)
        popen.assert_not_called()

    def tick_for(self, seconds):
        for _ in range(seconds // 2):
            self.clock += 2
            queue.tick()

    def test_busy_submit_requires_explicit_confirmation_without_a_remote_job_or_daemon(self):
        self.state(busy=["GPU-a"])
        job_id = str(uuid.uuid4())
        response = queue.submit(job_id, self.profile(), False)
        self.assertTrue(response["requiresQueueConfirmation"])
        self.assertIsNone(response["job"])
        self.assertIsNone(queue.load_job(job_id))
        self.ensure_daemon.assert_not_called()
        self.assertFalse(self.launches)

    def test_submit_retries_and_concurrent_duplicate_requests_create_exactly_one_job(self):
        profile, job_id = self.profile(), str(uuid.uuid4())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: queue.submit(job_id, profile, True), range(2)))
        self.assertEqual(len(queue.all_jobs()), 1)
        self.assertEqual([item["job"]["id"] for item in results], [job_id, job_id])
        profile["extraArguments"] = "--epochs 99"
        with self.assertRaisesRegex(ValueError, "同一任务标识"):
            queue.submit(job_id, profile, True)
        self.assertEqual(queue.load_job(job_id)["profile"]["extraArguments"], "--epochs 2")

    def test_all_selected_gpus_must_be_free_for_a_continuous_30_seconds(self):
        self.state(busy=["GPU-b"])
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(["GPU-a", "GPU-b"]), True)
        self.tick_for(30)
        self.assertFalse(self.launches)
        self.state(busy=[])
        self.tick_for(30)
        self.assertFalse(self.launches)  # First idle sample begins, not ends, the interval.
        self.tick_for(2)
        self.assertEqual(self.launches, [job_id])
        self.assertEqual(queue.load_job(job_id)["gpuIndices"], [0, 1])

    def test_any_compute_pid_including_current_user_blocks_training(self):
        self.state(busy=["GPU-a"], used={"GPU-a": 1})
        devices, error = queue.probe_gpus(8)
        self.assertIsNone(error)
        self.assertEqual(devices[0]["computeProcessCount"], 1)
        self.assertFalse(devices[0]["eligible"])
        self.assertTrue(queue.readiness(self.profile(), devices))

    def test_utilization_and_memory_thresholds_use_unrounded_values(self):
        self.state(utilization={"GPU-a": 10.1})
        self.assertTrue(queue.readiness(self.profile(), queue.probe_gpus()[0]))
        self.state(utilization={"GPU-a": 10}, used={"GPU-a": 16384.1})
        self.assertTrue(queue.readiness(self.profile(), queue.probe_gpus()[0]))
        self.state(used={"GPU-a": 16384})
        self.assertFalse(queue.readiness(self.profile(), queue.probe_gpus()[0]))

    def test_overlapping_gpu_jobs_are_fifo_while_disjoint_jobs_run_in_parallel(self):
        self.state(busy=["GPU-b"])
        first, second, independent = [str(uuid.uuid4()) for _ in range(3)]
        queue.submit(first, self.profile(["GPU-a", "GPU-b"]), True)
        self.clock += 0.1
        queue.submit(second, self.profile(["GPU-a"]), True)
        self.clock += 0.1
        queue.submit(independent, self.profile(["GPU-c"]), True)
        self.tick_for(32)
        self.assertEqual(self.launches, [independent])
        self.state(busy=[])
        self.tick_for(32)
        self.assertEqual(self.launches, [independent, first])
        self.assertEqual(queue.load_job(second)["status"], "queued")

    def test_probe_failure_resets_stability_and_never_launches(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        self.tick_for(20)
        self.state(error=True)
        self.tick_for(12)
        self.assertFalse(self.launches)
        self.assertEqual(queue.load_job(job_id)["status"], "queued")
        self.state(error=False)
        self.tick_for(30)
        self.assertFalse(self.launches)
        self.tick_for(2)
        self.assertEqual(self.launches, [job_id])

    def test_sampling_gap_does_not_count_as_proven_continuous_idle_time(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        self.clock += 120
        queue.tick()
        self.assertFalse(self.launches)
        self.assertEqual(queue.load_job(job_id)["stableSince"], self.clock)

    def test_queued_cancel_is_durable_and_cannot_later_launch(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        self.assertEqual(queue.cancel(job_id)["status"], "cancelled")
        self.assertEqual(queue.cancel(job_id)["status"], "cancelled")
        self.tick_for(40)
        self.assertFalse(self.launches)
        self.assertEqual(json.loads(queue.job_path(job_id).read_text())["status"], "cancelled")

    def test_read_only_check_and_list_never_start_a_daemon_or_training(self):
        queue.check_paths(self.profile([]))
        queue.dispatch({"action": "list"})
        self.ensure_daemon.assert_not_called()
        self.assertFalse(self.launches)

    def test_completed_runner_keeps_gpu_reserved_until_it_really_exits(self):
        first, second = str(uuid.uuid4()), str(uuid.uuid4())
        queue.submit(first, self.profile(), True)
        self.tick_for(30)
        job = queue.load_job(first)
        self.assertEqual(job["status"], "running")
        job["status"] = "completed"
        queue.save_job(job)
        queue.submit(second, self.profile(), True)
        self.tick_for(32)
        self.assertEqual(self.launches, [first])
        self.assertEqual(queue.load_job(second)["status"], "queued")

    def test_recycled_pid_is_not_signalled(self):
        job = {"id": str(uuid.uuid4()), "pid": 12345, "processIdentity": {"pid": 12345, "startTicks": "old", "bootId": "old-boot"}}
        with mock.patch.object(queue.os, "killpg", create=True) as killpg:
            queue.signal_job(job, signal.SIGTERM)
            killpg.assert_not_called()

    def test_running_cancel_keeps_reservation_until_process_exit(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        self.tick_for(30)
        with mock.patch.object(queue, "signal_job") as signal_job:
            result = queue.cancel(job_id)
            self.assertEqual(result["status"], "running")
            self.assertIn("正在停止", result["waitingReason"])
            signal_job.assert_called_once()
            self.assertEqual(signal_job.call_args.args[1], signal.SIGTERM)
            self.clock += 8
            queue.tick()
            self.assertEqual(queue.load_job(job_id)["status"], "running")
            self.assertEqual(signal_job.call_args.args[1], queue.KILL_SIGNAL)
        with mock.patch.object(queue, "identity_alive", return_value=False):
            queue.tick()
        self.assertEqual(queue.load_job(job_id)["status"], "cancelled")

    def test_real_runner_uses_literal_arguments_selected_gpus_and_working_directory(self):
        profile, job_id = self.profile(["GPU-a", "GPU-c"]), str(uuid.uuid4())
        output = self.workdir / "received.json"
        config = self.workdir / "config ' $(literal).yaml"
        config.write_text("training: true\n", encoding="utf-8")
        self.script.write_text("import json, os, sys\nfrom pathlib import Path\nPath('received.json').write_text(json.dumps({'args':sys.argv[1:],'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'cwd':os.getcwd()}))\n", encoding="utf-8")
        profile["configPath"] = str(config)
        profile["extraArguments"] = "--label 'a b $(literal)'"
        queue.submit(job_id, profile, True)
        job = queue.load_job(job_id)
        job.update({"status": "running", "pid": os.getpid(), "gpuIndices": [0, 2], "processIdentity": {"pid": os.getpid(), "startTicks": "10", "bootId": "test-boot"}})
        queue.save_job(job)
        queue.run_job(job_id)
        received = json.loads(output.read_text())
        self.assertEqual(received["args"], ["--config", str(config.resolve()), "--label", "a b $(literal)"])
        self.assertEqual(received["cuda"], "GPU-a,GPU-c")
        self.assertEqual(Path(received["cwd"]).resolve(), self.workdir.resolve())
        self.assertEqual(queue.load_job(job_id)["status"], "completed")
        self.assertEqual(queue.load_job(job_id)["exitCode"], 0)

    def test_cancel_before_actual_child_creation_prevents_training(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        job = queue.load_job(job_id)
        job.update({"status": "running", "pid": os.getpid(), "processIdentity": {"pid": os.getpid(), "startTicks": "10", "bootId": "test-boot"}, "cancelRequestedAt": self.clock})
        queue.save_job(job)
        with mock.patch.object(queue.subprocess, "Popen") as child:
            queue.run_job(job_id)
            child.assert_not_called()
        self.assertEqual(queue.load_job(job_id)["status"], "cancelled")

    def test_nested_script_keeps_project_cwd_and_logs_resolved_launch_context(self):
        tools = self.workdir / "tools"
        tools.mkdir()
        script = tools / "train.py"
        (self.workdir / "relative-data.txt").write_text("project data", encoding="utf-8")
        script.write_text("from pathlib import Path\nPath('received.txt').write_text(Path('relative-data.txt').read_text())\n", encoding="utf-8")
        profile, job_id = self.profile(), str(uuid.uuid4())
        profile["scriptPath"] = str(script)
        queue.submit(job_id, profile, True)
        job = queue.load_job(job_id)
        job.update({"status": "running", "pid": os.getpid(), "gpuIndices": [0], "processIdentity": {"pid": os.getpid(), "startTicks": "10", "bootId": "test-boot"}})
        queue.save_job(job)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            queue.run_job(job_id)
        self.assertEqual((self.workdir / "received.txt").read_text(), "project data")
        self.assertEqual(queue.load_job(job_id)["status"], "completed")
        self.assertIn(str(self.workdir.resolve()), log.getvalue())
        self.assertIn(str(script.resolve()), log.getvalue())
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-a", log.getvalue())

    def test_failed_training_is_recorded_without_automatic_retry(self):
        self.script.write_text("raise SystemExit(7)\n", encoding="utf-8")
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        job = queue.load_job(job_id)
        job.update({"status": "running", "pid": os.getpid(), "gpuIndices": [0], "processIdentity": {"pid": os.getpid(), "startTicks": "10", "bootId": "test-boot"}})
        queue.save_job(job)
        queue.run_job(job_id)
        self.assertEqual(queue.load_job(job_id)["status"], "failed")
        self.assertEqual(queue.load_job(job_id)["exitCode"], 7)
        self.tick_for(40)
        self.assertFalse(self.launches)

    def test_config_validation_rejects_shell_operators_bad_flags_and_control_characters(self):
        for field, value in (("interpreter", "python && touch bad"), ("extraArguments", "--a 1 | cat"), ("extraArguments", "--label 'unterminated"), ("configFlag", "--config;touch"), ("scriptPath", "bad\0.py")):
            profile = self.profile()
            profile[field] = value
            with self.assertRaises(ValueError, msg=field):
                queue.normalize_profile(profile, require_gpus=True)

    def test_relative_interpreter_uses_project_working_directory_in_preflight_and_execution(self):
        environment = self.workdir / ".venv"
        environment.mkdir()
        executable = environment / ("python.exe" if os.name == "nt" else "python")
        shutil.copy2(sys.executable, executable)
        profile = self.profile()
        profile["interpreter"] = shlex.quote(str(Path(".venv") / executable.name)) + " -u"
        report, issues, _ = queue.check_paths(profile)
        self.assertTrue(report["interpreterAvailable"])
        self.assertFalse(issues)
        self.assertEqual(queue.command_argv(profile)[0], str(executable.resolve()))

    def test_shared_home_queues_are_isolated_by_stable_execution_host_identity(self):
        shared = self.root / "shared-home" / ".racktop" / "modelgym-training" / "hosts"
        host_a = queue.stable_host_namespace("machine-a", "gpu-host")
        host_b = queue.stable_host_namespace("machine-b", "gpu-host")
        self.assertNotEqual(host_a, host_b)
        self.assertEqual(host_a, queue.stable_host_namespace("machine-a", "gpu-host"))
        job_id = str(uuid.uuid4())
        with mock.patch.object(queue, "STATE_DIR", shared / host_a):
            queue.submit(job_id, self.profile(["GPU-a"]), True)
        with mock.patch.object(queue, "STATE_DIR", shared / host_b):
            self.assertIsNone(queue.load_job(job_id))
            queue.submit(job_id, self.profile(["GPU-b"]), True)
            self.assertEqual(queue.load_job(job_id)["gpuUuids"], ["GPU-b"])
        with mock.patch.object(queue, "STATE_DIR", shared / host_a):
            self.assertEqual(queue.load_job(job_id)["gpuUuids"], ["GPU-a"])

    def test_damaged_record_fails_closed_without_overwriting_or_starting_other_jobs(self):
        valid_id, damaged_id = str(uuid.uuid4()), str(uuid.uuid4())
        queue.submit(valid_id, self.profile(), True)
        path = queue.job_path(damaged_id)
        path.parent.mkdir()
        path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已暂停自动启动"):
            queue.tick()
        self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
        self.assertFalse(self.launches)
        self.assertEqual(queue.load_job(valid_id)["status"], "stabilizing")
        # An operator repairs/removes the damaged fixture; valid jobs can resume.
        path.unlink()
        self.tick_for(32)
        self.assertEqual(self.launches, [valid_id])

    def test_duplicate_submit_never_replaces_a_damaged_accepted_record(self):
        job_id, profile = str(uuid.uuid4()), self.profile()
        queue.submit(job_id, profile, True)
        path = queue.job_path(job_id)
        path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "状态损坏"):
            queue.submit(job_id, profile, True)
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")
        self.assertFalse(self.launches)

    def test_damaged_neighbor_does_not_block_escalating_a_running_cancellation(self):
        job_id = str(uuid.uuid4())
        queue.submit(job_id, self.profile(), True)
        self.tick_for(30)
        job = queue.load_job(job_id)
        self.assertEqual(job["status"], "running")
        job["cancelRequestedAt"] = self.clock - queue.CANCEL_GRACE_SECONDS
        queue.save_job(job)
        damaged_path = queue.job_path(str(uuid.uuid4()))
        damaged_path.parent.mkdir()
        damaged_path.write_text("{broken", encoding="utf-8")
        with mock.patch.object(queue, "signal_job") as signal_job:
            with self.assertRaisesRegex(ValueError, "已暂停自动启动"):
                queue.tick()
            signal_job.assert_called_once()
            self.assertEqual(signal_job.call_args.args[1], queue.KILL_SIGNAL)


if __name__ == "__main__":
    unittest.main()

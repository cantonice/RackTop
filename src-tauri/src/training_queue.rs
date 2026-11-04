//! ModelGym's durable, remote training queue and a small local configuration cache.
//! The existing RackTop database/credential identity is intentionally unchanged.
use crate::{collector, models::Server, storage::Database};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    path::Path,
    process::Stdio,
    sync::Mutex,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::State;
use tokio::{io::AsyncWriteExt, time::timeout};

const WORKER: &str = include_str!("../assets/modelgym-training.py");

fn default_interpreter() -> String {
    "python -u".into()
}
fn default_config_mode() -> String {
    "positional".into()
}
fn default_config_flag() -> String {
    "--config".into()
}
fn default_minimum_free_memory_gb() -> f64 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingProfile {
    pub id: String,
    pub name: String,
    pub server_id: String,
    pub working_directory: String,
    #[serde(default = "default_interpreter")]
    pub interpreter: String,
    #[serde(default)]
    pub datalist_script_path: String,
    #[serde(default)]
    pub datalist_arguments: String,
    pub script_path: String,
    pub config_path: String,
    #[serde(default = "default_config_mode")]
    pub config_mode: String,
    #[serde(default = "default_config_flag")]
    pub config_flag: String,
    #[serde(default)]
    pub extra_arguments: String,
    pub gpu_uuids: Vec<String>,
    #[serde(default = "default_minimum_free_memory_gb")]
    pub minimum_free_memory_gb: f64,
    #[serde(default)]
    pub created_at: Option<f64>,
    #[serde(default)]
    pub updated_at: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingGpu {
    pub uuid: String,
    pub index: u32,
    pub name: String,
    pub utilization: f64,
    pub memory_total_mb: f64,
    pub memory_used_mb: f64,
    pub compute_process_count: u32,
    pub eligible: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPathCheck {
    pub ok: bool,
    pub working_directory_exists: bool,
    pub script_exists: bool,
    pub config_exists: bool,
    #[serde(default)]
    pub datalist_script_exists: Option<bool>,
    pub python_available: bool,
    pub nvidia_smi_available: bool,
    pub interpreter_available: bool,
    pub gpus: Vec<TrainingGpu>,
    pub command_preview: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingJob {
    pub id: String,
    pub profile_id: String,
    pub name: String,
    pub server_id: String,
    pub profile: TrainingProfile,
    pub gpu_uuids: Vec<String>,
    pub gpu_indices: Vec<u32>,
    pub command: String,
    pub status: String,
    #[serde(default)]
    pub datalist_status: Option<String>,
    pub waiting_reason: String,
    pub created_at: f64,
    pub started_at: Option<f64>,
    pub ended_at: Option<f64>,
    pub pid: Option<u32>,
    pub exit_code: Option<i32>,
    pub log_path: String,
    #[serde(default)]
    pub connection_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingSubmission {
    pub job: Option<TrainingJob>,
    pub requires_queue_confirmation: bool,
    pub reasons: Vec<String>,
}

pub struct TrainingStore {
    connection: Mutex<Connection>,
}

impl TrainingStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let connection = Connection::open(path).map_err(|error| error.to_string())?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|error| error.to_string())?;
        connection.execute_batch("PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS training_profiles(id TEXT PRIMARY KEY,server_id TEXT NOT NULL,payload TEXT NOT NULL,updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS training_servers(server_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS training_jobs(server_id TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(server_id,id));")
            .map_err(|error| error.to_string())?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    fn list_profiles(&self) -> Result<Vec<TrainingProfile>, String> {
        let connection = self.connection.lock().map_err(|error| error.to_string())?;
        let mut statement = connection
            .prepare("SELECT payload FROM training_profiles ORDER BY updated_at DESC,id")
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        rows.map(|row| {
            serde_json::from_str(&row.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect()
    }

    fn save_profile(&self, mut profile: TrainingProfile) -> Result<TrainingProfile, String> {
        validate_profile(&profile, false)?;
        let timestamp = unix_now();
        profile.name = profile.name.trim().to_string();
        profile.created_at = Some(profile.created_at.unwrap_or(timestamp));
        profile.updated_at = Some(timestamp);
        let payload = serde_json::to_string(&profile).map_err(|error| error.to_string())?;
        let connection = self.connection.lock().map_err(|error| error.to_string())?;
        connection.execute("INSERT INTO training_profiles(id,server_id,payload,updated_at) VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET server_id=excluded.server_id,payload=excluded.payload,updated_at=excluded.updated_at", params![profile.id, profile.server_id, payload, timestamp]).map_err(|error| error.to_string())?;
        connection
            .execute(
                "INSERT OR IGNORE INTO training_servers(server_id) VALUES(?1)",
                [&profile.server_id],
            )
            .map_err(|error| error.to_string())?;
        Ok(profile)
    }

    fn delete_profile(&self, id: &str) -> Result<(), String> {
        validate_id(id)?;
        self.connection
            .lock()
            .map_err(|error| error.to_string())?
            .execute("DELETE FROM training_profiles WHERE id=?1", [id])
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn remember_server(&self, id: &str) -> Result<(), String> {
        self.connection
            .lock()
            .map_err(|error| error.to_string())?
            .execute(
                "INSERT OR IGNORE INTO training_servers(server_id) VALUES(?1)",
                [id],
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn server_ids(&self) -> Result<Vec<String>, String> {
        let connection = self.connection.lock().map_err(|error| error.to_string())?;
        let mut statement = connection
            .prepare("SELECT server_id FROM training_servers ORDER BY server_id")
            .map_err(|error| error.to_string())?;
        statement
            .query_map([], |row| row.get(0))
            .map_err(|error| error.to_string())?
            .collect::<Result<_, _>>()
            .map_err(|error| error.to_string())
    }

    fn cache_job(&self, job: &TrainingJob) -> Result<(), String> {
        validate_id(&job.id)?;
        let payload = serde_json::to_string(job).map_err(|error| error.to_string())?;
        self.connection.lock().map_err(|error| error.to_string())?.execute("INSERT INTO training_jobs(server_id,id,payload,created_at) VALUES(?1,?2,?3,?4) ON CONFLICT(server_id,id) DO UPDATE SET payload=excluded.payload,created_at=excluded.created_at", params![job.server_id, job.id, payload, job.created_at]).map_err(|error| error.to_string())?;
        Ok(())
    }

    fn cached_jobs(&self, server_id: &str) -> Result<Vec<TrainingJob>, String> {
        let connection = self.connection.lock().map_err(|error| error.to_string())?;
        let mut statement = connection
            .prepare(
                "SELECT payload FROM training_jobs WHERE server_id=?1 ORDER BY created_at DESC,id",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([server_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        rows.map(|row| {
            serde_json::from_str(&row.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect()
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn validate_id(value: &str) -> Result<(), String> {
    let parsed = uuid::Uuid::parse_str(value).map_err(|_| "训练标识必须为 UUID".to_string())?;
    if parsed.hyphenated().to_string() != value.to_ascii_lowercase() {
        return Err("训练标识必须为标准 UUID".into());
    }
    Ok(())
}

fn validate_text(
    value: &str,
    label: &str,
    maximum: usize,
    allow_empty: bool,
) -> Result<(), String> {
    if value.len() > maximum
        || value
            .chars()
            .any(|character| matches!(character, '\0' | '\n' | '\r'))
        || (!allow_empty && value.trim().is_empty())
    {
        return Err(format!("{label}为空、包含无效字符或过长"));
    }
    Ok(())
}

fn validate_profile(profile: &TrainingProfile, require_gpus: bool) -> Result<(), String> {
    validate_id(&profile.id)?;
    for (value, label, maximum, allow_empty) in [
        (profile.name.as_str(), "配置名称", 240, false),
        (&profile.server_id, "服务器标识", 200, false),
        (&profile.working_directory, "工作目录", 4096, false),
        (&profile.interpreter, "启动前缀", 4096, false),
        (&profile.script_path, "训练脚本路径", 4096, false),
        (&profile.config_path, "配置文件路径", 4096, false),
        (&profile.config_flag, "配置参数名", 100, true),
        (&profile.extra_arguments, "附加参数", 16384, true),
        (&profile.datalist_script_path, "datalist 生成脚本路径", 4096, true),
        (&profile.datalist_arguments, "datalist 生成脚本参数", 16384, true),
    ] {
        validate_text(value, label, maximum, allow_empty)?;
    }
    if profile.datalist_script_path.trim().is_empty() && !profile.datalist_arguments.trim().is_empty() {
        return Err("请填写 datalist 生成脚本，或清空生成脚本参数".into());
    }
    if !matches!(profile.config_mode.as_str(), "flag" | "positional") {
        return Err("配置文件传参方式无效".into());
    }
    if profile.config_mode == "flag" {
        let flag = profile.config_flag.trim_start_matches('-');
        if !profile.config_flag.starts_with('-')
            || flag.is_empty()
            || profile.config_flag.starts_with("---")
            || !flag
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err("配置参数名必须为单个 -x 或 --config 形式的标记".into());
        }
    }
    if (require_gpus && profile.gpu_uuids.is_empty())
        || profile.gpu_uuids.len() > 16
        || profile.gpu_uuids.iter().collect::<BTreeSet<_>>().len() != profile.gpu_uuids.len()
    {
        return Err("请选择同一服务器上 1 到 16 张不同的 GPU".into());
    }
    for device in &profile.gpu_uuids {
        if !device.starts_with("GPU-")
            || device.len() <= 4
            || device.len() > 100
            || !device
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err("GPU UUID 无效；目前仅支持 NVIDIA 物理 GPU".into());
        }
    }
    if !profile.minimum_free_memory_gb.is_finite()
        || !(0.0..=100_000.0).contains(&profile.minimum_free_memory_gb)
    {
        return Err("每卡最低空闲显存必须为有效非负数".into());
    }
    if [profile.created_at, profile.updated_at]
        .into_iter()
        .flatten()
        .any(|timestamp| !timestamp.is_finite() || timestamp < 0.0)
    {
        return Err("训练配置时间无效".into());
    }
    Ok(())
}

fn password(
    database: &Database,
    server: &Server,
    allow_prompt: bool,
) -> Result<Option<String>, String> {
    if server.auth_method == "password" {
        database.get_password(&server.id, allow_prompt)
    } else {
        Ok(None)
    }
}

fn remote_script(request: &Value) -> Result<String, String> {
    let deployment_nonce = uuid::Uuid::new_v4();
    let worker = WORKER.replace("\r\n", "\n");
    let digest = format!("{:x}", Sha256::digest(worker.as_bytes()));
    let encoded_worker = STANDARD.encode(worker);
    let encoded_request =
        STANDARD.encode(serde_json::to_vec(request).map_err(|error| error.to_string())?);
    Ok(format!(
        r#"set -eu
umask 077
command -v python3 >/dev/null 2>&1 || {{ printf '%s\n' '{{"error":"远端缺少 python3；ModelGym 训练队列需要 Python 3.8 或更新版本"}}'; exit 41; }}
state="$HOME/.racktop/modelgym-training"
if [ -L "$state" ]; then printf '%s\n' '{{"error":"远端训练队列目录不能是符号链接"}}'; exit 42; fi
mkdir -p "$state"
chmod 700 "$state"
if [ ! -r "$state/worker.sha256" ] || [ "$(cat "$state/worker.sha256")" != '{digest}' ] || [ ! -r "$state/worker.py" ]; then
  temporary="$state/.worker.{deployment_nonce}"
  trap 'rm -f "$temporary"' EXIT HUP INT TERM
  printf '%s' '{encoded_worker}' | base64 -d > "$temporary"
  chmod 700 "$temporary"
  mv -f "$temporary" "$state/worker.py"
  printf '%s' '{digest}' > "$state/worker.sha256"
fi
python3 "$state/worker.py" rpc '{encoded_request}'
"#
    ))
}

async fn remote<T: DeserializeOwned>(
    server: &Server,
    password: Option<&str>,
    request: Value,
) -> Result<T, String> {
    remote_with_timeout(server, password, request, Duration::from_secs(60)).await
}

async fn remote_with_timeout<T: DeserializeOwned>(
    server: &Server,
    password: Option<&str>,
    request: Value,
    duration: Duration,
) -> Result<T, String> {
    let script = remote_script(&request)?;
    let (mut command, target) = collector::configured_ssh_command(server, password)?;
    command.kill_on_drop(true);
    // Pipe the bundled script through stdin: a Windows process command line must
    // not carry the worker's base64 payload (the Windows limit is ~32 KiB).
    command
        .arg(target)
        .arg("sh -s")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let operation = async {
        let mut child = command
            .spawn()
            .map_err(|error| format!("无法启动系统 ssh：{error}"))?;
        let mut stdin = child.stdin.take().ok_or("无法打开 SSH 输入通道")?;
        stdin
            .write_all(script.as_bytes())
            .await
            .map_err(|error| format!("无法发送训练队列请求：{error}"))?;
        stdin
            .shutdown()
            .await
            .map_err(|error| format!("无法完成训练队列请求：{error}"))?;
        drop(stdin);
        let output = child
            .wait_with_output()
            .await
            .map_err(|error| format!("训练队列 SSH 请求失败：{error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let response = stdout
            .lines()
            .rev()
            .find_map(|line| serde_json::from_str::<Value>(line).ok());
        if let Some(response) = response {
            if let Some(error) = response.get("error").and_then(Value::as_str) {
                return Err(error.to_string());
            }
            if output.status.success() {
                let result = response.get("result").ok_or("远端训练队列未返回结果")?;
                return serde_json::from_value(result.clone())
                    .map_err(|error| format!("远端训练队列响应格式无效：{error}"));
            }
        }
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(if stderr.is_empty() {
            "远端训练队列未返回确认；提交请求可能已接收，请使用同一任务标识重试".into()
        } else {
            collector::classify_ssh_error(&stderr)
        })
    };
    timeout(duration, operation).await.map_err(|_| {
        "训练队列请求超时；请重连查询或使用同一任务标识重试，避免重复训练".to_string()
    })?
}

#[tauri::command]
pub fn training_list_profiles(
    training: State<'_, TrainingStore>,
) -> Result<Vec<TrainingProfile>, String> {
    training.list_profiles()
}

#[tauri::command]
pub fn training_save_profile(
    database: State<'_, Database>,
    training: State<'_, TrainingStore>,
    profile: TrainingProfile,
) -> Result<TrainingProfile, String> {
    database.get_server(&profile.server_id)?;
    training.save_profile(profile)
}

#[tauri::command]
pub fn training_delete_profile(
    training: State<'_, TrainingStore>,
    profile_id: String,
) -> Result<(), String> {
    training.delete_profile(&profile_id)
}

#[tauri::command]
pub async fn training_check_paths(
    database: State<'_, Database>,
    profile: TrainingProfile,
) -> Result<TrainingPathCheck, String> {
    validate_profile(&profile, false)?;
    let server = database.get_server(&profile.server_id)?;
    let credential = password(&database, &server, true)?;
    remote(
        &server,
        credential.as_deref(),
        json!({"action":"check_paths","profile":profile}),
    )
    .await
}

#[tauri::command]
pub async fn training_submit(
    database: State<'_, Database>,
    training: State<'_, TrainingStore>,
    job_id: String,
    profile: TrainingProfile,
    queue_if_busy: bool,
) -> Result<TrainingSubmission, String> {
    validate_id(&job_id)?;
    validate_profile(&profile, true)?;
    let server = database.get_server(&profile.server_id)?;
    let credential = password(&database, &server, true)?;
    // Remember the endpoint before SSH, so a lost submit acknowledgement can be
    // reconciled by listing the remote queue after the application restarts.
    training.remember_server(&server.id)?;
    let response: TrainingSubmission = remote(
        &server,
        credential.as_deref(),
        json!({"action":"submit","jobId":job_id,"profile":profile,"queueIfBusy":queue_if_busy}),
    )
    .await?;
    if let Some(job) = &response.job {
        training.cache_job(job)?;
    }
    Ok(response)
}

#[tauri::command]
pub async fn training_list_jobs(
    database: State<'_, Database>,
    training: State<'_, TrainingStore>,
    server_id: Option<String>,
) -> Result<Vec<TrainingJob>, String> {
    let servers = match server_id {
        Some(id) => vec![id],
        None => training.server_ids()?,
    };
    let mut jobs = Vec::new();
    let mut connection_errors = Vec::new();
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut requests = tokio::task::JoinSet::new();
    // Bounded parallel polling avoids one offline endpoint serially stalling all
    // training cards. Credential failures never retry interactive prompts.
    for id in servers {
        let server = match database.get_server(&id) {
            Ok(server) => server,
            Err(_) => continue,
        };
        let credential = password(&database, &server, false);
        let permits = semaphore.clone();
        requests.spawn(async move {
            let result: Result<Vec<TrainingJob>, String> = match credential {
                Ok(credential) => match permits.acquire_owned().await {
                    Ok(_permit) => {
                        remote_with_timeout(
                            &server,
                            credential.as_deref(),
                            json!({"action":"list"}),
                            Duration::from_secs(15),
                        )
                        .await
                    }
                    Err(error) => Err(error.to_string()),
                },
                Err(error) => Err(error),
            };
            (id, result)
        });
    }
    while let Some(completed) = requests.join_next().await {
        let (id, result) = completed.map_err(|error| format!("训练队列状态查询失败：{error}"))?;
        match result {
            Ok(remote_jobs) => {
                for job in remote_jobs {
                    training.cache_job(&job)?;
                    jobs.push(job);
                }
            }
            Err(error) => {
                connection_errors.push(format!("{id}: {error}"));
                for mut job in training.cached_jobs(&id)? {
                    job.connection_error = Some(error.clone());
                    jobs.push(job);
                }
            }
        }
    }
    if jobs.is_empty() && !connection_errors.is_empty() {
        return Err(format!("无法读取远端训练队列：{}", connection_errors.join("；")));
    }
    jobs.sort_by(|left, right| {
        right
            .created_at
            .total_cmp(&left.created_at)
            .then(left.id.cmp(&right.id))
    });
    Ok(jobs)
}

#[tauri::command]
pub async fn training_cancel_job(
    database: State<'_, Database>,
    training: State<'_, TrainingStore>,
    server_id: String,
    job_id: String,
) -> Result<TrainingJob, String> {
    validate_id(&job_id)?;
    let server = database.get_server(&server_id)?;
    let credential = password(&database, &server, true)?;
    let job: TrainingJob = remote(
        &server,
        credential.as_deref(),
        json!({"action":"cancel","jobId":job_id}),
    )
    .await?;
    training.cache_job(&job)?;
    Ok(job)
}

#[tauri::command]
pub async fn training_read_log(
    database: State<'_, Database>,
    server_id: String,
    job_id: String,
    lines: Option<u32>,
) -> Result<String, String> {
    validate_id(&job_id)?;
    let server = database.get_server(&server_id)?;
    let credential = password(&database, &server, true)?;
    remote(
        &server,
        credential.as_deref(),
        json!({"action":"read_log","jobId":job_id,"lines":lines.unwrap_or(400).clamp(20,2000)}),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> TrainingProfile {
        serde_json::from_value(json!({"id":"12345678-1234-4234-8234-123456789abc","name":"训练","serverId":"s1","workingDirectory":"/data/train","interpreter":"python -u","scriptPath":"train.py","configPath":"config.yaml","configMode":"flag","configFlag":"--config","extraArguments":"","gpuUuids":["GPU-a"],"minimumFreeMemoryGb":8})).unwrap()
    }
    #[test]
    fn missing_config_mode_defaults_to_positional_and_explicit_flag_is_preserved() {
        let mut value = serde_json::to_value(profile()).unwrap();
        value.as_object_mut().unwrap().remove("configMode");
        let defaulted: TrainingProfile = serde_json::from_value(value).unwrap();
        assert_eq!(defaulted.config_mode, "positional");
        assert_eq!(profile().config_mode, "flag");
    }
    #[test]
    fn datalist_profile_defaults_and_fields_survive_storage() {
        let mut configured = profile();
        assert!(configured.datalist_script_path.is_empty());
        assert!(configured.datalist_arguments.is_empty());
        configured.datalist_script_path = "tools/create list.py".into();
        configured.datalist_arguments = "--output 'data/train list.json'".into();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("modelgym-training.sqlite");
        TrainingStore::open(&path).unwrap().save_profile(configured.clone()).unwrap();
        let restored = TrainingStore::open(&path).unwrap().list_profiles().unwrap().remove(0);
        assert_eq!(restored.datalist_script_path, configured.datalist_script_path);
        assert_eq!(restored.datalist_arguments, configured.datalist_arguments);
    }
    #[test]
    fn datalist_paths_and_arguments_are_validated() {
        let mut configured = profile();
        configured.datalist_arguments = "--output list.json".into();
        assert!(validate_profile(&configured, true).unwrap_err().contains("生成脚本"));
        configured.datalist_script_path = "tools/list.py".into();
        assert!(validate_profile(&configured, true).is_ok());
        for mutate in [
            |profile: &mut TrainingProfile| profile.datalist_script_path.push('\0'),
            |profile: &mut TrainingProfile| profile.datalist_script_path.push('\n'),
            |profile: &mut TrainingProfile| profile.datalist_arguments.push('\r'),
            |profile: &mut TrainingProfile| profile.datalist_script_path = "x".repeat(4097),
            |profile: &mut TrainingProfile| profile.datalist_arguments = "x".repeat(16385),
        ] {
            let mut invalid = configured.clone();
            mutate(&mut invalid);
            assert!(validate_profile(&invalid, true).is_err());
        }
    }
    #[test]
    fn local_profiles_and_remote_cache_survive_reopen_without_touching_main_database() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("modelgym-training.sqlite");
        let saved = TrainingStore::open(&path)
            .unwrap()
            .save_profile(profile())
            .unwrap();
        assert!(saved.created_at.is_some());
        let store = TrainingStore::open(&path).unwrap();
        assert_eq!(store.list_profiles().unwrap()[0].name, "训练");
        assert_eq!(store.server_ids().unwrap(), vec!["s1"]);
        store.delete_profile(&saved.id).unwrap();
        assert!(store.list_profiles().unwrap().is_empty());
        assert!(!directory.path().join("racktop.sqlite").exists());
    }
    #[test]
    fn invalid_gpu_sets_paths_and_command_flags_are_rejected() {
        for mutate in [
            |profile: &mut TrainingProfile| profile.gpu_uuids.push("GPU-a".into()),
            |profile: &mut TrainingProfile| profile.script_path.push('\0'),
            |profile: &mut TrainingProfile| profile.interpreter.push('\n'),
            |profile: &mut TrainingProfile| profile.config_flag = "--config;bad".into(),
            |profile: &mut TrainingProfile| profile.minimum_free_memory_gb = f64::NAN,
        ] {
            let mut invalid = profile();
            mutate(&mut invalid);
            assert!(validate_profile(&invalid, true).is_err());
        }
        assert!(validate_profile(&profile(), true).is_ok());
    }
    #[test]
    fn ssh_payload_uses_stdin_and_preserves_non_ascii_and_shell_metacharacters_as_data() {
        let request =
            json!({"action":"check_paths","profile":{"scriptPath":"路径/it's $(touch bad).py"}});
        let script = remote_script(&request).unwrap();
        let encoded = STANDARD.encode(serde_json::to_vec(&request).unwrap());
        assert!(script.contains(&format!("rpc '{encoded}'")));
        assert!(!script.contains("$(touch bad)"));
        assert!(script.contains("umask 077"));
        assert!(script.contains("worker.sha256"));
    }
}

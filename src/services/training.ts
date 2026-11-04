import { invoke } from '@tauri-apps/api/core'
import { api } from './api'
import type { Server, Snapshot } from '../types/models'
import type { TrainingGpu, TrainingJob, TrainingPathCheck, TrainingProfile, TrainingSubmitResult } from '../types/training'

const PROFILE_KEY = 'modelgym.trainingProfiles.v1'
const JOB_KEY = 'modelgym.trainingJobs.preview.v1'

type TrainingServer = Pick<Server, 'host' | 'name' | 'username' | 'sshAlias'>

export function defaultTrainingInterpreter(server?: TrainingServer): string {
  if (!server?.username.trim()) return 'python -u'
  const host = server.host.trim()
  const ipv4 = /^(?:\d{1,3}\.){3}(\d{1,3})$/.exec(host)
  const number = ipv4 ? Number(ipv4[1]) : [host, server.sshAlias ?? '', server.name]
    .map((value) => /(?:^|[-_\s]|服务器|训练机|server|gpu)(20|21|22|23|26|27|28)(?=$|[-_\s.()（）]|号)/i.exec(value)?.[1])
    .find(Boolean)
  const environment = [20, 21, 23].includes(Number(number)) ? 'torch29-cu130'
    : [22, 26, 27, 28].includes(Number(number)) ? 'deploy' : null
  if (!environment) return 'python -u'
  const path = `/home/${server.username.trim()}/miniconda3/envs/${environment}/bin/python`
  return /^[A-Za-z0-9_./-]+$/.test(path) ? path : shellQuote(path)
}

export function trainingServerChange(profile: TrainingProfile, previousServer: Server | undefined, nextServer: Server | undefined): Pick<TrainingProfile, 'serverId' | 'gpuUuids' | 'interpreter'> {
  const interpreter = profile.interpreter.trim()
  return {
    serverId: nextServer?.id ?? '',
    gpuUuids: [],
    interpreter: !interpreter || interpreter === defaultTrainingInterpreter(previousServer)
      ? defaultTrainingInterpreter(nextServer) : profile.interpreter,
  }
}

function readPreview<T>(key: string): T[] {
  if (typeof localStorage === 'undefined') return []
  try { const value: unknown = JSON.parse(localStorage.getItem(key) ?? '[]'); return Array.isArray(value) ? value as T[] : [] } catch { return [] }
}

function writePreview<T>(key: string, items: T[]) {
  if (typeof localStorage !== 'undefined') localStorage.setItem(key, JSON.stringify(items))
}

export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'"'"'`)}'`
}

export function trainingCommandPreview(profile: TrainingProfile): string {
  const config = profile.configMode === 'positional'
    ? shellQuote(profile.configPath.trim())
    : `${profile.configFlag.trim()} ${shellQuote(profile.configPath.trim())}`
  const command = [profile.interpreter.trim(), shellQuote(profile.scriptPath.trim()), config, profile.extraArguments.trim()].filter(Boolean).join(' ')
  return `cd -- ${shellQuote(profile.workingDirectory.trim())}\nCUDA_VISIBLE_DEVICES=${shellQuote(profile.gpuUuids.join(','))} ${command}`
}

export function validateTrainingProfile(profile: TrainingProfile, requireGpu = true): string | null {
  if (!profile.name.trim()) return '请填写训练名称。'
  if (!profile.serverId) return '请选择训练服务器。'
  if (!profile.workingDirectory.trim()) return '请填写服务器工作目录。'
  if (!profile.interpreter.trim()) return '请填写 Python、Conda 或 torchrun 启动前缀。'
  if (!profile.scriptPath.trim()) return '请填写服务器上的训练脚本路径。'
  if (!profile.configPath.trim()) return '请填写服务器上的配置文件路径。'
  if (profile.configMode === 'flag' && !/^--?[A-Za-z][A-Za-z0-9_-]*$/.test(profile.configFlag.trim())) return '配置参数名应为 --config、--cfg 或 -c 这样的参数。'
  if (!Number.isFinite(profile.minimumFreeMemoryGb) || profile.minimumFreeMemoryGb < 0) return '最低空闲显存必须是大于或等于 0 的数字。'
  if (requireGpu && profile.gpuUuids.length === 0) return '请至少选择一张 GPU。'
  if (new Set(profile.gpuUuids).size !== profile.gpuUuids.length) return 'GPU 选择重复，请重新选择。'
  if ([profile.name, profile.workingDirectory, profile.interpreter, profile.scriptPath, profile.configPath, profile.extraArguments].some((value) => value.includes('\0'))) return '训练配置不能包含空字符。'
  if ([profile.interpreter, profile.extraArguments].some((value) => /[\r\n]/.test(value))) return '启动前缀和附加参数请写在同一行，不支持 shell 脚本。'
  return null
}

export function trainingGpuFromSnapshot(snapshot: Snapshot, minimumFreeMemoryGb: number): TrainingGpu[] {
  return snapshot.gpus.map((gpu) => {
    const computeProcessCount = snapshot.processes.filter((process) => process.gpuUuid === gpu.uuid).length
    const reasons: string[] = []
    if (gpu.uuid.startsWith('unavailable-') || gpu.memoryTotalMb <= 0 || !Number.isFinite(gpu.utilization)) reasons.push('GPU 指标不可读')
    if (!snapshot.processesSampled) reasons.push('等待完整进程采样')
    if (computeProcessCount > 0) reasons.push(`${computeProcessCount} 个计算进程`)
    if (gpu.utilization > 10) reasons.push(`利用率 ${Math.round(gpu.utilization)}% > 10%`)
    if ((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024 < minimumFreeMemoryGb) reasons.push(`空闲显存不足 ${minimumFreeMemoryGb} GB`)
    return { ...gpu, computeProcessCount, eligible: reasons.length === 0, reasons }
  })
}

export function trainingSnapshotIssue(server: Server | undefined, snapshot: Snapshot | undefined, checkedAt?: number): string | null {
  if (!server) return '请选择训练服务器。'
  if (server.status === 'offline' || snapshot?.status === 'offline') return '服务器离线，无法提交新训练。'
  if (checkedAt !== undefined && (!snapshot || checkedAt >= snapshot.timestamp) && Math.floor(Date.now() / 1000) - checkedAt <= Math.max(30, server.samplingIntervalSeconds * 6)) return null
  if (!snapshot) return '等待服务器 GPU 数据，或点击“检查路径与环境”取得最新结果。'
  if (snapshot.acceleratorVendor && snapshot.acceleratorVendor !== 'nvidia') return '当前训练队列支持 NVIDIA GPU。'
  if (Math.floor(Date.now() / 1000) - snapshot.timestamp > Math.max(30, server.samplingIntervalSeconds * 6)) return 'GPU 快照已过期，提交前将重新检查服务器。'
  return null
}

export const trainingApi = {
  async listProfiles(): Promise<TrainingProfile[]> {
    return api.isDesktop ? invoke('training_list_profiles') : readPreview(PROFILE_KEY)
  },
  async saveProfile(profile: TrainingProfile): Promise<TrainingProfile> {
    if (api.isDesktop) return invoke('training_save_profile', { profile })
    const saved = { ...profile, updatedAt: Math.floor(Date.now() / 1000) }
    writePreview(PROFILE_KEY, [saved, ...readPreview<TrainingProfile>(PROFILE_KEY).filter((item) => item.id !== profile.id)])
    return saved
  },
  async deleteProfile(profileId: string): Promise<void> {
    if (api.isDesktop) return invoke('training_delete_profile', { profileId })
    writePreview(PROFILE_KEY, readPreview<TrainingProfile>(PROFILE_KEY).filter((item) => item.id !== profileId))
  },
  async checkPaths(profile: TrainingProfile): Promise<TrainingPathCheck> {
    if (api.isDesktop) return invoke('training_check_paths', { profile })
    const snapshots = await api.listLatestSnapshots()
    const snapshot = snapshots.find((item) => item.serverId === profile.serverId)
    const gpus = snapshot ? trainingGpuFromSnapshot(snapshot, profile.minimumFreeMemoryGb) : []
    return { ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus, commandPreview: trainingCommandPreview(profile), message: '浏览器演示：路径与环境检查已模拟通过，未连接真实服务器。' }
  },
  async submit(jobId: string, profile: TrainingProfile, queueIfBusy: boolean): Promise<TrainingSubmitResult> {
    if (api.isDesktop) return invoke('training_submit', { jobId, profile, queueIfBusy })
    const existing = readPreview<TrainingJob>(JOB_KEY).find((job) => job.id === jobId)
    if (existing) return { job: existing, requiresQueueConfirmation: false, reasons: [] }
    const check = await this.checkPaths(profile)
    const selected = check.gpus.filter((gpu) => profile.gpuUuids.includes(gpu.uuid))
    const reasons = selected.flatMap((gpu) => gpu.reasons.map((reason) => `GPU ${gpu.index}：${reason}`))
    if (selected.length !== profile.gpuUuids.length) reasons.push('所选 GPU 不在当前快照中')
    if (reasons.length && !queueIfBusy) return { job: null, requiresQueueConfirmation: true, reasons }
    const now = Math.floor(Date.now() / 1000)
    const job: TrainingJob = { id: jobId, profileId: profile.id, profile: structuredClone(profile), name: profile.name, serverId: profile.serverId, gpuUuids: [...profile.gpuUuids], gpuIndices: selected.map((gpu) => gpu.index), command: check.commandPreview, status: reasons.length ? 'queued' : 'stabilizing', waitingReason: reasons.length ? reasons.join('；') : '浏览器演示：等待连续空闲 30 秒，不会实际启动训练。', createdAt: now, pid: null, logPath: `~/.racktop/modelgym-training/jobs/${jobId}/output.log` }
    writePreview(JOB_KEY, [job, ...readPreview<TrainingJob>(JOB_KEY)])
    return { job, requiresQueueConfirmation: false, reasons: [] }
  },
  async listJobs(serverId: string | null = null): Promise<TrainingJob[]> {
    if (api.isDesktop) return invoke('training_list_jobs', { serverId })
    return readPreview<TrainingJob>(JOB_KEY).filter((job) => !serverId || job.serverId === serverId)
  },
  async cancelJob(serverId: string, jobId: string): Promise<TrainingJob> {
    if (api.isDesktop) return invoke('training_cancel_job', { serverId, jobId })
    const jobs = readPreview<TrainingJob>(JOB_KEY)
    const found = jobs.find((job) => job.id === jobId && job.serverId === serverId)
    if (!found) throw new Error('训练任务不存在。')
    const cancelled: TrainingJob = { ...found, status: 'cancelled', waitingReason: '用户取消', endedAt: Math.floor(Date.now() / 1000) }
    writePreview(JOB_KEY, jobs.map((job) => job.id === jobId && job.serverId === serverId ? cancelled : job))
    return cancelled
  },
  async readLog(serverId: string, jobId: string, lines = 400): Promise<string> {
    if (api.isDesktop) return invoke('training_read_log', { serverId, jobId, lines })
    return '[ModelGym 浏览器演示]\n任务仅用于界面预览，未执行任何真实训练。'
  },
}

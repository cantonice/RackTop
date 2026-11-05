import { invoke } from '@tauri-apps/api/core'
import { api } from './api'
import type { Server, Snapshot } from '../types/models'
import { isActiveTrainingJob } from '../types/training'
import type { TrainingDeleteJobsResult, TrainingGpu, TrainingJob, TrainingPathCheck, TrainingProfile, TrainingSubmitResult } from '../types/training'

const PROFILE_KEY = 'modelgym.trainingProfiles.v1'
const JOB_KEY = 'modelgym.trainingJobs.preview.v1'
const DELETED_JOB_KEY = 'modelgym.trainingJobs.deleted.preview.v1'

export const DEFAULT_TRAINING_PROJECT_COLOR = '#3b82f6'
const PROJECT_COLORS = [DEFAULT_TRAINING_PROJECT_COLOR, '#8b5cf6', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#f97316', '#6366f1']
const PROJECT_COLOR_PATTERN = /^#[\da-f]{6}$/i

function projectName(profile: TrainingProfile): string { return profile.projectName?.trim() ?? '' }
function projectColor(profile: TrainingProfile): string {
  return projectName(profile) && PROJECT_COLOR_PATTERN.test(profile.projectColor ?? '') ? profile.projectColor!.toLowerCase() : DEFAULT_TRAINING_PROJECT_COLOR
}

export function trainingProjectIndex(profiles: TrainingProfile[]): { name: string; color: string; count: number }[] {
  const groups = new Map<string, { name: string; color: string; count: number }>()
  for (const profile of profiles) {
    const name = projectName(profile)
    const group = groups.get(name)
    if (group) group.count += 1
    else groups.set(name, { name, color: projectColor(profile), count: 1 })
  }
  return [...groups.values()].sort((left, right) => !left.name ? 1 : !right.name ? -1 : left.name.localeCompare(right.name, 'zh-CN'))
}

export function trainingProjectColor(name: string, profiles: TrainingProfile[]): string {
  const normalized = name.trim()
  if (!normalized) return DEFAULT_TRAINING_PROJECT_COLOR
  const groups = trainingProjectIndex(profiles)
  const existing = groups.find((group) => group.name === normalized)
  if (existing) return existing.color
  const used = new Set(groups.filter((group) => group.name).map((group) => group.color))
  const preset = PROJECT_COLORS.find((color) => !used.has(color))
  if (preset) return preset
  // Continue assigning unused RGB colors after the quick-access palette fills.
  for (let index = 0; index <= used.size; index++) {
    const color = '#' + (((groups.length + index + 1) * 0x9e3779) & 0xffffff).toString(16).padStart(6, '0')
    if (!used.has(color)) return color
  }
  return DEFAULT_TRAINING_PROJECT_COLOR
}

function validateProject(profile: TrainingProfile): string | null {
  const name = profile.projectName ?? ''
  if (/[\0\r\n]/.test(name) || new TextEncoder().encode(name.trim()).byteLength > 120) return '项目名称不能包含换行、空字符，且不能超过 120 字节。'
  if (profile.projectColor !== undefined && !PROJECT_COLOR_PATTERN.test(profile.projectColor)) return '项目重点色必须为 #RRGGBB 格式。'
  return null
}

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

export function datalistCommandPreview(profile: TrainingProfile): string {
  const script = profile.datalistScriptPath?.trim()
  if (!script) return ''
  const command = [profile.interpreter.trim(), shellQuote(script), profile.datalistArguments?.trim()].filter(Boolean).join(' ')
  return `cd -- ${shellQuote(profile.workingDirectory.trim())}\nCUDA_VISIBLE_DEVICES='' ${command}`
}

export function validateTrainingProfile(profile: TrainingProfile, requireGpu = true): string | null {
  if (!profile.name.trim()) return '请填写训练名称。'
  const projectIssue = validateProject(profile)
  if (projectIssue) return projectIssue
  if (!profile.serverId) return '请选择训练服务器。'
  if (!profile.workingDirectory.trim()) return '请填写服务器工作目录。'
  if (!profile.interpreter.trim()) return '请填写 Python、Conda 或 torchrun 启动前缀。'
  if (!profile.scriptPath.trim()) return '请填写服务器上的训练脚本路径。'
  if (!profile.configPath.trim()) return '请填写服务器上的配置文件路径。'
  if (!profile.datalistScriptPath?.trim() && profile.datalistArguments?.trim()) return '请填写 datalist 生成脚本，或清空生成脚本参数。'
  if (new TextEncoder().encode(profile.datalistScriptPath ?? '').byteLength > 4096 || new TextEncoder().encode(profile.datalistArguments ?? '').byteLength > 16384) return 'datalist 生成脚本路径或参数过长。'
  if (/[\r\n]/.test(profile.datalistScriptPath ?? '')) return 'datalist 生成脚本路径不能包含换行。'
  if (profile.configMode === 'flag' && !/^--?[A-Za-z][A-Za-z0-9_-]*$/.test(profile.configFlag.trim())) return '配置参数名应为 --config、--cfg 或 -c 这样的参数。'
  if (!Number.isFinite(profile.minimumFreeMemoryGb) || profile.minimumFreeMemoryGb < 0) return '最低空闲显存必须是大于或等于 0 的数字。'
  if (requireGpu && profile.gpuUuids.length === 0) return '请至少选择一张 GPU。'
  if (new Set(profile.gpuUuids).size !== profile.gpuUuids.length) return 'GPU 选择重复，请重新选择。'
  if ([profile.name, profile.workingDirectory, profile.interpreter, profile.scriptPath, profile.configPath, profile.extraArguments, profile.datalistScriptPath ?? '', profile.datalistArguments ?? ''].some((value) => value.includes('\0'))) return '训练配置不能包含空字符。'
  if ([profile.interpreter, profile.extraArguments, profile.datalistArguments ?? ''].some((value) => /[\r\n]/.test(value))) return '启动前缀和附加参数请写在同一行，不支持 shell 脚本。'
  return null
}

export function trainingGpuFromSnapshot(snapshot: Snapshot, minimumFreeMemoryGb: number): TrainingGpu[] {
  return snapshot.gpus.map((gpu) => {
    const computeProcessCount = snapshot.processes.filter((process) => process.gpuUuid === gpu.uuid).length
    const reasons: string[] = []
    if (gpu.uuid.startsWith('unavailable-') || !Number.isFinite(gpu.memoryTotalMb) || gpu.memoryTotalMb <= 0 || !Number.isFinite(gpu.memoryUsedMb) || gpu.memoryUsedMb < 0 || gpu.memoryUsedMb > gpu.memoryTotalMb) reasons.push('GPU 显存指标不可读')
    if ((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024 <= minimumFreeMemoryGb) reasons.push(`空闲显存须大于 ${minimumFreeMemoryGb} GB`)
    return { ...gpu, computeProcessCount, eligible: reasons.length === 0, reasons }
  })
}

function trainingQueueReasons(gpu: TrainingGpu, minimum: number): string[] {
  const reasons: string[] = []
  if (gpu.uuid.startsWith('unavailable-') || !Number.isFinite(gpu.memoryTotalMb) || gpu.memoryTotalMb <= 0 || !Number.isFinite(gpu.memoryUsedMb) || !Number.isFinite(gpu.utilization)) reasons.push('GPU 指标不可读')
  if (gpu.computeProcessCount > 0) reasons.push('存在计算进程')
  if (gpu.utilization > 10) reasons.push('利用率超过 10%')
  if ((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024 < minimum) reasons.push(`空闲显存不足 ${minimum} GB`)
  return reasons
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
    const issue = validateProject(profile)
    if (issue) throw new Error(issue)
    const profiles = readPreview<TrainingProfile>(PROFILE_KEY)
    const name = projectName(profile)
    const color = name ? (profile.projectColor?.toLowerCase() ?? DEFAULT_TRAINING_PROJECT_COLOR) : DEFAULT_TRAINING_PROJECT_COLOR
    const saved = { ...profile, projectName: name, projectColor: color, updatedAt: Math.floor(Date.now() / 1000) }
    const remaining = profiles.filter((item) => item.id !== profile.id).map((item) => projectName(item) === name ? { ...item, projectName: name, projectColor: color } : item)
    writePreview(PROFILE_KEY, [saved, ...remaining])
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
    return { ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, datalistScriptExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus, commandPreview: trainingCommandPreview(profile), message: '浏览器演示：路径与环境检查已模拟通过，未连接真实服务器。' }
  },
  async submit(jobId: string, profile: TrainingProfile, queueIfBusy: boolean): Promise<TrainingSubmitResult> {
    if (api.isDesktop) return invoke('training_submit', { jobId, profile, queueIfBusy })
    if (readPreview<{ serverId: string; jobId: string }>(DELETED_JOB_KEY).some((item) => item.serverId === profile.serverId && item.jobId === jobId)) throw new Error('此训练任务记录已删除，请新建任务。')
    const existing = readPreview<TrainingJob>(JOB_KEY).find((job) => job.id === jobId && job.serverId === profile.serverId)
    if (existing) return { job: existing, requiresQueueConfirmation: false, reasons: [] }
    const check = await this.checkPaths(profile)
    if (readPreview<{ serverId: string; jobId: string }>(DELETED_JOB_KEY).some((item) => item.serverId === profile.serverId && item.jobId === jobId)) throw new Error('此训练任务记录已删除，请新建任务。')
    const selected = check.gpus.filter((gpu) => profile.gpuUuids.includes(gpu.uuid))
    const reasons = selected.flatMap((gpu) => trainingQueueReasons(gpu, profile.minimumFreeMemoryGb).map((reason) => `GPU ${gpu.index}：${reason}`))
    if (selected.length !== profile.gpuUuids.length) reasons.push('所选 GPU 不在当前快照中')
    if (reasons.length && !queueIfBusy) return { job: null, requiresQueueConfirmation: true, reasons }
    const now = Math.floor(Date.now() / 1000)
    const hasDatalist = Boolean(profile.datalistScriptPath?.trim())
    const waitingReason = hasDatalist ? `浏览器演示：datalist 脚本仅预览，未实际运行。${reasons.join('；')}` : reasons.length ? reasons.join('；') : '浏览器演示：等待连续空闲 30 秒，不会实际启动训练。'
    const job: TrainingJob = { id: jobId, profileId: profile.id, profile: structuredClone(profile), name: profile.name, serverId: profile.serverId, gpuUuids: [...profile.gpuUuids], gpuIndices: selected.map((gpu) => gpu.index), command: check.commandPreview, status: reasons.length || hasDatalist ? 'queued' : 'stabilizing', datalistStatus: hasDatalist ? 'pending' : null, waitingReason, createdAt: now, pid: null, logPath: `~/.racktop/modelgym-training/jobs/${jobId}/output.log` }
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
  async deleteJobs(serverId: string, jobIds: string[]): Promise<TrainingDeleteJobsResult> {
    if (api.isDesktop) return invoke('training_delete_jobs', { serverId, jobIds })
    if (jobIds.length > 200) throw new Error('每次最多删除 200 条任务记录。')
    const jobs = readPreview<TrainingJob>(JOB_KEY)
    const result: TrainingDeleteJobsResult = { deletedJobIds: [], failures: [] }
    for (const jobId of new Set(jobIds)) {
      const job = jobs.find((item) => item.serverId === serverId && item.id === jobId)
      if (job && (isActiveTrainingJob(job) || !['completed', 'failed', 'cancelled'].includes(job.status))) result.failures.push({ jobId, message: '仅可删除已结束的任务记录。' })
      else result.deletedJobIds.push(jobId)
    }
    const deleted = new Set(result.deletedJobIds)
    writePreview(JOB_KEY, jobs.filter((job) => job.serverId !== serverId || !deleted.has(job.id)))
    const tombstones = readPreview<{ serverId: string; jobId: string }>(DELETED_JOB_KEY)
    writePreview(DELETED_JOB_KEY, [...tombstones.filter((item) => item.serverId !== serverId || !deleted.has(item.jobId)), ...result.deletedJobIds.map((jobId) => ({ serverId, jobId }))])
    return result
  },
  async readLog(serverId: string, jobId: string, lines = 400): Promise<string> {
    if (api.isDesktop) return invoke('training_read_log', { serverId, jobId, lines })
    return '[ModelGym 浏览器演示]\n任务仅用于界面预览，未执行任何真实训练。'
  },
}

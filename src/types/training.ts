export interface TrainingProfile {
  id: string
  name: string
  serverId: string
  workingDirectory: string
  interpreter: string
  scriptPath: string
  configPath: string
  configMode: 'flag' | 'positional'
  configFlag: string
  extraArguments: string
  gpuUuids: string[]
  minimumFreeMemoryGb: number
  createdAt?: number
  updatedAt?: number
}

export interface TrainingGpu {
  uuid: string
  index: number
  name: string
  utilization: number
  memoryTotalMb: number
  memoryUsedMb: number
  computeProcessCount: number
  eligible: boolean
  reasons: string[]
}

export interface TrainingPathCheck {
  ok: boolean
  workingDirectoryExists: boolean
  scriptExists: boolean
  configExists: boolean
  pythonAvailable: boolean
  nvidiaSmiAvailable: boolean
  interpreterAvailable?: boolean
  gpus: TrainingGpu[]
  commandPreview: string
  message: string
}

export type TrainingJobStatus = 'queued' | 'stabilizing' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface TrainingJob {
  id: string
  profileId?: string | null
  profile: TrainingProfile
  name: string
  serverId: string
  gpuUuids: string[]
  gpuIndices: number[]
  command: string
  status: TrainingJobStatus
  waitingReason: string
  createdAt: number
  startedAt?: number | null
  endedAt?: number | null
  pid?: number | null
  exitCode?: number | null
  logPath?: string | null
  connectionError?: string | null
}

export interface TrainingSubmitResult {
  job: TrainingJob | null
  requiresQueueConfirmation: boolean
  reasons: string[]
}

export const TRAINING_STATUS_LABELS: Record<TrainingJobStatus, string> = {
  queued: '排队中',
  stabilizing: '空闲确认中',
  running: '训练中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

export function isActiveTrainingJob(job: TrainingJob): boolean {
  return job.status === 'queued' || job.status === 'stabilizing' || job.status === 'running'
}

import { AlertCircle, Server as ServerIcon } from 'lucide-react'
import { trainingGpuFromSnapshot, trainingSnapshotIssue } from '../services/training'
import type { Server, Snapshot } from '../types/models'
import type { TrainingGpu } from '../types/training'

interface Props {
  servers: Server[]
  snapshots: Record<string, Snapshot>
  serverId: string
  gpuUuids: string[]
  selectedServerGpus: TrainingGpu[]
  selectedGpuCheckedAt?: number
  minimumFreeMemoryGb: number
  onSelect: (serverId: string, gpuUuid: string, checked: boolean) => void
}

export function TrainingGpuPicker({ servers, snapshots, serverId, gpuUuids, selectedServerGpus, selectedGpuCheckedAt, minimumFreeMemoryGb, onSelect }: Props) {
  return <div className="training-fleet" aria-label="所有服务器 GPU 状态">
    {servers.map((server) => {
      const selected = server.id === serverId
      const snapshot = snapshots[server.id]
      const gpus = selected ? selectedServerGpus : snapshot ? trainingGpuFromSnapshot(snapshot, minimumFreeMemoryGb) : []
      const issue = trainingSnapshotIssue(server, snapshot, selected ? selectedGpuCheckedAt : undefined)
      return <section className={`training-fleet__server${selected ? ' is-selected' : ''}`} key={server.id} aria-label={`${server.name} GPU 状态`}>
        <header><div><ServerIcon size={13} /><strong>{server.name}</strong>{selected && <span>当前选择</span>}</div><small>{gpus.length} 张 GPU</small></header>
        <p className="training-fleet__address">{server.username}@{server.host}</p>
        {issue && <p className="training-snapshot-note"><AlertCircle size={13} />{issue}</p>}
        <div className="training-gpu-grid" aria-label={`${server.name} 选择训练 GPU`}>{gpus.map((gpu) => {
          const checked = selected && gpuUuids.includes(gpu.uuid)
          const available = gpu.eligible && !issue
          const readable = gpu.memoryTotalMb > 0 && Number.isFinite(gpu.utilization)
          return <label className={`training-gpu${checked ? ' is-selected' : ''}${!available ? ' is-busy' : ''}`} key={gpu.uuid}>
            <input type="checkbox" checked={checked} aria-label={`${server.name} GPU ${gpu.index} ${gpu.name}`} onChange={(event) => onSelect(server.id, gpu.uuid, event.target.checked)} />
            <div><strong>GPU {gpu.index}<span className={available ? 'is-free' : 'is-busy'}>{issue ? '待确认' : gpu.eligible ? '当前空闲' : '需等待'}</span></strong><small title={gpu.name}>{gpu.name}</small>
              <p>{readable ? <><b>{Math.max(0, (gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024).toFixed(1)}</b> / {(gpu.memoryTotalMb / 1024).toFixed(1)} GB 可用<em>利用率 {Math.round(gpu.utilization)}% · {gpu.computeProcessCount} 个计算进程</em></> : 'GPU 指标不可读'}</p>
              {gpu.reasons.length > 0 && <span className="training-gpu__reason">{gpu.reasons.join(' · ')}</span>}
            </div>
          </label>
        })}</div>
        {!gpus.length && <p className="training-fleet__empty">{snapshot?.acceleratorVendor && snapshot.acceleratorVendor !== 'nvidia' ? '此服务器暂不支持 NVIDIA GPU 训练。' : '暂无 GPU 数据，等待服务器监控刷新。'}</p>}
      </section>
    })}
    {!servers.length && <div className="training-gpu-empty"><ServerIcon size={21} /><p>先从左侧添加训练服务器，再回来配置训练。</p></div>}
  </div>
}

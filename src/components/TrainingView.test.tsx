// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { Server, Snapshot } from '../types/models'
import type { TrainingJob, TrainingProfile } from '../types/training'
import { trainingApi } from '../services/training'
import { TrainingView } from './TrainingView'

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true

const server: Server = { id: 'server', name: 'Training server', host: '10.0.0.1', port: 22, username: 'alice', tags: [], samplingIntervalSeconds: 2, historyRetentionDays: 90, remoteHistoryEnabled: false, authMethod: 'sshAgent', status: 'online' }
const gpu = { uuid: 'GPU-0', index: 0, name: 'RTX 4090', utilization: 90, memoryUtilization: 70, memoryUsedMb: 16 * 1024, memoryTotalMb: 24 * 1024, temperatureCelsius: 60, powerWatts: 250 }
const snapshot = { serverId: server.id, timestamp: Math.floor(Date.now() / 1000), status: 'online', acceleratorVendor: 'nvidia', gpus: [gpu, { ...gpu, uuid: 'GPU-1', index: 1, utilization: 0, memoryUsedMb: 0 }], processes: [{ gpuUuid: 'GPU-0', isCurrentUser: false }], processesSampled: true } as Snapshot
const profile: TrainingProfile = { id: 'profile', name: 'ResNet baseline', serverId: server.id, workingDirectory: '/workspace/resnet', interpreter: 'python -u', scriptPath: 'train.py', configPath: 'configs/train.yaml', configMode: 'flag', configFlag: '--config', extraArguments: '--epochs 20', gpuUuids: ['GPU-0'], minimumFreeMemoryGb: 8, createdAt: 1, updatedAt: 1 }
const job: TrainingJob = { id: 'job', profileId: profile.id, profile, name: profile.name, serverId: server.id, gpuUuids: ['GPU-0'], gpuIndices: [0], command: 'python train.py --config configs/train.yaml', status: 'queued', waitingReason: 'GPU 0 存在计算进程', createdAt: 1 }

let container: HTMLDivElement
let root: ReturnType<typeof createRoot>

function button(text: string, scope: ParentNode = container): HTMLButtonElement {
  const found = [...scope.querySelectorAll<HTMLButtonElement>('button')].find((element) => element.textContent?.trim() === text)
  if (!found) throw new Error(`Button not found: ${text}`)
  return found
}

async function render() {
  await act(async () => root.render(<TrainingView servers={[server]} snapshots={{ [server.id]: snapshot }} onNotice={vi.fn()} />))
}

async function openSavedProfile() {
  await act(async () => button('训练配置 1').click())
  await act(async () => button('配置并启动').click())
}

beforeEach(() => {
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
  vi.spyOn(trainingApi, 'listProfiles').mockResolvedValue([structuredClone(profile)])
  vi.spyOn(trainingApi, 'listJobs').mockResolvedValue([])
  vi.spyOn(trainingApi, 'saveProfile').mockImplementation(async (value) => value)
  vi.spyOn(trainingApi, 'cancelJob').mockResolvedValue({ ...job, status: 'cancelled' })
  vi.spyOn(trainingApi, 'readLog').mockResolvedValue('epoch=1 loss=0.52')
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.restoreAllMocks()
})

describe('TrainingView submission and queue confirmation', () => {
  it('indexes independent projects, filters profiles and synchronizes their accent color after saving', async () => {
    const profiles = [
      { ...profile, id: 'a', name: 'A baseline', projectName: 'Project A', projectColor: '#ef4444' },
      { ...profile, id: 'a2', name: 'A experiment', projectName: 'Project A', projectColor: '#ef4444' },
      { ...profile, id: 'b', name: 'B baseline', projectName: 'Project B', projectColor: '#10b981' },
      { ...profile, id: 'legacy', name: 'Legacy profile' },
    ]
    vi.mocked(trainingApi.listProfiles).mockResolvedValue(profiles)
    await render()
    await act(async () => button('训练配置 4').click())
    expect(container.querySelectorAll('.training-profile')).toHaveLength(4)
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="筛选项目：Project A"]')!.click())
    expect(container.querySelectorAll('.training-profile')).toHaveLength(2)
    expect(container.querySelector<HTMLElement>('.training-profile')!.style.getPropertyValue('--project-accent')).toBe('#ef4444')
    let resolveProfiles!: (value: TrainingProfile[]) => void
    vi.mocked(trainingApi.listProfiles).mockImplementationOnce(() => new Promise((done) => { resolveProfiles = done }))
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="刷新训练任务"]')!.click())
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="编辑 A baseline"]')!.click())
    const color = container.querySelector<HTMLInputElement>('input[aria-label="项目重点色"]')!
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(color, '#8b5cf6')
      color.dispatchEvent(new Event('input', { bubbles: true }))
      color.dispatchEvent(new Event('change', { bubbles: true }))
    })
    await act(async () => button('保存配置').click())
    await act(async () => resolveProfiles(profiles))
    await act(async () => button('取消').click())
    expect([...container.querySelectorAll<HTMLElement>('.training-profile')].map((card) => card.style.getPropertyValue('--project-accent'))).toEqual(['#8b5cf6', '#8b5cf6'])
    await act(async () => button('新建训练').click())
    expect(container.querySelector<HTMLInputElement>('input[placeholder="选择或填写项目名（可选）"]')?.value).toBe('Project A')
    expect(container.querySelector<HTMLInputElement>('input[aria-label="项目重点色"]')?.value).toBe('#8b5cf6')
  })

  it('only selects ended jobs, confirms cleanup and retains failed deletions for retry', async () => {
    const done = { ...job, id: 'done', name: 'Done', status: 'completed' as const }
    const failed = { ...job, id: 'failed', name: 'Failed', status: 'failed' as const }
    const cancelled = { ...job, id: 'cancelled', name: 'Cancelled', status: 'cancelled' as const }
    vi.mocked(trainingApi.listJobs).mockResolvedValue([done, failed, cancelled, { ...job, id: 'running', status: 'running' }])
    const remove = vi.spyOn(trainingApi, 'deleteJobs').mockResolvedValue({ deletedJobIds: [done.id], failures: [{ jobId: failed.id, message: '日志仍被占用' }] })
    const onDeleted = vi.fn()
    await act(async () => root.render(<TrainingView servers={[server]} snapshots={{}} onNotice={vi.fn()} onJobsDeleted={onDeleted} />))
    await act(async () => button('最近结束').click())
    expect(container.querySelectorAll('input[aria-label^="选择删除"]')).toHaveLength(3)
    await act(async () => container.querySelector<HTMLInputElement>('[aria-label="全选已结束任务"]')!.click())
    await act(async () => button('删除所选记录及日志').click())
    expect(remove).not.toHaveBeenCalled()
    expect(container.querySelector('[role="alertdialog"]')?.textContent).toContain('3 项已结束任务')
    await act(async () => button('保留记录').click())
    expect(remove).not.toHaveBeenCalled()
    await act(async () => container.querySelector<HTMLInputElement>('[aria-label^="选择删除 Cancelled"]')!.click())
    await act(async () => button('删除所选记录及日志').click())
    await act(async () => button('确认删除记录及日志').click())
    expect(remove).toHaveBeenCalledWith(server.id, [done.id, failed.id])
    expect(container.querySelector('[aria-label^="选择删除 Done"]')).toBeNull()
    expect(container.querySelector<HTMLInputElement>('[aria-label^="选择删除 Failed"]')?.checked).toBe(true)
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('日志仍被占用')
    expect(onDeleted).toHaveBeenCalledWith([{ serverId: server.id, jobId: done.id }])
  })

  it('keeps same-ID records on other servers and ignores late pre-deletion list responses', async () => {
    const other = { ...server, id: 'other', name: 'Other server' }
    const done = { ...job, name: 'Same record', status: 'completed' as const }
    const sibling = { ...done, serverId: other.id }
    vi.mocked(trainingApi.listJobs).mockResolvedValue([done, sibling])
    vi.spyOn(trainingApi, 'deleteJobs').mockResolvedValue({ deletedJobIds: [job.id], failures: [] })
    await act(async () => root.render(<TrainingView servers={[server, other]} snapshots={{}} onNotice={vi.fn()} />))
    await act(async () => button('最近结束').click())
    let resolve!: (jobs: TrainingJob[]) => void
    vi.mocked(trainingApi.listJobs).mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="刷新训练任务"]')!.click())
    await act(async () => container.querySelector<HTMLInputElement>('[aria-label="选择删除 Same record（Training server）"]')!.click())
    await act(async () => button('删除所选记录及日志').click())
    await act(async () => button('确认删除记录及日志').click())
    await act(async () => resolve([done, sibling]))
    expect(container.querySelector('[aria-label="选择删除 Same record（Training server）"]')).toBeNull()
    expect(container.querySelector('[aria-label="选择删除 Same record（Other server）"]')).not.toBeNull()
  })

  it('puts optional datalist generation first and saves its script, arguments and CPU command preview', async () => {
    await render(); await openSavedProfile()
    expect(container.querySelector('.training-sheet__body')?.firstElementChild?.getAttribute('aria-label')).toBe('生成 datalist 配置')
    const script = container.querySelector<HTMLInputElement>('input[placeholder="tools/generate_datalist.py（可选）"]')!
    const argumentsInput = container.querySelector<HTMLInputElement>('input[placeholder="--output data/train.txt"]')!
    expect(script.value).toBe('')
    expect(container.querySelector('[aria-label="datalist 生成命令"]')).toBeNull()
    await act(async () => {
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!
      setValue.call(script, 'tools/make_datalist.py'); script.dispatchEvent(new Event('input', { bubbles: true }))
      setValue.call(argumentsInput, '--output "data/train list.txt"'); argumentsInput.dispatchEvent(new Event('input', { bubbles: true }))
    })
    const preview = container.querySelector('[aria-label="datalist 生成命令"]')?.textContent ?? ''
    expect(preview).toContain('CUDA_VISIBLE_DEVICES=')
    expect(preview).toContain('tools/make_datalist.py')
    expect(preview).not.toContain('GPU-0')
    expect(preview).not.toContain('configs/train.yaml')
    await act(async () => button('保存配置').click())
    expect(trainingApi.saveProfile).toHaveBeenLastCalledWith(expect.objectContaining({ datalistScriptPath: 'tools/make_datalist.py', datalistArguments: '--output "data/train list.txt"' }))
  })

  it('shows datalist progress and explains cancellation also stops generation', async () => {
    const generatingJob = { ...job, profile: { ...profile, datalistScriptPath: 'tools/data.py' }, datalistStatus: 'running' as const }
    vi.mocked(trainingApi.listJobs).mockResolvedValue([generatingJob])
    await render()
    expect(container.textContent).toContain('正在生成 datalist')
    await act(async () => button('取消排队').click())
    expect(container.querySelector('[role="alertdialog"]')?.textContent).toContain('datalist 生成脚本及其子进程也会停止')
    expect(trainingApi.cancelJob).not.toHaveBeenCalled()
  })

  it('defaults new training to positional config and follows server Conda defaults', async () => {
    const training21 = { ...server, id: '21', name: '21', host: '192.168.0.21' }
    const training22 = { ...server, id: '22', name: '22', host: '192.168.0.22' }
    await act(async () => root.render(<TrainingView servers={[training21, training22]} snapshots={{}} onNotice={vi.fn()} />))
    await act(async () => button('新建训练').click())
    expect(container.querySelector<HTMLInputElement>('input[type="number"]')?.value).toBe('10')
    const prefix = container.querySelector<HTMLInputElement>('input[placeholder="python -u / conda run -n train python / torchrun"]')!
    expect(prefix.value).toBe('/home/alice/miniconda3/envs/torch29-cu130/bin/python')
    expect(container.querySelector<HTMLInputElement>('input[type="radio"]:checked')?.parentElement?.textContent).toContain('位置式')
    const select = container.querySelector<HTMLSelectElement>('select')!
    await act(async () => { select.value = training22.id; select.dispatchEvent(new Event('change', { bubbles: true })) })
    expect(prefix.value).toBe('/home/alice/miniconda3/envs/deploy/bin/python')
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(prefix, 'torchrun --nproc_per_node=2')
      prefix.dispatchEvent(new Event('input', { bubbles: true }))
    })
    await act(async () => { select.value = training21.id; select.dispatchEvent(new Event('change', { bubbles: true })) })
    expect(prefix.value).toBe('torchrun --nproc_per_node=2')
  })

  it('shows every server GPU and switches the selected server when choosing a GPU there', async () => {
    const other = { ...server, id: '22', name: '22', host: '192.168.0.22' }
    const offline = { ...server, id: 'offline', name: 'Offline server', status: 'offline' as const }
    const stale = { ...server, id: 'stale', name: 'Stale server' }
    const allSnapshots = {
      [server.id]: snapshot,
      [other.id]: { ...snapshot, serverId: other.id },
      [offline.id]: { ...snapshot, serverId: offline.id, status: 'offline' as const },
      [stale.id]: { ...snapshot, serverId: stale.id, timestamp: 1 },
    }
    await act(async () => root.render(<TrainingView servers={[server, other, offline, stale]} snapshots={allSnapshots} onNotice={vi.fn()} />))
    await openSavedProfile()
    const overview = container.querySelector('[aria-label="所有服务器 GPU 状态"]')!
    expect(overview.querySelectorAll('input[type="checkbox"]')).toHaveLength(8)
    expect(overview.querySelector('[aria-label="Offline server GPU 状态"]')?.textContent).toContain('离线')
    expect(overview.querySelector('[aria-label="Stale server GPU 状态"]')?.textContent).toContain('已过期')
    expect(overview.querySelector('[aria-label="Stale server GPU 状态"]')?.textContent).not.toContain('当前空闲')
    await act(async () => overview.querySelector<HTMLInputElement>('input[aria-label="22 GPU 1 RTX 4090"]')!.click())
    expect(container.querySelector<HTMLSelectElement>('select')?.value).toBe(other.id)
    expect(overview.querySelector<HTMLInputElement>('input[aria-label="Training server GPU 0 RTX 4090"]')?.checked).toBe(false)
    expect(overview.querySelector<HTMLInputElement>('input[aria-label="22 GPU 1 RTX 4090"]')?.checked).toBe(true)
    await act(async () => button('保存配置').click())
    expect(trainingApi.saveProfile).toHaveBeenLastCalledWith(expect.objectContaining({ serverId: other.id, gpuUuids: ['GPU-1'], interpreter: '/home/alice/miniconda3/envs/deploy/bin/python' }))
  })

  it('uses freshly checked GPUs without a monitor snapshot and invalidates checks after editing paths', async () => {
    vi.spyOn(trainingApi, 'checkPaths').mockResolvedValue({ ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus: [{ ...gpu, utilization: 0, memoryUsedMb: 0, computeProcessCount: 0, eligible: true, reasons: [] }], commandPreview: '', message: '路径与环境检查通过' })
    await act(async () => root.render(<TrainingView servers={[server]} snapshots={{}} onNotice={vi.fn()} />))
    await openSavedProfile()
    await act(async () => button('检查路径与环境').click())
    const overview = container.querySelector('[aria-label="所有服务器 GPU 状态"]')!
    expect(overview.textContent).toContain('当前空闲')
    expect(overview.textContent).not.toContain('等待服务器 GPU 数据')
    await act(async () => overview.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click())
    expect(container.querySelector('[role="status"]')?.textContent).toBe('路径与环境检查通过')
    const directory = container.querySelector<HTMLInputElement>('input[placeholder="/home/user/projects/my-model"]')!
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(directory, '/another/project')
      directory.dispatchEvent(new Event('input', { bubbles: true }))
    })
    expect(container.querySelector('[role="status"]')).toBeNull()
  })

  it('shows busy GPU checkboxes and does not submit a queue until the explicit second confirmation', async () => {
    const submit = vi.spyOn(trainingApi, 'submit').mockResolvedValueOnce({ job: null, requiresQueueConfirmation: true, reasons: ['GPU 0 存在计算进程'] }).mockImplementationOnce(async (id) => ({ job: { ...job, id }, requiresQueueConfirmation: false, reasons: [] }))
    await render()
    await openSavedProfile()
    const busyGpu = container.querySelector<HTMLInputElement>('input[aria-label="Training server GPU 0 RTX 4090"]')
    expect(busyGpu).not.toBeNull()
    expect(busyGpu?.disabled).toBe(false)
    expect(busyGpu?.checked).toBe(true)
    expect(container.textContent).toContain('部分资源繁忙')
    await act(async () => button('开始训练').click())
    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit.mock.calls[0][2]).toBe(false)
    expect(container.querySelector('[role="alertdialog"]')?.textContent).toContain('是否后台排队')
    await act(async () => button('确认后台排队').click())
    expect(submit).toHaveBeenCalledTimes(2)
    expect(submit.mock.calls[1][0]).toBe(submit.mock.calls[0][0])
    expect(submit.mock.calls[1][1]).toEqual(submit.mock.calls[0][1])
    expect(submit.mock.calls[1][2]).toBe(true)
    expect(container.querySelector('[role="alertdialog"]')).toBeNull()
    expect(container.textContent).toContain('排队中')
  })

  it('returns to editing without submitting when the user declines queuing', async () => {
    const submit = vi.spyOn(trainingApi, 'submit').mockResolvedValue({ job: null, requiresQueueConfirmation: true, reasons: ['GPU 0 忙碌'] })
    await render(); await openSavedProfile()
    await act(async () => button('开始训练').click())
    await act(async () => button('返回修改').click())
    expect(submit).toHaveBeenCalledTimes(1)
    expect(container.querySelector('[role="dialog"]')?.textContent).toContain('配置模型训练')
    expect(container.querySelector<HTMLInputElement>('input[placeholder="例如：ResNet50 · 基线训练"]')?.value).toBe(profile.name)
  })

  it('updates positional config preview without stealing input focus on each edit', async () => {
    await render(); await openSavedProfile()
    const prefix = container.querySelector<HTMLInputElement>('input[placeholder="python -u / conda run -n train python / torchrun"]')!
    act(() => prefix.focus())
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!
    await act(async () => { setValue.call(prefix, 'conda run -n ml python'); prefix.dispatchEvent(new Event('input', { bubbles: true })) })
    expect(document.activeElement).toBe(prefix)
    const positional = [...container.querySelectorAll<HTMLInputElement>('input[type="radio"]')][1]
    await act(async () => positional.click())
    const preview = container.querySelector('[aria-label="最终训练命令"]')!.textContent!
    expect(preview).toContain("conda run -n ml python 'train.py' 'configs/train.yaml'")
    expect(preview).not.toContain('--config')
  })

  it('requires confirmation before stopping a running training task', async () => {
    vi.mocked(trainingApi.listJobs).mockResolvedValue([{ ...job, status: 'running', pid: 888 }])
    await render()
    await act(async () => button('停止训练').click())
    expect(trainingApi.cancelJob).not.toHaveBeenCalled()
    const dialog = container.querySelector('[role="alertdialog"]')!
    expect(dialog.textContent).toContain('未保存的训练进度可能丢失')
    await act(async () => button('停止训练', dialog).click())
    expect(trainingApi.cancelJob).toHaveBeenCalledWith(server.id, job.id)
  })

  it('does not send incomplete forms and can still open configuration without servers', async () => {
    const submit = vi.spyOn(trainingApi, 'submit')
    await render()
    await act(async () => button('新建训练').click())
    await act(async () => button('开始训练').click())
    expect(submit).not.toHaveBeenCalled()
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('请填写训练名称')
    await act(async () => button('取消').click())
    await act(async () => root.render(<TrainingView servers={[]} snapshots={{}} onNotice={vi.fn()} />))
    await act(async () => button('新建训练').click())
    expect(container.textContent).toContain('先从左侧添加训练服务器')
    expect(button('开始训练').disabled).toBe(true)
  })

  it('recovers a waiting queue using its original job identity and profile only after confirmation', async () => {
    vi.mocked(trainingApi.listJobs).mockResolvedValue([job])
    const submit = vi.spyOn(trainingApi, 'submit').mockResolvedValue({ job, requiresQueueConfirmation: false, reasons: [] })
    await render()
    await act(async () => button('恢复队列').click())
    expect(submit).not.toHaveBeenCalled()
    const dialog = container.querySelector('[role="alertdialog"]')!
    expect(dialog.textContent).toContain('不会新增重复任务')
    await act(async () => button('恢复队列', dialog).click())
    expect(submit).toHaveBeenCalledOnce()
    expect(submit).toHaveBeenCalledWith(job.id, job.profile, true)
  })

  it('ignores a late log response after switching to a different task', async () => {
    const another = { ...job, id: 'another-job', name: 'Another training' }
    vi.mocked(trainingApi.listJobs).mockResolvedValue([job, another])
    let resolveOld!: (value: string) => void
    vi.mocked(trainingApi.readLog).mockImplementation((_serverId, jobId) => jobId === job.id ? new Promise<string>((resolve) => { resolveOld = resolve }) : Promise.resolve('NEW TASK LOG'))
    await render()
    await act(async () => container.querySelector<HTMLButtonElement>(`[aria-label="查看 ${job.name} 日志"]`)?.click())
    await act(async () => button('完成').click())
    await act(async () => container.querySelector<HTMLButtonElement>(`[aria-label="查看 ${another.name} 日志"]`)?.click())
    await act(async () => resolveOld('OLD TASK LOG'))
    expect(container.querySelector('[aria-label="训练日志内容"]')?.textContent).toBe('NEW TASK LOG')
  })

  it('saves a reusable script configuration before GPUs are selected', async () => {
    await render(); await openSavedProfile()
    await act(async () => container.querySelector<HTMLInputElement>('input[aria-label="Training server GPU 0 RTX 4090"]')?.click())
    await act(async () => button('保存配置').click())
    expect(trainingApi.saveProfile).toHaveBeenCalledWith(expect.objectContaining({ gpuUuids: [], scriptPath: profile.scriptPath, configPath: profile.configPath }))
  })

  it('reuses the same submission identity after a lost response to avoid duplicate training', async () => {
    const submit = vi.spyOn(trainingApi, 'submit').mockRejectedValueOnce(new Error('connection interrupted')).mockImplementationOnce(async (id) => ({ job: { ...job, id }, requiresQueueConfirmation: false, reasons: [] }))
    await render(); await openSavedProfile()
    await act(async () => button('开始训练').click())
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('connection interrupted')
    await act(async () => button('开始训练').click())
    expect(submit.mock.calls[1][0]).toBe(submit.mock.calls[0][0])
  })

  it('keeps a running job active until the server confirms that stopping has completed', async () => {
    vi.mocked(trainingApi.listJobs).mockResolvedValue([{ ...job, status: 'running' }])
    vi.mocked(trainingApi.cancelJob).mockResolvedValue({ ...job, status: 'running', waitingReason: '正在停止训练及其子进程' })
    await render()
    await act(async () => button('停止训练').click())
    await act(async () => button('停止训练', container.querySelector('[role="alertdialog"]')!).click())
    expect(container.textContent).toContain('训练中')
    expect(container.textContent).not.toContain('已取消')
  })
})

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
  it('shows busy GPU checkboxes and does not submit a queue until the explicit second confirmation', async () => {
    const submit = vi.spyOn(trainingApi, 'submit').mockResolvedValueOnce({ job: null, requiresQueueConfirmation: true, reasons: ['GPU 0 存在计算进程'] }).mockImplementationOnce(async (id) => ({ job: { ...job, id }, requiresQueueConfirmation: false, reasons: [] }))
    await render()
    await openSavedProfile()
    const busyGpu = container.querySelector<HTMLInputElement>('input[aria-label="GPU 0 RTX 4090"]')
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
    await act(async () => container.querySelector<HTMLInputElement>('input[aria-label="GPU 0 RTX 4090"]')?.click())
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

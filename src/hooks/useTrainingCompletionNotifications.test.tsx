// @vitest-environment jsdom
import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { TrainingJob } from '../types/training'
import { useTrainingCompletionNotifications } from './useTrainingCompletionNotifications'

const mocks = vi.hoisted(() => ({ listJobs: vi.fn(), notify: vi.fn() }))
vi.mock('../services/training', () => ({ trainingApi: { listJobs: mocks.listJobs } }))
vi.mock('../services/api', () => ({ api: { notify: mocks.notify } }))

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true
const STORAGE_KEY = 'modelgym.trainingCompletionNotifications.v1'
const base: TrainingJob = { id: 'job', serverId: 'server', name: 'Baseline', profile: { id: 'profile', name: 'Baseline', serverId: 'server', workingDirectory: '/workspace', interpreter: 'python', scriptPath: 'train.py', configPath: 'config.py', configMode: 'positional', configFlag: '--config', extraArguments: '', gpuUuids: ['GPU-0'], minimumFreeMemoryGb: 8 }, gpuUuids: ['GPU-0'], gpuIndices: [0], command: 'python train.py config.py', status: 'running', waitingReason: '', createdAt: 1 }
let container: HTMLDivElement
let root: ReturnType<typeof createRoot>
let hook: ReturnType<typeof useTrainingCompletionNotifications>

function Harness({ enabled = true }: { enabled?: boolean }) { hook = useTrainingCompletionNotifications(enabled); return null }
async function render(enabled = true) { await act(async () => root.render(<Harness enabled={enabled} />)) }
async function tick(milliseconds = 10_000) { await act(async () => { await vi.advanceTimersByTimeAsync(milliseconds) }) }
function restart() { act(() => root.unmount()); root = createRoot(container) }

beforeEach(() => {
  vi.useFakeTimers()
  localStorage.clear()
  mocks.listJobs.mockReset().mockResolvedValue([])
  mocks.notify.mockReset().mockResolvedValue(undefined)
  container = document.createElement('div')
  document.body.appendChild(container)
  root = createRoot(container)
})
afterEach(() => {
  act(() => root.unmount())
  container.remove()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('app-level training completion reminders', () => {
  it('suppresses a late completion poll after its history record has been deleted', async () => {
    mocks.listJobs.mockResolvedValue([base])
    await render()
    let resolve!: (jobs: TrainingJob[]) => void
    mocks.listJobs.mockReturnValueOnce(new Promise<TrainingJob[]>((done) => { resolve = done }))
    await tick()
    act(() => hook.dismiss(JSON.stringify([base.serverId, base.id])))
    await act(async () => resolve([{ ...base, status: 'completed' }]))
    expect(hook.alerts).toEqual([])
    expect(mocks.notify).not.toHaveBeenCalled()
  })
  it('tracks a just-accepted job even when it finishes before the next poll sees it active', async () => {
    await render()
    act(() => hook.observeJob(base))
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'failed', waitingReason: 'datalist exited immediately' }])
    await tick()
    expect(hook.alerts).toHaveLength(1)
    expect(hook.alerts[0].body).toContain('datalist exited immediately')
  })

  it('reminds for an accepted fast failure even when the terminal poll arrives before the submit reply', async () => {
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'failed', waitingReason: 'fast failure' }])
    await render()
    expect(hook.alerts).toEqual([])
    act(() => hook.observeJob(base))
    await tick()
    expect(hook.alerts).toHaveLength(1)
    act(() => hook.observeJob(base))
    await tick()
    expect(hook.alerts).toHaveLength(1)
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  })
  it('silently baselines old history and reminds once when an observed active job completes', async () => {
    mocks.listJobs.mockResolvedValue([{ ...base, id: 'old', status: 'completed' }, { ...base, id: 'old-failed', status: 'failed' }, base])
    await render()
    expect(hook.alerts).toEqual([])
    expect(mocks.notify).not.toHaveBeenCalled()
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'completed', endedAt: 20, exitCode: 0 }])
    await tick()
    expect(hook.alerts).toHaveLength(1)
    expect(hook.alerts[0]).toMatchObject({ title: '训练已完成', job: { id: base.id } })
    expect(mocks.notify).toHaveBeenCalledWith('训练已完成', expect.stringContaining(base.name), { kind: 'training', jobId: base.id, serverId: base.serverId })
    await tick()
    expect(mocks.notify).toHaveBeenCalledTimes(1)
    act(() => hook.dismiss(hook.alerts[0].id))
    expect(hook.alerts).toEqual([])
  })

  it('queues multiple completions, isolates identical job IDs by server, and ignores cancellation and SSH errors', async () => {
    const other = { ...base, serverId: 'other', name: 'Other server' }
    const cancelled = { ...base, id: 'cancelled', status: 'queued' as const }
    const disconnected = { ...base, id: 'disconnected' }
    mocks.listJobs.mockResolvedValue([base, other, cancelled, disconnected])
    await render()
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'completed' }, { ...other, status: 'failed', waitingReason: 'datalist failed' }, { ...cancelled, status: 'cancelled' }, { ...disconnected, status: 'failed', connectionError: 'SSH timeout' }])
    await tick()
    expect(hook.alerts.map((alert) => alert.title)).toEqual(['训练已完成', '训练失败'])
    expect(hook.alerts[1].body).toContain('datalist failed')
    expect(hook.alerts[0].id).not.toBe(hook.alerts[1].id)
    expect(mocks.notify).toHaveBeenCalledTimes(2)
    mocks.listJobs.mockResolvedValue([{ ...disconnected, status: 'failed', connectionError: null }])
    await tick()
    expect(hook.alerts).toHaveLength(3)
    expect(mocks.notify).toHaveBeenCalledTimes(3)
  })

  it('remembers active jobs across reopening and does not repeat an already reported completion', async () => {
    mocks.listJobs.mockResolvedValue([base])
    await render()
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!).active).toContain(JSON.stringify([base.serverId, base.id]))
    restart()
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'completed' }])
    await render()
    expect(hook.alerts).toHaveLength(1)
    expect(mocks.notify).toHaveBeenCalledTimes(1)
    restart()
    await render()
    expect(hook.alerts).toEqual([])
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  })

  it('does not overlap slow polls or dispatch a late response after unmount', async () => {
    const stored = { active: [JSON.stringify([base.serverId, base.id])], seen: [] }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored))
    let resolve!: (jobs: TrainingJob[]) => void
    mocks.listJobs.mockReturnValue(new Promise<TrainingJob[]>((done) => { resolve = done }))
    await render()
    await tick(30_000)
    expect(mocks.listJobs).toHaveBeenCalledTimes(1)
    restart()
    await act(async () => { resolve([{ ...base, status: 'completed' }]) })
    expect(mocks.notify).not.toHaveBeenCalled()
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual(stored)
  })

  it('waits for startup readiness and continues after polling or system notification failures', async () => {
    await render(false)
    await tick()
    expect(mocks.listJobs).not.toHaveBeenCalled()
    mocks.listJobs.mockResolvedValue([base])
    await render(true)
    mocks.listJobs.mockRejectedValueOnce(new Error('network unavailable'))
    await tick()
    expect(hook.alerts).toEqual([])
    mocks.notify.mockRejectedValue(new Error('permission denied'))
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'failed', waitingReason: 'exit 1' }])
    await tick()
    expect(hook.alerts).toHaveLength(1)
    const next = { ...base, id: 'next' }
    mocks.listJobs.mockResolvedValue([next])
    await tick()
    mocks.listJobs.mockResolvedValue([{ ...next, status: 'completed' }])
    await tick()
    expect(hook.alerts).toHaveLength(2)
    expect(mocks.notify).toHaveBeenCalledTimes(2)
  })

  it('survives malformed or unavailable persistence without losing in-session reminders', async () => {
    localStorage.setItem(STORAGE_KEY, '{bad JSON')
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota exceeded') })
    mocks.listJobs.mockResolvedValue([base])
    await render()
    mocks.listJobs.mockResolvedValue([{ ...base, status: 'completed' }])
    await tick()
    await tick()
    expect(hook.alerts).toHaveLength(1)
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  })
})

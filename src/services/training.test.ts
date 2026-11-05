import { describe, expect, it, vi } from 'vitest'
import type { Server, Snapshot } from '../types/models'
import type { TrainingProfile } from '../types/training'
import { DEFAULT_TRAINING_PROJECT_COLOR, datalistCommandPreview, defaultTrainingInterpreter, trainingApi, trainingCommandPreview, trainingGpuFromSnapshot, trainingProjectColor, trainingProjectIndex, trainingServerChange, validateTrainingProfile } from './training'

const profile: TrainingProfile = { id: 'profile', name: 'Model training', serverId: 'server', workingDirectory: '/work/my model', interpreter: 'conda run -n train python -u', scriptPath: "scripts/train's model.py", configPath: 'configs/base config.yaml', configMode: 'flag', configFlag: '--config', extraArguments: '--epochs 20 --label "baseline run"', gpuUuids: ['GPU-0'], minimumFreeMemoryGb: 8 }

function server(number: number, username = 'zhouzhen'): Server {
  return { id: `server-${number}`, name: String(number), host: `192.168.0.${number}`, port: 22, username, tags: [], samplingIntervalSeconds: 2, historyRetentionDays: 90, remoteHistoryEnabled: false, authMethod: 'sshAgent', status: 'online' }
}

describe('independent training project classifications', () => {
  it('indexes trimmed project names and legacy unclassified profiles and assigns distinct available colors', () => {
    const profiles = [profile, { ...profile, id: 'a', projectName: ' Detection ', projectColor: '#8B5CF6' }, { ...profile, id: 'b', projectName: 'Detection', projectColor: '#8b5cf6' }]
    expect(trainingProjectIndex(profiles)).toEqual([{ name: 'Detection', color: '#8b5cf6', count: 2 }, { name: '', color: DEFAULT_TRAINING_PROJECT_COLOR, count: 1 }])
    expect(trainingProjectColor(' Detection ', profiles)).toBe('#8b5cf6')
    expect(trainingProjectColor('', profiles)).toBe(DEFAULT_TRAINING_PROJECT_COLOR)
    expect(trainingProjectColor('Segmentation', profiles)).not.toBe('#8b5cf6')
    const many: TrainingProfile[] = []
    for (let index = 0; index < 16; index++) {
      const name = `Project ${index}`
      many.push({ ...profile, id: name, projectName: name, projectColor: trainingProjectColor(name, many) })
    }
    expect(new Set(many.map((item) => item.projectColor)).size).toBe(16)
    expect(trainingCommandPreview(profiles[1])).toBe(trainingCommandPreview(profile))
  })

  it('validates project names by trimmed UTF-8 byte length and accepts only hex colors', () => {
    expect(validateTrainingProfile({ ...profile, projectName: ` ${'数'.repeat(40)} `, projectColor: '#ABC123' })).toBeNull()
    expect(validateTrainingProfile({ ...profile, projectName: '数'.repeat(41) })).toContain('120 字节')
    for (const name of ['project\n', 'project\r', 'project\0']) expect(validateTrainingProfile({ ...profile, projectName: name })).toContain('项目名称')
    for (const color of ['red', '#abc', '#12345g', '#12345678', '']) expect(validateTrainingProfile({ ...profile, projectColor: color })).toContain('#RRGGBB')
  })

  it('persists shared colors across a project and leaves other projects and job snapshots intact', async () => {
    const items = new Map<string, string>()
    vi.stubGlobal('localStorage', { getItem: (key: string) => items.get(key) ?? null, setItem: (key: string, value: string) => items.set(key, value) })
    try {
      await trainingApi.saveProfile({ ...profile, id: 'one', projectName: ' Detection ', projectColor: '#ABC123' })
      await trainingApi.saveProfile({ ...profile, id: 'other', projectName: 'Segmentation', projectColor: '#10b981' })
      await trainingApi.saveProfile({ ...profile, id: 'two', projectName: 'Detection', projectColor: '#8b5cf6' })
      const restored = await trainingApi.listProfiles()
      expect(restored.filter((item) => item.projectName === 'Detection').map((item) => item.projectColor)).toEqual(['#8b5cf6', '#8b5cf6'])
      expect(restored.find((item) => item.id === 'other')?.projectColor).toBe('#10b981')
      await trainingApi.saveProfile({ ...restored.find((item) => item.id === 'one')!, projectName: 'New project', projectColor: '#ec4899' })
      expect((await trainingApi.listProfiles()).find((item) => item.id === 'two')?.projectColor).toBe('#8b5cf6')
      expect((await trainingApi.saveProfile({ ...profile, projectName: '  ' })).projectColor).toBe(DEFAULT_TRAINING_PROJECT_COLOR)
    } finally { vi.unstubAllGlobals() }
  })
})

describe('training history deletion preview', () => {
  it('rejects a delayed submit after cleanup has already marked that task deleted', async () => {
    const items = new Map<string, string>()
    vi.stubGlobal('localStorage', { getItem: (key: string) => items.get(key) ?? null, setItem: (key: string, value: string) => items.set(key, value) })
    let resolve!: (value: Awaited<ReturnType<typeof trainingApi.checkPaths>>) => void
    const check = vi.spyOn(trainingApi, 'checkPaths').mockImplementationOnce(() => new Promise((done) => { resolve = done }))
    try {
      const pending = expect(trainingApi.submit('late', profile, true)).rejects.toThrow('已删除')
      await trainingApi.deleteJobs(profile.serverId, ['late'])
      resolve({ ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus: [], commandPreview: '', message: 'demo' })
      await pending
      expect(await trainingApi.listJobs()).toEqual([])
    } finally { check.mockRestore(); vi.unstubAllGlobals() }
  })

  it('deletes only selected ended records on the chosen server, reports active failures and prevents old submit retries', async () => {
    const items = new Map<string, string>()
    vi.stubGlobal('localStorage', { getItem: (key: string) => items.get(key) ?? null, setItem: (key: string, value: string) => items.set(key, value) })
    const check = vi.spyOn(trainingApi, 'checkPaths').mockResolvedValue({ ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus: [{ uuid: 'GPU-0', index: 0, name: 'GPU', utilization: 0, memoryUsedMb: 0, memoryTotalMb: 24 * 1024, computeProcessCount: 0, eligible: true, reasons: [] }], commandPreview: '', message: 'demo' })
    try {
      await trainingApi.submit('ended', profile, false)
      await trainingApi.cancelJob(profile.serverId, 'ended')
      await trainingApi.submit('active', profile, false)
      await trainingApi.submit('ended', { ...profile, serverId: 'other' }, false)
      const result = await trainingApi.deleteJobs(profile.serverId, ['ended', 'active', 'ended'])
      expect(result.deletedJobIds).toEqual(['ended'])
      expect(result.failures).toEqual([{ jobId: 'active', message: '仅可删除已结束的任务记录。' }])
      expect((await trainingApi.listJobs()).map((job) => `${job.serverId}:${job.id}`)).toEqual(['other:ended', 'server:active'])
      expect(await trainingApi.deleteJobs(profile.serverId, ['ended'])).toEqual({ deletedJobIds: ['ended'], failures: [] })
      await expect(trainingApi.submit('ended', profile, false)).rejects.toThrow('已删除')
      await expect(trainingApi.deleteJobs(profile.serverId, Array.from({ length: 201 }, (_, index) => String(index)))).rejects.toThrow('200')
    } finally { check.mockRestore(); vi.unstubAllGlobals() }
  })
})

describe('training server defaults', () => {
  it.each([20, 21, 23])('uses the CUDA 13 training environment on server %s', (number) => {
    expect(defaultTrainingInterpreter(server(number))).toBe('/home/zhouzhen/miniconda3/envs/torch29-cu130/bin/python')
  })

  it.each([22, 26, 27, 28])('uses the deploy environment on server %s', (number) => {
    expect(defaultTrainingInterpreter(server(number, 'alice'))).toBe('/home/alice/miniconda3/envs/deploy/bin/python')
  })

  it('supports numbered SSH aliases and display names without matching unrelated host numbers', () => {
    expect(defaultTrainingInterpreter({ ...server(21), host: 'training.internal', name: '训练服务器21' })).toContain('/torch29-cu130/')
    expect(defaultTrainingInterpreter({ ...server(22), host: 'training.internal', name: 'Training', sshAlias: 'gpu-22' })).toContain('/deploy/')
    expect(defaultTrainingInterpreter({ ...server(20), host: '192.168.0.120' })).toBe('python -u')
    expect(defaultTrainingInterpreter(server(25))).toBe('python -u')
    expect(defaultTrainingInterpreter(undefined)).toBe('python -u')
    expect(defaultTrainingInterpreter(server(21, ''))).toBe('python -u')
  })

  it('updates automatic prefixes and clears GPU selection when switching servers', () => {
    const previous = server(21)
    const next = server(22, 'alice')
    expect(trainingServerChange({ ...profile, interpreter: defaultTrainingInterpreter(previous) }, previous, next)).toEqual({ serverId: next.id, gpuUuids: [], interpreter: '/home/alice/miniconda3/envs/deploy/bin/python' })
    expect(trainingServerChange({ ...profile, interpreter: 'python -u' }, server(25), next).interpreter).toBe(defaultTrainingInterpreter(next))
    expect(trainingServerChange({ ...profile, interpreter: '' }, previous, next).interpreter).toBe(defaultTrainingInterpreter(next))
  })

  it('preserves a custom launch prefix when switching servers', () => {
    expect(trainingServerChange(profile, server(21), server(22))).toMatchObject({ interpreter: profile.interpreter, gpuUuids: [] })
  })
})

describe('training command and resource boundary', () => {
  it('previews an optional CPU datalist script with its own arguments and quoted paths', () => {
    expect(datalistCommandPreview(profile)).toBe('')
    const datalistProfile = { ...profile, datalistScriptPath: "tools/create list's.py", datalistArguments: '--output "data/train list.json"' }
    expect(datalistCommandPreview(datalistProfile)).toBe(`cd -- '/work/my model'\nCUDA_VISIBLE_DEVICES='' conda run -n train python -u 'tools/create list'"'"'s.py' --output "data/train list.json"`)
    expect(datalistCommandPreview(datalistProfile)).not.toContain(profile.configPath)
    expect(trainingCommandPreview(datalistProfile)).toBe(trainingCommandPreview(profile))
    expect(validateTrainingProfile(datalistProfile)).toBeNull()
  })

  it('rejects orphaned or malformed datalist script arguments while accepting legacy profiles', () => {
    expect(validateTrainingProfile(profile)).toBeNull()
    expect(validateTrainingProfile({ ...profile, datalistArguments: '--output list.json' })).toContain('生成脚本')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: 'tools/list\0.py' })).toContain('空字符')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: 'tools/list\n.py' })).toContain('换行')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: 'tools/list.py', datalistArguments: '--output a\npython bad.py' })).toContain('同一行')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: 'x'.repeat(4097) })).toContain('过长')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: '数'.repeat(1366) })).toContain('过长')
    expect(validateTrainingProfile({ ...profile, datalistScriptPath: 'tools/list.py', datalistArguments: 'x'.repeat(16385) })).toContain('过长')
  })

  it('keeps datalist generation pending in the browser demo instead of claiming it ran', async () => {
    const items = new Map<string, string>()
    vi.stubGlobal('localStorage', { getItem: (key: string) => items.get(key) ?? null, setItem: (key: string, value: string) => items.set(key, value) })
    const check = vi.spyOn(trainingApi, 'checkPaths').mockResolvedValue({ ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, datalistScriptExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus: [{ uuid: 'GPU-0', index: 0, name: 'GPU', utilization: 0, memoryUsedMb: 0, memoryTotalMb: 24 * 1024, computeProcessCount: 0, eligible: true, reasons: [] }], commandPreview: trainingCommandPreview(profile), message: 'demo' })
    try {
      const configured = { ...profile, datalistScriptPath: 'tools/list.py', datalistArguments: '--output list.json' }
      const saved = await trainingApi.saveProfile(configured)
      expect((await trainingApi.listProfiles())[0]).toMatchObject({ datalistScriptPath: configured.datalistScriptPath, datalistArguments: configured.datalistArguments })
      const result = await trainingApi.submit('demo-datalist', saved, false)
      expect(result.job).toMatchObject({ status: 'queued', datalistStatus: 'pending', pid: null })
      expect(result.job?.waitingReason).toContain('未实际运行')
      const repeated = await trainingApi.submit('demo-datalist', saved, false)
      expect(repeated.job).toEqual(result.job)
      expect(check).toHaveBeenCalledTimes(1)
    } finally {
      check.mockRestore()
      vi.unstubAllGlobals()
    }
  })

  it('quotes remote paths and supports named or positional config arguments', () => {
    expect(trainingCommandPreview({ ...profile, gpuUuids: ['GPU-selected-a', 'GPU-selected-b'] })).toBe(`cd -- '/work/my model'\nCUDA_VISIBLE_DEVICES='GPU-selected-a,GPU-selected-b' conda run -n train python -u 'scripts/train'"'"'s model.py' --config 'configs/base config.yaml' --epochs 20 --label "baseline run"`)
    expect(trainingCommandPreview({ ...profile, configMode: 'positional' })).toContain(`'scripts/train'"'"'s model.py' 'configs/base config.yaml' --epochs`)
    expect(trainingCommandPreview({ ...profile, configMode: 'positional' })).not.toContain('--config')
  })

  it('rejects malformed config flags, duplicate GPUs and unsupported multiline commands', () => {
    expect(validateTrainingProfile(profile)).toBeNull()
    expect(validateTrainingProfile({ ...profile, configFlag: '--config; reboot' })).toContain('配置参数名')
    expect(validateTrainingProfile({ ...profile, gpuUuids: ['GPU-0', 'GPU-0'] })).toContain('重复')
    expect(validateTrainingProfile({ ...profile, interpreter: 'source env.sh\npython' })).toContain('同一行')
    expect(validateTrainingProfile({ ...profile, minimumFreeMemoryGb: Number.NaN })).toContain('最低空闲显存')
    expect(validateTrainingProfile({ ...profile, gpuUuids: [] })).toContain('至少选择')
    expect(validateTrainingProfile({ ...profile, gpuUuids: [] }, false)).toBeNull()
  })

  it('uses strictly more free memory than the configured threshold regardless of GPU load', () => {
    const gpu = { uuid: 'GPU-0', index: 0, name: 'RTX', utilization: 10, memoryUtilization: 0, memoryUsedMb: 0, memoryTotalMb: 24 * 1024, temperatureCelsius: 30, powerWatts: 20 }
    const snapshot = { gpus: [gpu, { ...gpu, uuid: 'GPU-1', index: 1, utilization: 11 }, { ...gpu, uuid: 'GPU-2', index: 2, memoryUsedMb: 20 * 1024 }], processes: [{ gpuUuid: 'GPU-1', isCurrentUser: true }], processesSampled: true } as Snapshot
    const candidates = trainingGpuFromSnapshot(snapshot, 8)
    expect(candidates).toHaveLength(3)
    expect(candidates[0].eligible).toBe(true)
    expect(candidates[1]).toMatchObject({ eligible: true, computeProcessCount: 1 })
    expect(candidates[2].reasons.join(' ')).toContain('空闲显存须大于')
    expect(trainingGpuFromSnapshot({ ...snapshot, processesSampled: false }, 8)[0].eligible).toBe(true)
    expect(trainingGpuFromSnapshot({ ...snapshot, gpus: [{ ...gpu, utilization: 99, memoryUsedMb: 14 * 1024 }] }, 10)[0].eligible).toBe(false)
    expect(trainingGpuFromSnapshot({ ...snapshot, gpus: [{ ...gpu, utilization: 99, memoryUsedMb: 14 * 1024 - 1 }] }, 10)[0].eligible).toBe(true)
    expect(trainingGpuFromSnapshot({ ...snapshot, gpus: [{ ...gpu, memoryUsedMb: Number.NaN }] }, 10)[0].eligible).toBe(false)
  })

  it('still requires queue confirmation for compute load even when the display shows enough memory', async () => {
    const check = vi.spyOn(trainingApi, 'checkPaths').mockResolvedValue({ ok: true, workingDirectoryExists: true, scriptExists: true, configExists: true, pythonAvailable: true, nvidiaSmiAvailable: true, gpus: [{ uuid: 'GPU-0', index: 0, name: 'GPU', utilization: 99, memoryUsedMb: 0, memoryTotalMb: 24 * 1024, computeProcessCount: 1, eligible: true, reasons: [] }], commandPreview: '', message: 'demo' })
    try {
      const result = await trainingApi.submit('memory-display-qa', profile, false)
      expect(result.requiresQueueConfirmation).toBe(true)
      expect(result.reasons.join(' ')).toContain('计算进程')
      expect(result.job).toBeNull()
    } finally { check.mockRestore() }
  })
})

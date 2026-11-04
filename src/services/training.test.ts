import { describe, expect, it } from 'vitest'
import type { Server, Snapshot } from '../types/models'
import type { TrainingProfile } from '../types/training'
import { defaultTrainingInterpreter, trainingCommandPreview, trainingGpuFromSnapshot, trainingServerChange, validateTrainingProfile } from './training'

const profile: TrainingProfile = { id: 'profile', name: 'Model training', serverId: 'server', workingDirectory: '/work/my model', interpreter: 'conda run -n train python -u', scriptPath: "scripts/train's model.py", configPath: 'configs/base config.yaml', configMode: 'flag', configFlag: '--config', extraArguments: '--epochs 20 --label "baseline run"', gpuUuids: ['GPU-0'], minimumFreeMemoryGb: 8 }

function server(number: number, username = 'zhouzhen'): Server {
  return { id: `server-${number}`, name: String(number), host: `192.168.0.${number}`, port: 22, username, tags: [], samplingIntervalSeconds: 2, historyRetentionDays: 90, remoteHistoryEnabled: false, authMethod: 'sshAgent', status: 'online' }
}

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

  it('keeps busy cards visible and requires zero compute processes, <=10% utilization and enough memory', () => {
    const gpu = { uuid: 'GPU-0', index: 0, name: 'RTX', utilization: 10, memoryUtilization: 0, memoryUsedMb: 0, memoryTotalMb: 24 * 1024, temperatureCelsius: 30, powerWatts: 20 }
    const snapshot = { gpus: [gpu, { ...gpu, uuid: 'GPU-1', index: 1, utilization: 11 }, { ...gpu, uuid: 'GPU-2', index: 2, memoryUsedMb: 20 * 1024 }], processes: [{ gpuUuid: 'GPU-1', isCurrentUser: true }], processesSampled: true } as Snapshot
    const candidates = trainingGpuFromSnapshot(snapshot, 8)
    expect(candidates).toHaveLength(3)
    expect(candidates[0].eligible).toBe(true)
    expect(candidates[1]).toMatchObject({ eligible: false, computeProcessCount: 1 })
    expect(candidates[1].reasons.join(' ')).toContain('11%')
    expect(candidates[2].reasons.join(' ')).toContain('空闲显存不足')
    expect(trainingGpuFromSnapshot({ ...snapshot, processesSampled: false }, 8)[0].eligible).toBe(false)
  })
})

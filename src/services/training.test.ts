import { describe, expect, it } from 'vitest'
import type { Snapshot } from '../types/models'
import type { TrainingProfile } from '../types/training'
import { trainingCommandPreview, trainingGpuFromSnapshot, validateTrainingProfile } from './training'

const profile: TrainingProfile = { id: 'profile', name: 'Model training', serverId: 'server', workingDirectory: '/work/my model', interpreter: 'conda run -n train python -u', scriptPath: "scripts/train's model.py", configPath: 'configs/base config.yaml', configMode: 'flag', configFlag: '--config', extraArguments: '--epochs 20 --label "baseline run"', gpuUuids: ['GPU-0'], minimumFreeMemoryGb: 8 }

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

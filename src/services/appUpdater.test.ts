import { afterEach, describe, expect, it, vi } from 'vitest'

const { check, invoke } = vi.hoisted(() => ({ check: vi.fn(), invoke: vi.fn() }))
vi.mock('@tauri-apps/plugin-updater', () => ({ check }))
vi.mock('@tauri-apps/plugin-process', () => ({ relaunch: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke, Channel: class {} }))

import { checkDesktopAppUpdate, MODEL_GYM_UPDATES_ENABLED } from './appUpdater'
import { api } from './api'

afterEach(() => { vi.resetAllMocks(); vi.unstubAllGlobals() })

describe('ModelGym update isolation', () => {
  it.each(['Windows NT 10.0', 'Macintosh', 'Linux'])('never requests an upstream desktop update on %s', async userAgent => {
    vi.stubGlobal('navigator', { userAgent })
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    expect(MODEL_GYM_UPDATES_ENABLED).toBe(false)
    expect(await checkDesktopAppUpdate()).toBeNull()
    expect(check).not.toHaveBeenCalled()
    expect(invoke).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
  })

  it('blocks the legacy GitHub release service without contacting the network', async () => {
    const fetch = vi.fn()
    vi.stubGlobal('fetch', fetch)
    await expect(api.getLatestRelease()).rejects.toThrow('ModelGym 自动更新尚未配置')
    expect(fetch).not.toHaveBeenCalled()
    expect(invoke).not.toHaveBeenCalled()
  })
})

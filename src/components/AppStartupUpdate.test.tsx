// @vitest-environment jsdom
import { StrictMode, act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import packageInfo from '../../package.json'
import { api } from '../services/api'
import { saveCachedUpdate } from '../utils/updateCheck'

vi.mock('./SshTerminal', () => ({ SshTerminal: () => null }))
vi.mock('./TrainingView', () => ({ TrainingView: () => <section data-testid="training-view">训练配置入口</section> }))
;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true

class ResizeObserverStub implements ResizeObserver {
  constructor(_callback: ResizeObserverCallback) {}
  observe(_target: Element, _options?: ResizeObserverOptions) {}
  unobserve(_target: Element) {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub
Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  value: vi.fn(() => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
})
let root: ReturnType<typeof createRoot> | null = null
beforeEach(() => {
  vi.spyOn(api, 'listServers').mockResolvedValue([])
  vi.spyOn(api, 'listLatestSnapshots').mockResolvedValue([])
})
afterEach(() => {
  if (root) act(() => root?.unmount())
  root = null
  document.body.innerHTML = ''
  localStorage.clear()
  vi.restoreAllMocks()
})

async function renderApp() {
  const container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await act(async () => { root?.render(<StrictMode><App /></StrictMode>); await Promise.resolve() })
  return container
}

describe('ModelGym upgrade identity and navigation', () => {
  it('never checks or offers an upstream RackTop release, including a cached release', async () => {
    saveCachedUpdate({ lastCheckedAt: Date.now(), release: { version: '99.0.0', url: 'https://github.com/Tongzh-SEU/RackTop/releases/tag/v99.0.0' } })
    const getLatestRelease = vi.spyOn(api, 'getLatestRelease')
    const container = await renderApp()
    expect(getLatestRelease).not.toHaveBeenCalled()
    expect(container.querySelector('.brand__update--trigger')).toBeNull()
    await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="关于 ModelGym"]')?.click())
    expect(container.textContent).toContain('ModelGym ' + packageInfo.version)
    expect(container.textContent).toContain('手动安装更新')
    expect(container.textContent).toContain('已停用原 RackTop 更新源')
    expect([...container.querySelectorAll('button')].some((button) => button.textContent === '重新检查')).toBe(false)
    expect(getLatestRelease).not.toHaveBeenCalled()
  })

  it('makes model training reachable with no configured servers and preserves the other menus', async () => {
    const container = await renderApp()
    const training = [...container.querySelectorAll<HTMLButtonElement>('nav button')].find((button) => button.textContent === '模型训练')
    expect(training).toBeDefined()
    for (const label of ['总览', '空闲算力', '我的进程', '我的项目']) expect(container.querySelector('nav')?.textContent).toContain(label)
    await act(async () => training?.click())
    expect(container.querySelector('[data-testid="training-view"]')).not.toBeNull()
    expect(container.querySelector('h1')?.textContent).toBe('模型训练')
  })

  it('keeps the upstream author attribution separate from product updates', async () => {
    const open = vi.spyOn(window, 'open').mockReturnValue(null)
    const container = await renderApp()
    await act(async () => container.querySelector<HTMLButtonElement>('button[aria-label="关于 ModelGym"]')?.click())
    expect(container.textContent).toContain('上游 RackTop 原作者')
    const author = [...container.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent?.includes('GitHub @Tongzh-SEU'))
    await act(async () => author?.click())
    expect(open).toHaveBeenCalledWith('https://github.com/Tongzh-SEU', '_blank', 'noopener,noreferrer')
  })
})

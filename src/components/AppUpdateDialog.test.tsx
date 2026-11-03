// @vitest-environment jsdom

import { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AppUpdateDialog, AppUpdateProgressButton, AppUpdateTriggerButton } from './AppUpdateDialog'

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: ReturnType<typeof createRoot> | null = null

afterEach(() => {
  if (root) act(() => root?.unmount())
  root = null
  document.body.innerHTML = ''
})

function renderDialog(phase: 'downloading' | 'installing' | 'error') {
  const container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  const handlers = { onClose: vi.fn(), onRetry: vi.fn(), onManualDownload: vi.fn() }
  act(() => root?.render(<AppUpdateDialog
    state={{ phase, version: '1.25.4', downloadedBytes: 7_000_000, totalBytes: 12_000_000, error: phase === 'error' ? '网络连接失败' : undefined }}
    {...handlers}
  />))
  return { container, handlers }
}

describe('AppUpdateDialog', () => {
  it('keeps the familiar circular update icon for the idle trigger', () => {
    const container = document.createElement('div')
    document.body.append(container)
    const onClick = vi.fn()
    root = createRoot(container)
    act(() => root?.render(<AppUpdateTriggerButton version="1.25.6" onClick={onClick} />))
    expect(container.querySelector('.brand__update-ring')).toBeNull()
    expect(container.querySelector('.brand__update-icon')).not.toBeNull()
    act(() => container.querySelector('button')?.click())
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('renders download progress as a compact blue ring with accessible progress', () => {
    const container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
    act(() => root?.render(<AppUpdateProgressButton state={{ phase: 'downloading', version: '1.25.6', downloadedBytes: 3, totalBytes: 4 }} />))
    const progress = container.querySelector('[role="progressbar"]')
    expect(progress?.getAttribute('aria-valuenow')).toBe('75')
    expect(progress?.getAttribute('aria-label')).toContain('75%')
    expect(container.querySelector('.brand__update-ring-value')?.getAttribute('stroke-dashoffset')).toBe(`${2 * Math.PI * 10 * .25}`)
  })

  it('keeps unknown-size downloads stationary without a false percentage', () => {
    const container = document.createElement('div')
    document.body.append(container)
    root = createRoot(container)
    act(() => root?.render(<AppUpdateProgressButton state={{ phase: 'downloading', version: '1.25.6', downloadedBytes: 1, totalBytes: null }} />))
    const progress = container.querySelector('[role="progressbar"]')
    expect(progress?.hasAttribute('aria-valuenow')).toBe(false)
    expect(container.querySelector('.brand__update-ring-track')).not.toBeNull()
    expect(container.querySelector('.brand__update-ring-value')).toBeNull()
  })

  it('shows compact progress and no cancellation control while active', () => {
    const { container } = renderDialog('downloading')
    expect(container.textContent).toContain('正在下载 ModelGym 1.25.4')
    expect(container.textContent).toContain('MB /')
    expect(container.querySelector('[role="progressbar"]')?.getAttribute('aria-valuenow')).toBe('58')
    expect(container.querySelector('button')).toBeNull()
  })

  it('keeps installation non-cancellable', () => {
    const { container } = renderDialog('installing')
    expect(container.textContent).toContain('正在安装 ModelGym 1.25.4')
    expect(container.querySelector('button')).toBeNull()
  })

  it('offers close, retry, and manual download only after failure', () => {
    const { container, handlers } = renderDialog('error')
    const buttons = [...container.querySelectorAll('button')]
    expect(buttons.map((button) => button.textContent).join(' ')).toContain('手动下载')
    expect(buttons.map((button) => button.textContent).join(' ')).toContain('重试')
    act(() => buttons.find((button) => button.textContent?.includes('手动下载'))?.click())
    act(() => buttons.find((button) => button.textContent?.includes('重试'))?.click())
    expect(handlers.onManualDownload).toHaveBeenCalledOnce()
    expect(handlers.onRetry).toHaveBeenCalledOnce()
  })
})

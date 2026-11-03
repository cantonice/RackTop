import { CircleArrowUp, ExternalLink, RefreshCw, X } from 'lucide-react'
import { appUpdatePercent, formatUpdateBytes, type AppUpdateState } from '../utils/appUpdate'

interface AppUpdateDialogProps {
  state: AppUpdateState
  onClose: () => void
  onRetry: () => void
  onManualDownload: () => void
}

interface AppUpdateProgressButtonProps {
  state: AppUpdateState
}

function UpdateRing({ value }: { value?: number }) {
  const circumference = 2 * Math.PI * 10
  const offset = value === undefined ? undefined : circumference * (1 - value / 100)
  return <svg className="brand__update-ring" viewBox="0 0 24 24" aria-hidden="true">
    <circle className="brand__update-ring-track" cx="12" cy="12" r="10" />
    {value !== undefined ? <circle className="brand__update-ring-value" cx="12" cy="12" r="10" strokeDasharray={circumference} strokeDashoffset={offset} /> : null}
  </svg>
}

export function AppUpdateTriggerButton({ version, onClick }: { version: string; onClick: () => void }) {
  return <button className="brand__update brand__update--trigger" onClick={onClick} aria-label={`有新版本 ModelGym ${version}，点击下载并安装`} title={`更新到 ModelGym ${version}`}>
    <CircleArrowUp className="brand__update-icon" size={16} strokeWidth={2} />
  </button>
}

export function AppUpdateProgressButton({ state }: AppUpdateProgressButtonProps) {
  const percent = appUpdatePercent(state)
  const determinate = percent !== null && state.phase === 'downloading'
  const value = determinate ? Math.round(percent) : state.phase === 'installing' ? 100 : undefined
  const label = state.phase === 'checking'
    ? '正在检查更新'
    : state.phase === 'installing'
      ? `正在安装 ModelGym ${state.version}`
      : determinate
        ? `正在下载 ModelGym ${state.version}，已完成 ${value}%`
        : `正在下载 ModelGym ${state.version}`

  return <span className="brand__update-progress" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={value} title={label}>
    <UpdateRing value={value} />
    <CircleArrowUp className="brand__update-icon" size={16} strokeWidth={2} />
  </span>
}

export function AppUpdateDialog({ state, onClose, onRetry, onManualDownload }: AppUpdateDialogProps) {
  const failed = state.phase === 'error'
  const checking = state.phase === 'checking'
  const installing = state.phase === 'installing'
  const percent = appUpdatePercent(state)
  const progressLabel = state.totalBytes
    ? `${formatUpdateBytes(state.downloadedBytes)} / ${formatUpdateBytes(state.totalBytes)}`
    : state.downloadedBytes > 0
      ? `${formatUpdateBytes(state.downloadedBytes)} / 未知大小`
      : '正在准备下载...'

  return <div className="scrim app-update-scrim">
    <section className={`sheet app-update-sheet${failed ? ' app-update-sheet--error' : ''}`} role="dialog" aria-modal="true" aria-labelledby="app-update-title" aria-describedby={failed ? 'app-update-error' : undefined}>
      {failed ? <><div className="app-update-failure">
        <div><h2 id="app-update-title">无法完成更新</h2><p id="app-update-error">{state.error ?? '更新失败，请重试或手动下载安装。'}</p></div>
        <button className="icon-button" onClick={onClose} aria-label="关闭更新窗口"><X size={17} /></button>
      </div></> : <div className="app-update-body">
        <div className="app-update-status-copy">
          <strong id="app-update-title">{checking ? '正在检查更新' : installing ? `正在安装 ModelGym ${state.version}` : `正在下载 ModelGym ${state.version}`}</strong>
          <span>{checking ? '请稍候...' : installing ? '下载完成' : progressLabel}</span>
        </div>
        <div className={`app-update-progress${percent === null || checking || installing ? ' is-indeterminate' : ''}`} role="progressbar" aria-label={checking ? '正在检查更新' : installing ? '正在安装更新' : '更新下载进度'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent === null || checking || installing ? undefined : Math.round(percent)}>
          <span style={percent === null || checking || installing ? undefined : { transform: `scaleX(${percent / 100})` }} />
        </div>
        <small>下载和安装过程中无法取消。正在运行的终端和同步任务将中断。</small>
      </div>}
      {failed && <footer className="sheet__footer">
        <button className="button button--secondary button--small" onClick={onManualDownload}><ExternalLink size={13} />手动下载</button>
        <button className="button button--primary button--small" onClick={onRetry}><RefreshCw size={13} />重试</button>
      </footer>}
    </section>
  </div>
}

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { AlertCircle, Check, CheckCircle2, ChevronRight, Clock3, Dumbbell, FileCode2, FileCog, LoaderCircle, Pencil, Play, Plus, RefreshCw, ScrollText, Server as ServerIcon, Square, Trash2, X } from 'lucide-react'
import { api } from '../services/api'
import { TrainingGpuPicker } from './TrainingGpuPicker'
import { datalistCommandPreview, defaultTrainingInterpreter, trainingApi, trainingCommandPreview, trainingGpuFromSnapshot, trainingProjectColor, trainingProjectIndex, trainingServerChange, validateTrainingProfile } from '../services/training'
import type { Server, Snapshot } from '../types/models'
import { isActiveTrainingJob, TRAINING_STATUS_LABELS } from '../types/training'
import type { TrainingGpu, TrainingJob, TrainingPathCheck, TrainingProfile } from '../types/training'

export interface TrainingViewProps {
  servers: Server[]
  snapshots: Record<string, Snapshot>
  onNotice: (message: string) => void
  onRefreshServer?: (serverId: string) => void | Promise<void>
  onJobAccepted?: (job: TrainingJob) => void
  completionJobId?: string | null
  onJobsDeleted?: (jobs: Array<{ serverId: string; jobId: string }>) => void
}

function newProfile(servers: Server[]): TrainingProfile {
  const now = Math.floor(Date.now() / 1000)
  const server = servers.find((item) => item.status === 'online') ?? servers[0]
  return { id: crypto.randomUUID(), name: '', serverId: server?.id ?? '', workingDirectory: '', interpreter: defaultTrainingInterpreter(server), datalistScriptPath: '', datalistArguments: '', scriptPath: '', configPath: '', configMode: 'positional', configFlag: '--config', extraArguments: '', gpuUuids: [], minimumFreeMemoryGb: 10, createdAt: now, updatedAt: now }
}

function messageFor(error: unknown): string { return String(error).replace(/^Error:\s*/, '') }
function timestamp(value?: number | null): string { return value ? new Date(value * 1000).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—' }
function gpuNumbers(job: TrainingJob): string { return job.gpuIndices.length ? job.gpuIndices.map((index) => `GPU ${index}`).join(' · ') : `${job.gpuUuids.length} 张指定 GPU` }
function jobKey(job: Pick<TrainingJob, 'serverId' | 'id'>): string { return JSON.stringify([job.serverId, job.id]) }
function datalistStatus(job: TrainingJob): string | null {
  if (!job.profile.datalistScriptPath?.trim()) return null
  if (job.status === 'cancelled') return job.datalistStatus === 'completed' ? 'datalist 已生成' : 'datalist 生成已取消'
  if (job.datalistStatus === 'running') return '正在生成 datalist'
  if (job.datalistStatus === 'completed') return 'datalist 已生成'
  if (job.datalistStatus === 'failed') return 'datalist 生成失败'
  return '等待生成 datalist'
}

function Dialog({ title, description, onClose, busy = false, wide = false, alert = false, children, footer }: { title: string; description?: string; onClose: () => void; busy?: boolean; wide?: boolean; alert?: boolean; children: ReactNode; footer?: ReactNode }) {
  const titleId = useId()
  const descriptionId = useId()
  const ref = useRef<HTMLElement>(null)
  const closeRef = useRef(onClose)
  const busyRef = useRef(busy)
  closeRef.current = onClose
  busyRef.current = busy
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const first = ref.current?.querySelector<HTMLElement>('[data-autofocus],input:not([disabled]),select:not([disabled]),textarea:not([disabled]),button:not([disabled])')
    ;(first ?? ref.current)?.focus()
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); if (!busyRef.current) closeRef.current(); return }
      if (event.key !== 'Tab') return
      const focusable = [...ref.current?.querySelectorAll<HTMLElement>('button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex="0"]') ?? []]
      if (!focusable.length) { event.preventDefault(); ref.current?.focus(); return }
      const firstElement = focusable[0]
      const lastElement = focusable[focusable.length - 1]
      if (event.shiftKey && (document.activeElement === firstElement || !ref.current?.contains(document.activeElement))) { event.preventDefault(); lastElement.focus() }
      if (!event.shiftKey && (document.activeElement === lastElement || !ref.current?.contains(document.activeElement))) { event.preventDefault(); firstElement.focus() }
    }
    document.addEventListener('keydown', keydown)
    return () => { document.removeEventListener('keydown', keydown); if (previous?.isConnected) previous.focus() }
  }, [])
  return <div className="scrim training-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}><section className={`sheet training-sheet${wide ? ' training-sheet--wide' : ''}`} ref={ref} role={alert ? 'alertdialog' : 'dialog'} aria-modal="true" aria-busy={busy} aria-labelledby={titleId} aria-describedby={description ? descriptionId : undefined} tabIndex={-1}>
    <header className="training-sheet__header"><div><p className="eyebrow">MODELGYM · TRAINING</p><h2 id={titleId}>{title}</h2>{description && <p id={descriptionId}>{description}</p>}</div><button className="icon-button" aria-label={`关闭${title}`} disabled={busy} onClick={onClose}><X size={18} /></button></header>
    <div className="training-sheet__body">{children}</div>
    {footer && <footer className="training-sheet__footer">{footer}</footer>}
  </section></div>
}

function currentGpuStatus(gpu: TrainingGpu, minimum: number): TrainingGpu {
  const reasons: string[] = []
  if (gpu.uuid.startsWith('unavailable-') || !Number.isFinite(gpu.memoryTotalMb) || gpu.memoryTotalMb <= 0 || !Number.isFinite(gpu.memoryUsedMb) || gpu.memoryUsedMb < 0 || gpu.memoryUsedMb > gpu.memoryTotalMb) reasons.push('GPU 显存指标不可读')
  if ((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024 < minimum) reasons.push(`空闲显存不足 ${minimum} GB`)
  return { ...gpu, eligible: reasons.length === 0, reasons }
}

export function TrainingView({ servers, snapshots, onNotice, onRefreshServer, onJobAccepted, completionJobId, onJobsDeleted }: TrainingViewProps) {
  const [tab, setTab] = useState<'jobs' | 'profiles'>('jobs')
  const [jobFilter, setJobFilter] = useState<'active' | 'recent'>('active')
  useEffect(() => { if (completionJobId) { setTab('jobs'); setJobFilter('recent') } }, [completionJobId])
  const [profiles, setProfiles] = useState<TrainingProfile[]>([])
  const [projectFilter, setProjectFilter] = useState<string | null>(null)
  const projectOptionsId = useId()
  const projectGroups = useMemo(() => trainingProjectIndex(profiles), [profiles])
  const visibleProfiles = projectFilter === null ? profiles : profiles.filter((item) => (item.projectName?.trim() ?? '') === projectFilter)
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [editorOpen, setEditorOpen] = useState(false)
  const [profile, setProfile] = useState<TrainingProfile>(() => newProfile(servers))
  const [memoryInput, setMemoryInput] = useState('10')
  const [operation, setOperation] = useState<'checking' | 'saving' | 'submitting' | 'cancelling' | 'deletingJobs' | null>(null)
  const [selectedHistory, setSelectedHistory] = useState<Set<string>>(new Set())
  const [pendingDeletion, setPendingDeletion] = useState<TrainingJob[] | null>(null)
  const [deletionErrors, setDeletionErrors] = useState<string[]>([])
  const deletedJobs = useRef(new Set<string>())
  const [formError, setFormError] = useState<string | null>(null)
  const [check, setCheck] = useState<{ serverId: string; checkedAt: number; result: TrainingPathCheck } | null>(null)
  const [pendingQueue, setPendingQueue] = useState<{ jobId: string; profile: TrainingProfile; reasons: string[] } | null>(null)
  const [pendingAction, setPendingAction] = useState<{ kind: 'cancel' | 'delete' | 'resume'; job?: TrainingJob; profile?: TrainingProfile } | null>(null)
  const [logJob, setLogJob] = useState<TrainingJob | null>(null)
  const [logText, setLogText] = useState('')
  const [logError, setLogError] = useState<string | null>(null)
  const [logLoading, setLogLoading] = useState(false)
  const loadBusy = useRef(false)
  const profilesRevision = useRef(0)
  const alive = useRef(true)
  const logJobId = useRef<string | null>(null)
  const logRequest = useRef(0)
  const submissionIdentity = useRef<{ key: string; jobId: string } | null>(null)
  logJobId.current = logJob ? jobKey(logJob) : null

  const load = useCallback(async (manual = false) => {
    if (loadBusy.current) return
    loadBusy.current = true
    if (manual) setRefreshing(true)
    const revision = profilesRevision.current
    try {
      const [nextProfiles, nextJobs] = await Promise.all([trainingApi.listProfiles(), trainingApi.listJobs()])
      if (!alive.current) return
      if (revision === profilesRevision.current) setProfiles(nextProfiles)
      setJobs(nextJobs.filter((job) => !deletedJobs.current.has(jobKey(job))))
      setLoadError(null)
    } catch (error) { if (alive.current) setLoadError(messageFor(error)) }
    finally { loadBusy.current = false; if (alive.current) { setLoading(false); setRefreshing(false) } }
  }, [])

  useEffect(() => {
    alive.current = true
    void load()
    const timer = window.setInterval(() => void load(), 5000)
    return () => { alive.current = false; window.clearInterval(timer) }
  }, [load])

  const selectedSnapshot = snapshots[profile.serverId]
  const gpus = useMemo(() => {
    if (check?.serverId === profile.serverId && (!selectedSnapshot || check.checkedAt >= selectedSnapshot.timestamp)) return check.result.gpus.map((gpu) => currentGpuStatus(gpu, profile.minimumFreeMemoryGb))
    return selectedSnapshot ? trainingGpuFromSnapshot(selectedSnapshot, profile.minimumFreeMemoryGb) : []
  }, [check, profile.minimumFreeMemoryGb, profile.serverId, selectedSnapshot])
  const selectedGpus = gpus.filter((gpu) => profile.gpuUuids.includes(gpu.uuid))
  const preview = trainingCommandPreview(profile)
  const activeJobs = jobs.filter(isActiveTrainingJob)
  const queuedCount = activeJobs.filter((job) => job.status !== 'running').length
  const runningCount = activeJobs.filter((job) => job.status === 'running').length
  const visibleJobs = (jobFilter === 'active' ? activeJobs : jobs.filter((job) => !isActiveTrainingJob(job))).slice().sort((left, right) => right.createdAt - left.createdAt)
  const selectedHistoryJobs = jobs.filter((job) => !isActiveTrainingJob(job) && selectedHistory.has(jobKey(job)))
  useEffect(() => {
    if (projectFilter !== null && !projectGroups.some((group) => group.name === projectFilter)) setProjectFilter(null)
  }, [projectFilter, projectGroups])
  useEffect(() => {
    const finished = new Set(jobs.filter((job) => !isActiveTrainingJob(job)).map(jobKey))
    setSelectedHistory((current) => {
      const next = new Set([...current].filter((key) => finished.has(key)))
      return next.size === current.size ? current : next
    })
  }, [jobs])

  function updateProfile(patch: Partial<TrainingProfile>) {
    setProfile((current) => ({ ...current, ...(patch.serverId !== undefined && patch.serverId !== current.serverId ? trainingServerChange(current, servers.find((server) => server.id === current.serverId), servers.find((server) => server.id === patch.serverId)) : {}), ...patch }))
    setFormError(null)
    if ((patch.serverId !== undefined && patch.serverId !== profile.serverId) || (['workingDirectory', 'interpreter', 'datalistScriptPath', 'datalistArguments', 'scriptPath', 'configPath', 'configMode', 'configFlag', 'extraArguments'] as const).some((key) => patch[key] !== undefined && patch[key] !== profile[key])) setCheck(null)
  }

  function openEditor(source?: TrainingProfile) {
    const next = source ? structuredClone(source) : { ...newProfile(servers), projectName: projectFilter ?? '', projectColor: trainingProjectColor(projectFilter ?? '', profiles) }
    setProfile(next)
    setMemoryInput(String(next.minimumFreeMemoryGb))
    setFormError(null)
    setCheck(null)
    setEditorOpen(true)
  }

  function updateProjectName(name: string) {
    const suggested = trainingProjectColor(name, profiles)
    const known = projectGroups.find((group) => group.name === name.trim())
    const customized = profile.projectColor && profile.projectColor !== trainingProjectColor(profile.projectName ?? '', profiles)
    updateProfile({ projectName: name, projectColor: known?.color ?? (customized ? profile.projectColor : suggested) })
  }

  async function confirmHistoryDeletion() {
    if (!pendingDeletion || operation) return
    setOperation('deletingJobs'); setDeletionErrors([])
    const grouped = new Map<string, TrainingJob[]>()
    for (const job of pendingDeletion) grouped.set(job.serverId, [...(grouped.get(job.serverId) ?? []), job])
    const removed: Array<{ serverId: string; jobId: string }> = []
    const errors: string[] = []
    for (const [serverId, serverJobs] of grouped) {
      for (let offset = 0; offset < serverJobs.length; offset += 200) {
        const batch = serverJobs.slice(offset, offset + 200)
        try {
          const result = await trainingApi.deleteJobs(serverId, batch.map((job) => job.id))
          const confirmed = new Set(result.deletedJobIds)
          const batchRemoved: Array<{ serverId: string; jobId: string }> = []
          for (const job of batch) {
            if (confirmed.has(job.id)) {
              deletedJobs.current.add(jobKey(job)); removed.push({ serverId, jobId: job.id }); batchRemoved.push({ serverId, jobId: job.id })
            } else errors.push(`${servers.find((server) => server.id === serverId)?.name ?? '服务器已移除'} · ${job.name}：${result.failures.find((failure) => failure.jobId === job.id)?.message ?? '服务器未确认清理成功'}`)
          }
          if (batchRemoved.length) onJobsDeleted?.(batchRemoved)
        } catch (error) { for (const job of batch) errors.push(`${servers.find((server) => server.id === serverId)?.name ?? '服务器已移除'} · ${job.name}：${messageFor(error)}`) }
      }
    }
    if (alive.current) {
      setJobs((current) => current.filter((job) => !deletedJobs.current.has(jobKey(job))))
      setSelectedHistory((current) => new Set([...current].filter((key) => !deletedJobs.current.has(key))))
      setLogJob((current) => current && deletedJobs.current.has(jobKey(current)) ? null : current)
      setPendingDeletion(null); setDeletionErrors(errors); setOperation(null)
    }
    onNotice(removed.length ? `已删除 ${removed.length} 项记录及受管日志${errors.length ? `，${errors.length} 项清理失败` : ''}` : '清理未完成，请查看失败原因后重试')
  }

  async function inspectPaths() {
    const error = validateTrainingProfile(profile, false)
    if (error) { setFormError(error); return }
    setOperation('checking'); setFormError(null)
    try {
      const result = await trainingApi.checkPaths(profile)
      setCheck({ serverId: profile.serverId, checkedAt: Math.floor(Date.now() / 1000), result })
      if (!result.ok) setFormError(result.message || '路径或运行环境检查未通过。')
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  async function saveProfile() {
    const error = validateTrainingProfile(profile, false)
    if (error) { setFormError(error); return }
    setOperation('saving'); setFormError(null)
    try {
      const saved = await trainingApi.saveProfile(profile)
      profilesRevision.current += 1
      setProfiles((current) => [saved, ...current.filter((item) => item.id !== saved.id).map((item) => saved.projectName?.trim() && item.projectName?.trim() === saved.projectName.trim() ? { ...item, projectColor: saved.projectColor } : item)])
      setProfile(saved)
      onNotice(`已保存训练配置“${saved.name}”`)
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  function acceptJob(job: TrainingJob) {
    onJobAccepted?.(job)
    submissionIdentity.current = null
    setJobs((current) => [job, ...current.filter((item) => jobKey(item) !== jobKey(job))])
    setEditorOpen(false); setPendingQueue(null); setTab('jobs'); setJobFilter('active')
    onNotice(job.status === 'running' ? `训练“${job.name}”已启动` : `训练“${job.name}”已提交至服务器后台`)
    void onRefreshServer?.(job.serverId)
  }

  async function submitTraining() {
    const error = validateTrainingProfile(profile)
    if (error) { setFormError(error); return }
    const submission = structuredClone(profile)
    const submissionKey = JSON.stringify(submission)
    if (submissionIdentity.current?.key !== submissionKey) submissionIdentity.current = { key: submissionKey, jobId: crypto.randomUUID() }
    const jobId = submissionIdentity.current.jobId
    setOperation('submitting'); setFormError(null)
    try {
      const result = await trainingApi.submit(jobId, submission, false)
      if (result.requiresQueueConfirmation) {
        setEditorOpen(false)
        setPendingQueue({ jobId, profile: submission, reasons: result.reasons })
      } else if (result.job) acceptJob(result.job)
      else setFormError('服务器没有返回训练任务，请检查连接后重试。')
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  async function confirmQueue() {
    if (!pendingQueue || operation) return
    setOperation('submitting'); setFormError(null)
    try {
      const result = await trainingApi.submit(pendingQueue.jobId, pendingQueue.profile, true)
      if (result.job) acceptJob(result.job)
      else setFormError(result.reasons.join('；') || '服务器尚未接受排队任务，请重试。')
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  async function confirmAction() {
    if (!pendingAction || operation) return
    setOperation('cancelling'); setFormError(null)
    try {
      if (pendingAction.kind === 'delete' && pendingAction.profile) {
        await trainingApi.deleteProfile(pendingAction.profile.id)
        profilesRevision.current += 1
        setProfiles((current) => current.filter((item) => item.id !== pendingAction.profile!.id))
        onNotice('已删除训练配置，已提交的任务不受影响')
      } else if (pendingAction.kind === 'resume' && pendingAction.job) {
        const job = pendingAction.job
        const result = await trainingApi.submit(job.id, job.profile, true)
        if (!result.job) throw new Error(result.reasons.join('；') || '服务器未接受恢复请求。')
        onJobAccepted?.(result.job)
        setJobs((current) => current.map((item) => jobKey(item) === jobKey(job) ? result.job! : item))
        onNotice('已恢复服务器后台队列，仍按原配置和 GPU 等待空闲')
      } else if (pendingAction.job) {
        const job = pendingAction.job
        const updated = await trainingApi.cancelJob(job.serverId, job.id)
        setJobs((current) => current.map((item) => jobKey(item) === jobKey(job) ? updated : item))
        onNotice(updated.status === 'running' ? '已发送停止请求，正在等待训练进程退出' : updated.status === 'cancelled' ? '已取消训练任务' : updated.datalistStatus === 'running' ? '已发送停止请求，正在等待 datalist 生成脚本退出' : '任务状态已更新')
      }
      setPendingAction(null)
      void load()
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  const loadLog = useCallback(async (job: TrainingJob) => {
    const request = ++logRequest.current
    setLogLoading(true); setLogError(null)
    const current = () => alive.current && logJobId.current === jobKey(job) && logRequest.current === request
    try { const content = await trainingApi.readLog(job.serverId, job.id); if (current()) setLogText(content) }
    catch (error) { if (current()) setLogError(messageFor(error)) }
    finally { if (current()) setLogLoading(false) }
  }, [])

  useEffect(() => {
    if (!logJob) return
    setLogText('')
    void loadLog(logJob)
    const timer = window.setInterval(() => void loadLog(logJob), 5000)
    return () => window.clearInterval(timer)
  }, [loadLog, logJob])

  const cancellingRun = pendingAction?.job?.status === 'running'
  return <section className="training-view" aria-label="模型训练">
    <div className="training-toolbar"><div className="training-segments" role="group" aria-label="训练视图"><button aria-pressed={tab === 'jobs'} className={tab === 'jobs' ? 'is-active' : ''} onClick={() => setTab('jobs')}>训练任务 <span>{activeJobs.length}</span></button><button aria-pressed={tab === 'profiles'} className={tab === 'profiles' ? 'is-active' : ''} onClick={() => setTab('profiles')}>训练配置 <span>{profiles.length}</span></button></div><div className="training-toolbar__actions"><button className="button button--secondary" onClick={() => void load(true)} disabled={refreshing} aria-label="刷新训练任务">{refreshing ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}刷新</button><button className="button button--primary" onClick={() => openEditor()}><Plus size={15} />新建训练</button></div></div>
    {!api.isDesktop && <p className="training-preview-note"><AlertCircle size={15} />浏览器预览：路径检查和任务提交均为演示，不执行真实训练。</p>}
    {loadError && <div className="training-error" role="alert"><AlertCircle size={16} /><span>无法刷新训练数据：{loadError}。已有任务状态已保留。</span><button onClick={() => void load(true)}>重试</button></div>}
    <div className="training-summary"><div><span className="training-summary__icon"><Dumbbell size={17} /></span><span><small>正在训练</small><strong>{runningCount}<em>项任务</em></strong></span></div><div><span className="training-summary__icon is-waiting"><Clock3 size={17} /></span><span><small>等待资源</small><strong>{queuedCount}<em>项任务</em></strong></span></div><div className="training-summary__policy"><CheckCircle2 size={16} /><p><strong>等待空闲，自动开始</strong><small>指定 GPU 连续空闲 30 秒，并完成可选的 datalist 生成后启动。</small></p></div></div>
    {tab === 'jobs' ? <section className="training-task-section" aria-label="训练任务列表"><div className="training-section-heading"><h2>训练任务</h2><div className="training-task-filter" role="group" aria-label="任务状态"><button className={jobFilter === 'active' ? 'is-active' : ''} aria-pressed={jobFilter === 'active'} onClick={() => setJobFilter('active')}>进行中</button><button className={jobFilter === 'recent' ? 'is-active' : ''} aria-pressed={jobFilter === 'recent'} onClick={() => setJobFilter('recent')}>最近结束</button></div></div>
      {jobFilter === 'recent' && <div className="training-history-selection"><label><input type="checkbox" aria-label="全选已结束任务" disabled={!visibleJobs.length || operation === 'deletingJobs'} checked={visibleJobs.length > 0 && visibleJobs.every((job) => selectedHistory.has(jobKey(job)))} ref={(element) => { if (element) element.indeterminate = selectedHistoryJobs.length > 0 && !visibleJobs.every((job) => selectedHistory.has(jobKey(job))) }} onChange={(event) => { setSelectedHistory(event.target.checked ? new Set(visibleJobs.map(jobKey)) : new Set()); setDeletionErrors([]) }} />全选</label><span>已选 {selectedHistoryJobs.length} 项</span><button className="button button--danger" disabled={!selectedHistoryJobs.length || operation !== null} onClick={() => { setDeletionErrors([]); setPendingDeletion([...selectedHistoryJobs]) }}><Trash2 size={13} />删除所选记录及日志</button></div>}
      {jobFilter === 'recent' && deletionErrors.length > 0 && <div className="training-error training-deletion-errors" role="alert"><strong>本次清理返回以下错误：</strong><ul>{deletionErrors.map((error, index) => <li key={index}>{error}</li>)}</ul></div>}
      {loading ? <div className="training-empty" role="status"><LoaderCircle className="spin" size={24} /><strong>正在读取训练任务…</strong></div> : visibleJobs.length ? <div className="training-job-list">{visibleJobs.map((job) => <article className={`training-job${selectedHistory.has(jobKey(job)) ? ' is-selected' : ''}`} key={jobKey(job)}><div className="training-job__leading">{jobFilter === 'recent' && <input type="checkbox" checked={selectedHistory.has(jobKey(job))} aria-label={`选择删除 ${job.name}（${servers.find((server) => server.id === job.serverId)?.name ?? job.serverId}）`} disabled={operation === 'deletingJobs'} onChange={(event) => { setSelectedHistory((current) => { const next = new Set(current); if (event.target.checked) next.add(jobKey(job)); else next.delete(jobKey(job)); return next }); setDeletionErrors([]) }} />}<div className={`training-job__status is-${job.status}`}>{job.status === 'running' ? <Dumbbell size={18} /> : job.status === 'queued' || job.status === 'stabilizing' ? <Clock3 size={18} /> : job.status === 'completed' ? <CheckCircle2 size={18} /> : <Square size={17} />}</div></div><div className="training-job__content"><header><h3>{job.name}</h3><span className={`training-status is-${job.status}`}>{TRAINING_STATUS_LABELS[job.status]}</span></header>{datalistStatus(job) && <p className={`training-job__datalist is-${job.datalistStatus ?? 'pending'}`}>{datalistStatus(job)}</p>}<p className="training-job__meta"><span><ServerIcon size={12} />{servers.find((server) => server.id === job.serverId)?.name ?? '服务器已移除'}</span><span>{gpuNumbers(job)}</span><span>提交于 {timestamp(job.createdAt)}</span>{job.startedAt && <span>启动于 {timestamp(job.startedAt)}</span>}</p><p className={`training-job__reason${job.connectionError ? ' is-error' : ''}`}>{job.connectionError ? `连接失败：${job.connectionError}。服务器队列可能仍在运行，恢复连接后继续查询。` : job.waitingReason || (job.status === 'running' ? '训练进程正在服务器后台运行' : job.exitCode != null ? `退出码 ${job.exitCode}` : '任务状态已更新')}</p><details><summary>查看启动命令 <ChevronRight size={12} /></summary><pre>{job.command}</pre></details></div><div className="training-job__actions">{(job.status === 'queued' || job.status === 'stabilizing') && <button className="button button--secondary" onClick={() => { setFormError(null); setPendingAction({ kind: 'resume', job }) }}><Play size={12} />恢复队列</button>}<button className="button button--secondary" onClick={() => { setLogError(null); setLogJob(job) }} aria-label={`查看 ${job.name} 日志`}><ScrollText size={14} />日志</button>{isActiveTrainingJob(job) && <button className={`button ${job.status === 'running' ? 'button--danger' : 'button--secondary'}`} onClick={() => { setFormError(null); setPendingAction({ kind: 'cancel', job }) }}><Square size={12} />{job.status === 'running' ? '停止训练' : '取消排队'}</button>}</div></article>)}</div> : <div className="training-empty"><span className="training-empty__mark"><Dumbbell size={32} /></span><h3>{jobFilter === 'active' ? '让下一次训练准备就绪' : '还没有结束的训练任务'}</h3><p>{jobFilter === 'active' ? '选择服务器上的训练脚本与配置文件，再指定 GPU。资源繁忙时，可以确认加入后台队列。' : '完成、失败和已取消的任务会保留在这里。'}</p>{jobFilter === 'active' && <button className="button button--primary" onClick={() => openEditor()}><Plus size={15} />创建第一项训练</button>}</div>}
    </section> : <section className="training-task-section" aria-label="训练配置列表"><div className="training-section-heading"><div><h2>训练配置</h2><p>保存脚本、环境与显存要求，下一次直接选择运行。</p></div></div><div className="training-project-index" role="group" aria-label="项目分类索引"><button aria-label="显示全部项目" aria-pressed={projectFilter === null} className={projectFilter === null ? 'is-active' : ''} onClick={() => setProjectFilter(null)}>全部项目<span>{profiles.length}</span></button>{projectGroups.map((group) => <button key={group.name} aria-label={`筛选项目：${group.name || '未分类'}`} aria-pressed={projectFilter === group.name} className={projectFilter === group.name ? 'is-active' : ''} style={{ '--project-accent': group.color } as CSSProperties} onClick={() => setProjectFilter(group.name)}><i aria-hidden="true" />{group.name || '未分类'}<span>{group.count}</span></button>)}</div>{profiles.length ? <div className="training-profile-grid">{visibleProfiles.map((saved) => <article className="training-profile" style={{ '--project-accent': trainingProjectColor(saved.projectName ?? '', profiles) } as CSSProperties} data-project={saved.projectName?.trim() ?? ''} key={saved.id}><div className="training-profile__heading"><span><FileCode2 size={20} /></span><h3>{saved.name}</h3><button className="icon-button" aria-label={`编辑 ${saved.name}`} onClick={() => openEditor(saved)}><Pencil size={15} /></button></div><p className="training-project-tag"><i aria-hidden="true" />{saved.projectName?.trim() || '未分类'}</p><p><ServerIcon size={13} />{servers.find((server) => server.id === saved.serverId)?.name ?? '服务器已移除'}</p><dl><dt>训练脚本</dt><dd title={saved.scriptPath}>{saved.scriptPath}</dd><dt>配置文件</dt><dd title={saved.configPath}>{saved.configPath}</dd><dt>资源要求</dt><dd>{saved.gpuUuids.length || '待选择'} 张 GPU · 每卡空闲 ≥ {saved.minimumFreeMemoryGb} GB</dd></dl><footer><button className="icon-button" aria-label={`删除 ${saved.name}`} onClick={() => { setFormError(null); setPendingAction({ kind: 'delete', profile: saved }) }}><Trash2 size={14} /></button><button className="button button--secondary" onClick={() => openEditor(saved)}><Play size={13} />配置并启动</button></footer></article>)}</div> : <div className="training-empty"><FileCog size={30} /><h3>保存一套可重复使用的训练配置</h3><p>训练配置独立于“我的项目”，使用服务器上现有的脚本和配置文件。</p><button className="button button--primary" onClick={() => openEditor()}><Plus size={15} />添加训练配置</button></div>}</section>}

    {editorOpen && <Dialog title="配置模型训练" description="使用服务器上已有的脚本和配置文件，选择本次训练固定使用的 GPU。" onClose={() => setEditorOpen(false)} busy={operation !== null} wide footer={<><p className="training-footer-hint"><Clock3 size={14} />启动前会重新检查资源；繁忙时先征求你的排队确认。</p><button className="button button--secondary" disabled={operation !== null} onClick={() => setEditorOpen(false)}>取消</button><button className="button button--secondary" disabled={operation !== null} onClick={() => void saveProfile()}>{operation === 'saving' ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}保存配置</button><button className="button button--primary" disabled={operation !== null || servers.length === 0} onClick={() => void submitTraining()}>{operation === 'submitting' ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}{operation === 'submitting' ? '正在检查…' : '开始训练'}</button></>}>
      <section className="training-form-section training-datalist" aria-label="生成 datalist 配置" inert={operation !== null}>
        <header><span><FileCog size={13} /></span><div><h3>生成 datalist</h3><p>可选。任务提交后立即运行一次，与 GPU 空闲检测同时进行；生成失败则停止任务。</p></div></header>
        <div className="training-fields"><label>生成脚本<input className="mono" value={profile.datalistScriptPath ?? ''} onChange={(event) => updateProfile({ datalistScriptPath: event.target.value })} placeholder="tools/generate_datalist.py（可选）" spellCheck={false} /></label><label>生成脚本参数<input className="mono" value={profile.datalistArguments ?? ''} onChange={(event) => updateProfile({ datalistArguments: event.target.value })} placeholder="--output data/train.txt" spellCheck={false} /></label></div>
        <p className="training-datalist__hint">使用下方的工作目录和启动前缀，默认不使用 GPU。所需配置文件请在生成脚本参数中填写；留空脚本则跳过。</p>
        {profile.datalistScriptPath?.trim() && <pre aria-label="datalist 生成命令"><code>{datalistCommandPreview(profile)}</code></pre>}
      </section>
      <div className="training-editor" inert={operation !== null}><div className="training-editor__configuration"><section className="training-form-section"><header><span>1</span><div><h3>训练脚本与配置</h3><p>文件路径相对于工作目录，也可以填写绝对路径。</p></div></header><div className="training-fields"><label>训练名称<input data-autofocus value={profile.name} onChange={(event) => updateProfile({ name: event.target.value })} placeholder="例如：ResNet50 · 基线训练" /></label><label>训练服务器<select value={profile.serverId} onChange={(event) => updateProfile({ serverId: event.target.value, gpuUuids: [] })}><option value="">选择服务器</option>{servers.map((server) => <option value={server.id} key={server.id}>{server.name}{server.status === 'offline' ? ' · 离线' : ''}</option>)}</select></label><label>项目分类<input value={profile.projectName ?? ''} list={projectOptionsId} onChange={(event) => updateProjectName(event.target.value)} placeholder="选择或填写项目名（可选）" /><datalist id={projectOptionsId}>{projectGroups.filter((group) => group.name).map((group) => <option value={group.name} key={group.name} />)}</datalist></label><label>项目重点色<div className="training-project-color"><input type="color" aria-label="项目重点色" disabled={!profile.projectName?.trim()} value={profile.projectColor ?? trainingProjectColor(profile.projectName ?? '', profiles)} onChange={(event) => updateProfile({ projectColor: event.target.value })} /><span>{profile.projectColor ?? trainingProjectColor(profile.projectName ?? '', profiles)}</span></div><small>同名项目共用颜色，保存后同步到该项目全部配置。</small></label><label className="training-field--full">工作目录<input className="mono" value={profile.workingDirectory} onChange={(event) => updateProfile({ workingDirectory: event.target.value })} placeholder="/home/user/projects/my-model" spellCheck={false} /></label><label className="training-field--full">启动前缀<input className="mono" value={profile.interpreter} onChange={(event) => updateProfile({ interpreter: event.target.value })} placeholder="python -u / conda run -n train python / torchrun" spellCheck={false} /><small>支持 Python、Conda run 和 torchrun。按命令参数解析，不支持 source、&&、管道或重定向。</small></label><label className="training-field--full">训练脚本<input className="mono" value={profile.scriptPath} onChange={(event) => updateProfile({ scriptPath: event.target.value })} placeholder="train.py 或 /home/user/project/train.py" spellCheck={false} /></label><label className="training-field--full">配置文件<input className="mono" value={profile.configPath} onChange={(event) => updateProfile({ configPath: event.target.value })} placeholder="configs/train.yaml" spellCheck={false} /></label><fieldset className="training-config-mode training-field--full"><legend>配置文件传入方式</legend><label><input type="radio" name="training-config-mode" checked={profile.configMode === 'flag'} onChange={() => updateProfile({ configMode: 'flag' })} />参数式</label><label><input type="radio" name="training-config-mode" checked={profile.configMode === 'positional'} onChange={() => updateProfile({ configMode: 'positional' })} />位置式（紧跟脚本）</label>{profile.configMode === 'flag' && <label className="training-config-flag">参数名<input aria-label="配置参数名" value={profile.configFlag} onChange={(event) => updateProfile({ configFlag: event.target.value })} placeholder="--config" spellCheck={false} /></label>}</fieldset><label className="training-field--full">附加参数<span className="training-optional">可选</span><input className="mono" value={profile.extraArguments} onChange={(event) => updateProfile({ extraArguments: event.target.value })} placeholder="--epochs 100 --batch-size 32" spellCheck={false} /><small>参数中的空格可用引号包裹；不支持多行 shell 命令。</small></label></div><div className="training-path-check"><button className="button button--secondary" disabled={operation !== null || servers.length === 0} onClick={() => void inspectPaths()}>{operation === 'checking' ? <LoaderCircle className="spin" size={14} /> : <CheckCircle2 size={14} />}检查路径与环境</button>{check && <p role="status" className={check.result.ok ? 'is-success' : 'is-error'}>{check.result.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}{check.result.message || (check.result.ok ? '路径与环境检查通过' : '检查未通过')}</p>}</div></section></div>
      <div className="training-editor__resources"><section className="training-form-section"><header><span>2</span><div><h3>所有服务器与 GPU</h3><p>直接点选任意服务器的显卡；切换服务器会清空原来的选卡。</p></div></header><label className="training-memory-field">每卡最低空闲显存<div><input type="number" min="0" step="0.5" value={memoryInput} onChange={(event) => { setMemoryInput(event.target.value); if (event.target.value !== '') updateProfile({ minimumFreeMemoryGb: Number(event.target.value) }) }} onBlur={() => { if (!Number.isFinite(Number(memoryInput)) || memoryInput.trim() === '') { setMemoryInput(String(profile.minimumFreeMemoryGb)) } }} /><span>GB</span></div></label><p className="training-gpu-rule">空闲显存 ≥ {profile.minimumFreeMemoryGb} GB，连续稳定 30 秒后开训。利用率和计算进程仅供参考。</p><TrainingGpuPicker servers={servers} snapshots={snapshots} serverId={profile.serverId} gpuUuids={profile.gpuUuids} selectedServerGpus={gpus} selectedGpuCheckedAt={check?.serverId === profile.serverId ? check.checkedAt : undefined} minimumFreeMemoryGb={profile.minimumFreeMemoryGb} onSelect={(serverId, uuid, checked) => updateProfile({ serverId, gpuUuids: serverId === profile.serverId ? checked ? [...profile.gpuUuids, uuid] : profile.gpuUuids.filter((item) => item !== uuid) : [uuid] })} />{profile.gpuUuids.some((uuid) => !gpus.some((gpu) => gpu.uuid === uuid)) && <p className="training-snapshot-note">已保存的部分 GPU 暂未出现在当前快照中，提交时会重新核对 UUID。</p>}<p className="training-selection-count">已选 <strong>{profile.gpuUuids.length}</strong> 张 GPU{selectedGpus.some((gpu) => !gpu.eligible) && ' · 部分资源繁忙，开始训练时将询问是否排队'}</p></section><section className="training-form-section training-command"><header><span>3</span><div><h3>最终启动命令</h3><p>先核对脚本、配置参数与卡号，再提交。</p></div></header><pre aria-label="最终训练命令"><code>{preview}</code></pre><p>排队锁定本次配置与 GPU UUID，不会自动换卡。关闭客户端后，服务器继续监测和启动任务。</p></section></div></div>
      {formError && <p className="training-error" role="alert"><AlertCircle size={16} />{formError}</p>}
    </Dialog>}

    {pendingDeletion && <Dialog title="删除所选训练记录及日志？" description={`将永久删除 ${pendingDeletion.length} 项已结束任务的记录，以及服务器上 ModelGym 管理的 datalist / 训练输出日志。`} onClose={() => setPendingDeletion(null)} busy={operation === 'deletingJobs'} alert footer={<><button className="button button--secondary" disabled={operation === 'deletingJobs'} onClick={() => setPendingDeletion(null)}>保留记录</button><button className="button button--danger" disabled={operation === 'deletingJobs'} onClick={() => void confirmHistoryDeletion()}>{operation === 'deletingJobs' ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}{operation === 'deletingJobs' ? '正在清理…' : '确认删除记录及日志'}</button></>}><ul className="training-deletion-list">{pendingDeletion.map((job) => <li key={jobKey(job)}><strong>{job.name}</strong><small>{servers.find((server) => server.id === job.serverId)?.name ?? '服务器已移除'} · {TRAINING_STATUS_LABELS[job.status]}</small></li>)}</ul><p className="training-queue-copy">清理仅涉及 ModelGym 受管任务目录；项目脚本、数据集、模型权重和训练配置会保留。连接失败或进程尚未退出的项目将保留记录并报告原因。</p></Dialog>}

    {pendingQueue && <Dialog title="GPU 尚未空闲，是否后台排队？" description={`“${pendingQueue.profile.name}”当前无法立即启动。确认后将排队任务提交到服务器。`} onClose={() => { setPendingQueue(null); setFormError(null); setEditorOpen(true) }} busy={operation !== null} alert footer={<><button className="button button--secondary" disabled={operation !== null} onClick={() => { setPendingQueue(null); setFormError(null); setEditorOpen(true) }}>返回修改</button><button className="button button--primary" disabled={operation !== null} onClick={() => void confirmQueue()}>{operation === 'submitting' ? <LoaderCircle className="spin" size={14} /> : <Clock3 size={14} />}确认后台排队</button></>}><ul className="training-queue-reasons">{pendingQueue.reasons.map((reason, index) => <li key={`${index}:${reason}`}><AlertCircle size={15} />{reason}</li>)}</ul><p className="training-queue-copy">{pendingQueue.profile.datalistScriptPath?.trim() && '确认提交后立即生成 datalist，每个任务只执行一次；生成失败将停止任务。'}只等待你选择的 {pendingQueue.profile.gpuUuids.length} 张 GPU。同一服务器上的这些卡连续空闲 30 秒{pendingQueue.profile.datalistScriptPath?.trim() ? '且 datalist 生成成功后' : '后'}自动开始训练；关闭 ModelGym 或重启 Windows 不会取消服务器上的排队。</p><p className="training-queue-copy">可随时回到“训练任务”取消排队。服务器重启后，待运行队列可通过“恢复队列”继续；已中断的训练不会自动重跑。</p>{formError && <p className="training-error" role="alert">{formError}</p>}</Dialog>}

    {pendingAction && <Dialog title={pendingAction.kind === 'delete' ? '删除这份训练配置？' : pendingAction.kind === 'resume' ? '恢复服务器后台队列？' : cancellingRun ? '确认停止训练？' : '确认取消排队？'} onClose={() => setPendingAction(null)} busy={operation !== null} alert footer={<><button className="button button--secondary" disabled={operation !== null} onClick={() => setPendingAction(null)}>返回</button><button className={`button ${pendingAction.kind === 'resume' ? 'button--primary' : 'button--danger'}`} disabled={operation !== null} onClick={() => void confirmAction()}>{operation ? <LoaderCircle className="spin" size={14} /> : null}{pendingAction.kind === 'delete' ? '删除配置' : pendingAction.kind === 'resume' ? '恢复队列' : cancellingRun ? '停止训练' : '取消排队'}</button></>}><p className="training-queue-copy">{pendingAction.kind === 'delete' ? `将删除“${pendingAction.profile?.name}”的可复用配置，已提交的训练任务保留。` : pendingAction.kind === 'resume' ? `继续等待“${pendingAction.job?.name}”原先选定的 GPU。服务器后台会重新监测空闲条件，条件连续满足 30 秒后自动启动；不会新增重复任务。` : cancellingRun ? `将停止“${pendingAction.job?.name}”及其训练子进程。未保存的训练进度可能丢失。` : `“${pendingAction.job?.name}”将从服务器等待队列取消，不再自动启动。${pendingAction.job?.datalistStatus === 'running' ? '正在运行的 datalist 生成脚本及其子进程也会停止。' : ''}如果它刚刚进入运行状态，也会停止训练进程，未保存的进度可能丢失。`}</p>{formError && <p className="training-error" role="alert">{formError}</p>}</Dialog>}

    {logJob && <Dialog title={`${logJob.name} · 训练日志`} description={`${servers.find((server) => server.id === logJob.serverId)?.name ?? logJob.serverId} · 每 5 秒刷新末尾日志`} onClose={() => setLogJob(null)} wide footer={<><span className="training-log-path">{logJob.logPath || '服务器受管训练日志'}</span><button className="button button--secondary" disabled={logLoading} onClick={() => void loadLog(logJob)}><RefreshCw size={14} />刷新日志</button><button className="button button--primary" onClick={() => setLogJob(null)}>完成</button></>}>{logError && <p className="training-error" role="alert">{logError}</p>}<pre className="training-log" aria-label="训练日志内容">{logText || (logLoading ? '正在读取日志…' : '暂无日志。排队期间会显示等待原因，训练启动后输出会写入这里。')}</pre></Dialog>}
  </section>
}

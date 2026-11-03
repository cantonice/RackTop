import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { AlertCircle, Check, CheckCircle2, ChevronRight, Clock3, Dumbbell, FileCode2, FileCog, LoaderCircle, Pencil, Play, Plus, RefreshCw, ScrollText, Server as ServerIcon, Square, Trash2, X } from 'lucide-react'
import { api } from '../services/api'
import { trainingApi, trainingCommandPreview, trainingGpuFromSnapshot, trainingSnapshotIssue, validateTrainingProfile } from '../services/training'
import type { Server, Snapshot } from '../types/models'
import { isActiveTrainingJob, TRAINING_STATUS_LABELS } from '../types/training'
import type { TrainingGpu, TrainingJob, TrainingPathCheck, TrainingProfile } from '../types/training'

export interface TrainingViewProps {
  servers: Server[]
  snapshots: Record<string, Snapshot>
  onNotice: (message: string) => void
  onRefreshServer?: (serverId: string) => void | Promise<void>
}

function newProfile(servers: Server[]): TrainingProfile {
  const now = Math.floor(Date.now() / 1000)
  return { id: crypto.randomUUID(), name: '', serverId: servers.find((server) => server.status === 'online')?.id ?? servers[0]?.id ?? '', workingDirectory: '', interpreter: 'python -u', scriptPath: '', configPath: '', configMode: 'flag', configFlag: '--config', extraArguments: '', gpuUuids: [], minimumFreeMemoryGb: 8, createdAt: now, updatedAt: now }
}

function messageFor(error: unknown): string { return String(error).replace(/^Error:\s*/, '') }
function timestamp(value?: number | null): string { return value ? new Date(value * 1000).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—' }
function gpuNumbers(job: TrainingJob): string { return job.gpuIndices.length ? job.gpuIndices.map((index) => `GPU ${index}`).join(' · ') : `${job.gpuUuids.length} 张指定 GPU` }

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
  if (gpu.uuid.startsWith('unavailable-') || gpu.memoryTotalMb <= 0 || !Number.isFinite(gpu.utilization)) reasons.push('GPU 指标不可读')
  if (gpu.computeProcessCount > 0) reasons.push(`${gpu.computeProcessCount} 个计算进程`)
  if (gpu.utilization > 10) reasons.push(`利用率 ${Math.round(gpu.utilization)}% > 10%`)
  if ((gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024 < minimum) reasons.push(`空闲显存不足 ${minimum} GB`)
  return { ...gpu, eligible: reasons.length === 0, reasons }
}

export function TrainingView({ servers, snapshots, onNotice, onRefreshServer }: TrainingViewProps) {
  const [tab, setTab] = useState<'jobs' | 'profiles'>('jobs')
  const [jobFilter, setJobFilter] = useState<'active' | 'recent'>('active')
  const [profiles, setProfiles] = useState<TrainingProfile[]>([])
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [editorOpen, setEditorOpen] = useState(false)
  const [profile, setProfile] = useState<TrainingProfile>(() => newProfile(servers))
  const [memoryInput, setMemoryInput] = useState('8')
  const [operation, setOperation] = useState<'checking' | 'saving' | 'submitting' | 'cancelling' | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  const [check, setCheck] = useState<{ serverId: string; checkedAt: number; result: TrainingPathCheck } | null>(null)
  const [pendingQueue, setPendingQueue] = useState<{ jobId: string; profile: TrainingProfile; reasons: string[] } | null>(null)
  const [pendingAction, setPendingAction] = useState<{ kind: 'cancel' | 'delete' | 'resume'; job?: TrainingJob; profile?: TrainingProfile } | null>(null)
  const [logJob, setLogJob] = useState<TrainingJob | null>(null)
  const [logText, setLogText] = useState('')
  const [logError, setLogError] = useState<string | null>(null)
  const [logLoading, setLogLoading] = useState(false)
  const loadBusy = useRef(false)
  const alive = useRef(true)
  const logJobId = useRef<string | null>(null)
  const logRequest = useRef(0)
  const submissionIdentity = useRef<{ key: string; jobId: string } | null>(null)
  logJobId.current = logJob?.id ?? null

  const load = useCallback(async (manual = false) => {
    if (loadBusy.current) return
    loadBusy.current = true
    if (manual) setRefreshing(true)
    try {
      const [nextProfiles, nextJobs] = await Promise.all([trainingApi.listProfiles(), trainingApi.listJobs()])
      if (!alive.current) return
      setProfiles(nextProfiles)
      setJobs(nextJobs)
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

  const selectedServer = servers.find((server) => server.id === profile.serverId)
  const selectedSnapshot = snapshots[profile.serverId]
  const gpus = useMemo(() => {
    if (check?.serverId === profile.serverId && (!selectedSnapshot || check.checkedAt >= selectedSnapshot.timestamp)) return check.result.gpus.map((gpu) => currentGpuStatus(gpu, profile.minimumFreeMemoryGb))
    return selectedSnapshot ? trainingGpuFromSnapshot(selectedSnapshot, profile.minimumFreeMemoryGb) : []
  }, [check, profile.minimumFreeMemoryGb, profile.serverId, selectedSnapshot])
  const selectedGpus = gpus.filter((gpu) => profile.gpuUuids.includes(gpu.uuid))
  const preview = trainingCommandPreview(profile)
  const snapshotIssue = trainingSnapshotIssue(selectedServer, selectedSnapshot)
  const activeJobs = jobs.filter(isActiveTrainingJob)
  const queuedCount = activeJobs.filter((job) => job.status !== 'running').length
  const runningCount = activeJobs.filter((job) => job.status === 'running').length
  const visibleJobs = (jobFilter === 'active' ? activeJobs : jobs.filter((job) => !isActiveTrainingJob(job))).slice().sort((left, right) => right.createdAt - left.createdAt)

  function updateProfile(patch: Partial<TrainingProfile>) {
    setProfile((current) => ({ ...current, ...patch }))
    setFormError(null)
    if (patch.serverId !== undefined) setCheck(null)
  }

  function openEditor(source?: TrainingProfile) {
    const next = source ? structuredClone(source) : newProfile(servers)
    setProfile(next)
    setMemoryInput(String(next.minimumFreeMemoryGb))
    setFormError(null)
    setCheck(null)
    setEditorOpen(true)
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
      setProfiles((current) => [saved, ...current.filter((item) => item.id !== saved.id)])
      setProfile(saved)
      onNotice(`已保存训练配置“${saved.name}”`)
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  function acceptJob(job: TrainingJob) {
    submissionIdentity.current = null
    setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)])
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
        setProfiles((current) => current.filter((item) => item.id !== pendingAction.profile!.id))
        onNotice('已删除训练配置，已提交的任务不受影响')
      } else if (pendingAction.kind === 'resume' && pendingAction.job) {
        const job = pendingAction.job
        const result = await trainingApi.submit(job.id, job.profile, true)
        if (!result.job) throw new Error(result.reasons.join('；') || '服务器未接受恢复请求。')
        setJobs((current) => current.map((item) => item.id === job.id ? result.job! : item))
        onNotice('已恢复服务器后台队列，仍按原配置和 GPU 等待空闲')
      } else if (pendingAction.job) {
        const job = pendingAction.job
        const updated = await trainingApi.cancelJob(job.serverId, job.id)
        setJobs((current) => current.map((item) => item.id === job.id ? updated : item))
        onNotice(updated.status === 'running' ? '已发送停止请求，正在等待训练进程退出' : updated.status === 'cancelled' ? '已取消训练任务' : '任务状态已更新')
      }
      setPendingAction(null)
      void load()
    } catch (error) { setFormError(messageFor(error)) }
    finally { setOperation(null) }
  }

  const loadLog = useCallback(async (job: TrainingJob) => {
    const request = ++logRequest.current
    setLogLoading(true); setLogError(null)
    const current = () => alive.current && logJobId.current === job.id && logRequest.current === request
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
    <div className="training-summary"><div><span className="training-summary__icon"><Dumbbell size={17} /></span><span><small>正在训练</small><strong>{runningCount}<em>项任务</em></strong></span></div><div><span className="training-summary__icon is-waiting"><Clock3 size={17} /></span><span><small>等待资源</small><strong>{queuedCount}<em>项任务</em></strong></span></div><div className="training-summary__policy"><CheckCircle2 size={16} /><p><strong>等待空闲，自动开始</strong><small>指定 GPU 同时连续空闲 30 秒后启动；服务器后台排队不依赖客户端在线。</small></p></div></div>
    {tab === 'jobs' ? <section className="training-task-section" aria-label="训练任务列表"><div className="training-section-heading"><h2>训练任务</h2><div className="training-task-filter" role="group" aria-label="任务状态"><button className={jobFilter === 'active' ? 'is-active' : ''} aria-pressed={jobFilter === 'active'} onClick={() => setJobFilter('active')}>进行中</button><button className={jobFilter === 'recent' ? 'is-active' : ''} aria-pressed={jobFilter === 'recent'} onClick={() => setJobFilter('recent')}>最近结束</button></div></div>
      {loading ? <div className="training-empty" role="status"><LoaderCircle className="spin" size={24} /><strong>正在读取训练任务…</strong></div> : visibleJobs.length ? <div className="training-job-list">{visibleJobs.map((job) => <article className="training-job" key={job.id}><div className={`training-job__status is-${job.status}`}>{job.status === 'running' ? <Dumbbell size={18} /> : job.status === 'queued' || job.status === 'stabilizing' ? <Clock3 size={18} /> : job.status === 'completed' ? <CheckCircle2 size={18} /> : <Square size={17} />}</div><div className="training-job__content"><header><h3>{job.name}</h3><span className={`training-status is-${job.status}`}>{TRAINING_STATUS_LABELS[job.status]}</span></header><p className="training-job__meta"><span><ServerIcon size={12} />{servers.find((server) => server.id === job.serverId)?.name ?? '服务器已移除'}</span><span>{gpuNumbers(job)}</span><span>提交于 {timestamp(job.createdAt)}</span>{job.startedAt && <span>启动于 {timestamp(job.startedAt)}</span>}</p><p className={`training-job__reason${job.connectionError ? ' is-error' : ''}`}>{job.connectionError ? `连接失败：${job.connectionError}。服务器队列可能仍在运行，恢复连接后继续查询。` : job.waitingReason || (job.status === 'running' ? '训练进程正在服务器后台运行' : job.exitCode != null ? `退出码 ${job.exitCode}` : '任务状态已更新')}</p><details><summary>查看启动命令 <ChevronRight size={12} /></summary><pre>{job.command}</pre></details></div><div className="training-job__actions">{(job.status === 'queued' || job.status === 'stabilizing') && <button className="button button--secondary" onClick={() => { setFormError(null); setPendingAction({ kind: 'resume', job }) }}><Play size={12} />恢复队列</button>}<button className="button button--secondary" onClick={() => { setLogError(null); setLogJob(job) }} aria-label={`查看 ${job.name} 日志`}><ScrollText size={14} />日志</button>{isActiveTrainingJob(job) && <button className={`button ${job.status === 'running' ? 'button--danger' : 'button--secondary'}`} onClick={() => { setFormError(null); setPendingAction({ kind: 'cancel', job }) }}><Square size={12} />{job.status === 'running' ? '停止训练' : '取消排队'}</button>}</div></article>)}</div> : <div className="training-empty"><span className="training-empty__mark"><Dumbbell size={32} /></span><h3>{jobFilter === 'active' ? '让下一次训练准备就绪' : '还没有结束的训练任务'}</h3><p>{jobFilter === 'active' ? '选择服务器上的训练脚本与配置文件，再指定 GPU。资源繁忙时，可以确认加入后台队列。' : '完成、失败和已取消的任务会保留在这里。'}</p>{jobFilter === 'active' && <button className="button button--primary" onClick={() => openEditor()}><Plus size={15} />创建第一项训练</button>}</div>}
    </section> : <section className="training-task-section" aria-label="训练配置列表"><div className="training-section-heading"><div><h2>训练配置</h2><p>保存脚本、环境与显存要求，下一次直接选择运行。</p></div></div>{profiles.length ? <div className="training-profile-grid">{profiles.map((saved) => <article className="training-profile" key={saved.id}><div className="training-profile__heading"><span><FileCode2 size={20} /></span><h3>{saved.name}</h3><button className="icon-button" aria-label={`编辑 ${saved.name}`} onClick={() => openEditor(saved)}><Pencil size={15} /></button></div><p><ServerIcon size={13} />{servers.find((server) => server.id === saved.serverId)?.name ?? '服务器已移除'}</p><dl><dt>训练脚本</dt><dd title={saved.scriptPath}>{saved.scriptPath}</dd><dt>配置文件</dt><dd title={saved.configPath}>{saved.configPath}</dd><dt>资源要求</dt><dd>{saved.gpuUuids.length || '待选择'} 张 GPU · 每卡空闲 ≥ {saved.minimumFreeMemoryGb} GB</dd></dl><footer><button className="icon-button" aria-label={`删除 ${saved.name}`} onClick={() => { setFormError(null); setPendingAction({ kind: 'delete', profile: saved }) }}><Trash2 size={14} /></button><button className="button button--secondary" onClick={() => openEditor(saved)}><Play size={13} />配置并启动</button></footer></article>)}</div> : <div className="training-empty"><FileCog size={30} /><h3>保存一套可重复使用的训练配置</h3><p>训练配置独立于“我的项目”，使用服务器上现有的脚本和配置文件。</p><button className="button button--primary" onClick={() => openEditor()}><Plus size={15} />添加训练配置</button></div>}</section>}

    {editorOpen && <Dialog title="配置模型训练" description="使用服务器上已有的脚本和配置文件，选择本次训练固定使用的 GPU。" onClose={() => setEditorOpen(false)} busy={operation !== null} wide footer={<><p className="training-footer-hint"><Clock3 size={14} />启动前会重新检查资源；繁忙时先征求你的排队确认。</p><button className="button button--secondary" disabled={operation !== null} onClick={() => setEditorOpen(false)}>取消</button><button className="button button--secondary" disabled={operation !== null} onClick={() => void saveProfile()}>{operation === 'saving' ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}保存配置</button><button className="button button--primary" disabled={operation !== null || servers.length === 0} onClick={() => void submitTraining()}>{operation === 'submitting' ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}{operation === 'submitting' ? '正在检查…' : '开始训练'}</button></>}>
      <div className="training-editor" inert={operation !== null}><div className="training-editor__configuration"><section className="training-form-section"><header><span>1</span><div><h3>训练脚本与配置</h3><p>文件路径相对于工作目录，也可以填写绝对路径。</p></div></header><div className="training-fields"><label>训练名称<input data-autofocus value={profile.name} onChange={(event) => updateProfile({ name: event.target.value })} placeholder="例如：ResNet50 · 基线训练" /></label><label>训练服务器<select value={profile.serverId} onChange={(event) => updateProfile({ serverId: event.target.value, gpuUuids: [] })}><option value="">选择服务器</option>{servers.map((server) => <option value={server.id} key={server.id}>{server.name}{server.status === 'offline' ? ' · 离线' : ''}</option>)}</select></label><label className="training-field--full">工作目录<input className="mono" value={profile.workingDirectory} onChange={(event) => updateProfile({ workingDirectory: event.target.value })} placeholder="/home/user/projects/my-model" spellCheck={false} /></label><label className="training-field--full">启动前缀<input className="mono" value={profile.interpreter} onChange={(event) => updateProfile({ interpreter: event.target.value })} placeholder="python -u / conda run -n train python / torchrun" spellCheck={false} /><small>支持 Python、Conda run 和 torchrun。按命令参数解析，不支持 source、&&、管道或重定向。</small></label><label className="training-field--full">训练脚本<input className="mono" value={profile.scriptPath} onChange={(event) => updateProfile({ scriptPath: event.target.value })} placeholder="train.py 或 /home/user/project/train.py" spellCheck={false} /></label><label className="training-field--full">配置文件<input className="mono" value={profile.configPath} onChange={(event) => updateProfile({ configPath: event.target.value })} placeholder="configs/train.yaml" spellCheck={false} /></label><fieldset className="training-config-mode training-field--full"><legend>配置文件传入方式</legend><label><input type="radio" name="training-config-mode" checked={profile.configMode === 'flag'} onChange={() => updateProfile({ configMode: 'flag' })} />参数式</label><label><input type="radio" name="training-config-mode" checked={profile.configMode === 'positional'} onChange={() => updateProfile({ configMode: 'positional' })} />位置式（紧跟脚本）</label>{profile.configMode === 'flag' && <label className="training-config-flag">参数名<input aria-label="配置参数名" value={profile.configFlag} onChange={(event) => updateProfile({ configFlag: event.target.value })} placeholder="--config" spellCheck={false} /></label>}</fieldset><label className="training-field--full">附加参数<span className="training-optional">可选</span><input className="mono" value={profile.extraArguments} onChange={(event) => updateProfile({ extraArguments: event.target.value })} placeholder="--epochs 100 --batch-size 32" spellCheck={false} /><small>参数中的空格可用引号包裹；不支持多行 shell 命令。</small></label></div><div className="training-path-check"><button className="button button--secondary" disabled={operation !== null || servers.length === 0} onClick={() => void inspectPaths()}>{operation === 'checking' ? <LoaderCircle className="spin" size={14} /> : <CheckCircle2 size={14} />}检查路径与环境</button>{check && <p role="status" className={check.result.ok ? 'is-success' : 'is-error'}>{check.result.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}{check.result.message || (check.result.ok ? '路径与环境检查通过' : '检查未通过')}</p>}</div></section></div>
      <div className="training-editor__resources"><section className="training-form-section"><header><span>2</span><div><h3>选择固定 GPU</h3><p>所有选中的卡必须在同一台服务器上同时满足空闲条件。</p></div></header><label className="training-memory-field">每卡最低空闲显存<div><input type="number" min="0" step="0.5" value={memoryInput} onChange={(event) => { setMemoryInput(event.target.value); if (event.target.value !== '') updateProfile({ minimumFreeMemoryGb: Number(event.target.value) }) }} onBlur={() => { if (!Number.isFinite(Number(memoryInput)) || memoryInput.trim() === '') { setMemoryInput(String(profile.minimumFreeMemoryGb)) } }} /><span>GB</span></div></label><p className="training-gpu-rule">无计算进程 · GPU 利用率 ≤ 10% · 显存达到要求，连续稳定 30 秒。繁忙卡也可以选中。</p>{snapshotIssue && <p className="training-snapshot-note"><AlertCircle size={13} />{snapshotIssue}</p>}<div className="training-gpu-grid" aria-label="选择训练 GPU">{gpus.map((gpu) => <label className={`training-gpu${profile.gpuUuids.includes(gpu.uuid) ? ' is-selected' : ''}${!gpu.eligible ? ' is-busy' : ''}`} key={gpu.uuid}><input type="checkbox" checked={profile.gpuUuids.includes(gpu.uuid)} aria-label={`GPU ${gpu.index} ${gpu.name}`} onChange={(event) => updateProfile({ gpuUuids: event.target.checked ? [...profile.gpuUuids, gpu.uuid] : profile.gpuUuids.filter((uuid) => uuid !== gpu.uuid) })} /><div><strong>GPU {gpu.index}<span className={gpu.eligible ? 'is-free' : 'is-busy'}>{gpu.eligible ? '当前空闲' : '需等待'}</span></strong><small title={gpu.name}>{gpu.name}</small><p><b>{Math.max(0, (gpu.memoryTotalMb - gpu.memoryUsedMb) / 1024).toFixed(1)}</b> / {(gpu.memoryTotalMb / 1024).toFixed(1)} GB 可用 <em>{Math.round(gpu.utilization)}% UTL</em></p>{gpu.reasons.length > 0 && <span className="training-gpu__reason">{gpu.reasons.join(' · ')}</span>}</div></label>)}</div>{!gpus.length && <div className="training-gpu-empty"><ServerIcon size={21} /><p>{servers.length ? '选择服务器后，点击“检查路径与环境”获取可用 GPU。' : '先从左侧添加训练服务器，再回来配置训练。'}</p></div>}{profile.gpuUuids.some((uuid) => !gpus.some((gpu) => gpu.uuid === uuid)) && <p className="training-snapshot-note">已保存的部分 GPU 暂未出现在当前快照中，提交时会重新核对 UUID。</p>}<p className="training-selection-count">已选 <strong>{profile.gpuUuids.length}</strong> 张 GPU{selectedGpus.some((gpu) => !gpu.eligible) && ' · 部分资源繁忙，开始训练时将询问是否排队'}</p></section><section className="training-form-section training-command"><header><span>3</span><div><h3>最终启动命令</h3><p>先核对脚本、配置参数与卡号，再提交。</p></div></header><pre aria-label="最终训练命令"><code>{preview}</code></pre><p>排队锁定本次配置与 GPU UUID，不会自动换卡。关闭客户端后，服务器继续监测和启动任务。</p></section></div></div>
      {formError && <p className="training-error" role="alert"><AlertCircle size={16} />{formError}</p>}
    </Dialog>}

    {pendingQueue && <Dialog title="GPU 尚未空闲，是否后台排队？" description={`“${pendingQueue.profile.name}”当前无法立即启动。确认后将排队任务提交到服务器。`} onClose={() => { setPendingQueue(null); setFormError(null); setEditorOpen(true) }} busy={operation !== null} alert footer={<><button className="button button--secondary" disabled={operation !== null} onClick={() => { setPendingQueue(null); setFormError(null); setEditorOpen(true) }}>返回修改</button><button className="button button--primary" disabled={operation !== null} onClick={() => void confirmQueue()}>{operation === 'submitting' ? <LoaderCircle className="spin" size={14} /> : <Clock3 size={14} />}确认后台排队</button></>}><ul className="training-queue-reasons">{pendingQueue.reasons.map((reason, index) => <li key={`${index}:${reason}`}><AlertCircle size={15} />{reason}</li>)}</ul><p className="training-queue-copy">只等待你选择的 {pendingQueue.profile.gpuUuids.length} 张 GPU。同一服务器上的这些卡连续空闲 30 秒后自动开始训练；关闭 ModelGym 或重启 Windows 不会取消服务器上的排队。</p><p className="training-queue-copy">可随时回到“训练任务”取消排队。服务器重启后，待运行队列可通过“恢复队列”继续；已中断的训练不会自动重跑。</p>{formError && <p className="training-error" role="alert">{formError}</p>}</Dialog>}

    {pendingAction && <Dialog title={pendingAction.kind === 'delete' ? '删除这份训练配置？' : pendingAction.kind === 'resume' ? '恢复服务器后台队列？' : cancellingRun ? '确认停止训练？' : '确认取消排队？'} onClose={() => setPendingAction(null)} busy={operation !== null} alert footer={<><button className="button button--secondary" disabled={operation !== null} onClick={() => setPendingAction(null)}>返回</button><button className={`button ${pendingAction.kind === 'resume' ? 'button--primary' : 'button--danger'}`} disabled={operation !== null} onClick={() => void confirmAction()}>{operation ? <LoaderCircle className="spin" size={14} /> : null}{pendingAction.kind === 'delete' ? '删除配置' : pendingAction.kind === 'resume' ? '恢复队列' : cancellingRun ? '停止训练' : '取消排队'}</button></>}><p className="training-queue-copy">{pendingAction.kind === 'delete' ? `将删除“${pendingAction.profile?.name}”的可复用配置，已提交的训练任务保留。` : pendingAction.kind === 'resume' ? `继续等待“${pendingAction.job?.name}”原先选定的 GPU。服务器后台会重新监测空闲条件，条件连续满足 30 秒后自动启动；不会新增重复任务。` : cancellingRun ? `将停止“${pendingAction.job?.name}”及其训练子进程。未保存的训练进度可能丢失。` : `“${pendingAction.job?.name}”将从服务器等待队列取消，不再自动启动。如果它刚刚进入运行状态，也会停止训练进程，未保存的进度可能丢失。`}</p>{formError && <p className="training-error" role="alert">{formError}</p>}</Dialog>}

    {logJob && <Dialog title={`${logJob.name} · 训练日志`} description={`${servers.find((server) => server.id === logJob.serverId)?.name ?? logJob.serverId} · 每 5 秒刷新末尾日志`} onClose={() => setLogJob(null)} wide footer={<><span className="training-log-path">{logJob.logPath || '服务器受管训练日志'}</span><button className="button button--secondary" disabled={logLoading} onClick={() => void loadLog(logJob)}><RefreshCw size={14} />刷新日志</button><button className="button button--primary" onClick={() => setLogJob(null)}>完成</button></>}>{logError && <p className="training-error" role="alert">{logError}</p>}<pre className="training-log" aria-label="训练日志内容">{logText || (logLoading ? '正在读取日志…' : '暂无日志。排队期间会显示等待原因，训练启动后输出会写入这里。')}</pre></Dialog>}
  </section>
}

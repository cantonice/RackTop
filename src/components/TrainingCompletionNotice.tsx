import { CheckCircle2, AlertCircle, X } from 'lucide-react'
import type { TrainingCompletionAlert } from '../hooks/useTrainingCompletionNotifications'

export function TrainingCompletionNotice({ alert, remaining, serverName, onDismiss, onView }: { alert: TrainingCompletionAlert; remaining: number; serverName: string; onDismiss: () => void; onView: () => void }) {
  return <section className={`training-completion-notice is-${alert.job.status}`} role="status" aria-live="polite" aria-label={alert.title}>
    <header>{alert.job.status === 'completed' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}<strong>{alert.title}</strong><button className="icon-button" onClick={onDismiss} aria-label="关闭训练结束提醒"><X size={16} /></button></header>
    <p>{alert.body}</p><small>{serverName}{remaining > 1 ? ` · 还有 ${remaining - 1} 条结果提醒` : ''}</small>
    <footer><button className="button button--secondary" onClick={onDismiss}>知道了</button><button className="button button--primary" onClick={onView}>查看训练任务</button></footer>
  </section>
}

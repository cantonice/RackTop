import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'
import { trainingApi } from '../services/training'
import { isActiveTrainingJob } from '../types/training'
import type { TrainingJob } from '../types/training'

const STORAGE_KEY = 'modelgym.trainingCompletionNotifications.v1'
const POLL_INTERVAL_MS = 10_000

interface Observations {
  active: Set<string>
  seen: Set<string>
}

export interface TrainingCompletionAlert {
  id: string
  job: TrainingJob
  title: string
  body: string
}

function jobIdentity(job: TrainingJob): string {
  return JSON.stringify([job.serverId, job.id])
}

function readObservations(): Observations {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (value && typeof value === 'object' && 'active' in value && 'seen' in value && Array.isArray(value.active) && Array.isArray(value.seen)) {
      return { active: new Set(value.active.filter((item): item is string => typeof item === 'string')), seen: new Set(value.seen.filter((item): item is string => typeof item === 'string')) }
    }
  } catch { /* Keep notification tracking usable when local storage is unavailable. */ }
  return { active: new Set(), seen: new Set() }
}

function saveObservations(value: Observations): void {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ active: [...value.active], seen: [...value.seen] })) }
  catch { /* The in-memory record still prevents duplicate reminders this session. */ }
}

function completionAlert(job: TrainingJob): TrainingCompletionAlert {
  const successful = job.status === 'completed'
  const reason = job.waitingReason || (job.exitCode != null ? `退出码 ${job.exitCode}` : '请查看训练日志。')
  return { id: jobIdentity(job), job, title: successful ? '训练已完成' : '训练失败', body: successful ? `“${job.name}”已完成。` : `“${job.name}”失败：${reason}` }
}

export function useTrainingCompletionNotifications(enabled = true): { alerts: TrainingCompletionAlert[]; dismiss: (id: string) => void; observeJob: (job: TrainingJob) => void } {
  const [alerts, setAlerts] = useState<TrainingCompletionAlert[]>([])
  const observationsRef = useRef<Observations | null>(null)
  const baselineRef = useRef(new Set<string>())
  const pollingRef = useRef(false)
  const dismiss = useCallback((id: string) => {
    const observations = observationsRef.current ?? (observationsRef.current = readObservations())
    observations.active.delete(id)
    observations.seen.add(id)
    baselineRef.current.delete(id)
    saveObservations(observations)
    setAlerts((current) => current.filter((alert) => alert.id !== id))
  }, [])
  const observeJob = useCallback((job: TrainingJob) => {
    const observations = observationsRef.current ?? (observationsRef.current = readObservations())
    const id = jobIdentity(job)
    // A very short task can reach the polling response before submit replies.
    // Explicit acceptance distinguishes that task from the historical baseline.
    if (baselineRef.current.delete(id)) observations.seen.delete(id)
    if (!observations.seen.has(id)) {
      observations.active.add(id)
      saveObservations(observations)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    const observations = observationsRef.current ?? (observationsRef.current = readObservations())
    let disposed = false
    async function poll() {
      if (pollingRef.current) return
      pollingRef.current = true
      try {
        const jobs = await trainingApi.listJobs()
        if (disposed) return
        const completed: TrainingCompletionAlert[] = []
        let changed = false
        for (const job of jobs) {
          const id = jobIdentity(job)
          if (isActiveTrainingJob(job)) {
            if (!observations.seen.has(id) && !observations.active.has(id)) { observations.active.add(id); changed = true }
            continue
          }
          // A cached terminal status returned with an SSH error is not confirmation.
          if (job.connectionError) continue
          if (job.status !== 'completed' && job.status !== 'failed' && job.status !== 'cancelled') continue
          const previouslyActive = observations.active.delete(id)
          if (previouslyActive) changed = true
          if (observations.seen.has(id)) continue
          observations.seen.add(id)
          if (!previouslyActive) baselineRef.current.add(id)
          changed = true
          if (previouslyActive && job.status !== 'cancelled') completed.push(completionAlert(job))
        }
        // Persist before dispatch so another poll/reopen cannot repeat a reminder.
        if (changed) saveObservations(observations)
        if (completed.length) {
          setAlerts((current) => [...current, ...completed])
          for (const alert of completed) {
            try { void api.notify(alert.title, alert.body, { kind: 'training', jobId: alert.job.id, serverId: alert.job.serverId }).catch(() => {}) }
            catch { /* System notification failures must not hide the in-app alert. */ }
          }
        }
      } catch { /* Connection failures leave observed active jobs intact for retry. */ }
      finally { pollingRef.current = false }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), POLL_INTERVAL_MS)
    return () => { disposed = true; window.clearInterval(timer) }
  }, [enabled])

  return { alerts, dismiss, observeJob }
}

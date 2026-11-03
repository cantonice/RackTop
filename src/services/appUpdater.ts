import type { DownloadEvent, Update } from '@tauri-apps/plugin-updater'
import { relaunch } from '@tauri-apps/plugin-process'

export type DesktopAppUpdate = Update
export type DesktopDownloadEvent = DownloadEvent

// Keep the old data identity for upgrades, but never install RackTop releases.
// A future ModelGym update service must be configured before this is enabled.
export const MODEL_GYM_UPDATES_ENABLED = false

export async function checkDesktopAppUpdate(): Promise<DesktopAppUpdate | null> {
  return null
}

export function relaunchUpdatedApp() {
  return relaunch()
}

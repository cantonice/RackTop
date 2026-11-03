import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { api } from './services/api'
import { detectAppPlatform } from './utils/platform'
import { getCurrentWindow } from '@tauri-apps/api/window'
import { syncLinuxWindowShape } from './utils/linuxWindowShape'
import './styles.css'
import './training.css'

document.documentElement.dataset.platform = detectAppPlatform(api.isDesktop, navigator.userAgent)
if (document.documentElement.dataset.platform === 'linux') {
  void syncLinuxWindowShape('linux', getCurrentWindow(), document.documentElement).catch(console.error)
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

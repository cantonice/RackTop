// @vitest-environment node
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { expect, it } from 'vitest'

const read = path => readFileSync(path, 'utf8').replaceAll('\r\n', '\n')
const config = JSON.parse(read('src-tauri/tauri.conf.json'))
const nsis = read('src-tauri/windows/installer.nsi')
const hooks = read('src-tauri/windows/hooks.nsh')
const wix = read('src-tauri/windows/modelgym.wxs')

function legacyUpgradeCode() {
  const namespace = Buffer.from('6ba7b8109dad11d180b400c04fd430c8', 'hex')
  const bytes = createHash('sha1').update(Buffer.concat([namespace, Buffer.from('RackTop.exe.app.x64')])).digest().subarray(0, 16)
  bytes[6] = (bytes[6] & 15) | 80
  bytes[8] = (bytes[8] & 63) | 128
  const hex = bytes.toString('hex')
  return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join('-')
}

it('keeps the installed RackTop data and MSI upgrade identities after the product is renamed', () => {
  expect(config.identifier).toBe('com.racktop.desktop')
  expect(config.mainBinaryName).toBe('ModelGym')
  expect(config.bundle.windows.wix.upgradeCode).toBe(legacyUpgradeCode())
  expect(nsis).toContain('!define UNINSTKEY "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\RackTop"')
  expect(nsis).toContain('!define MANUPRODUCTKEY "${MANUKEY}\\RackTop"')
  // A slash before a Handlebars expression is escaped in the template; the
  // literal final segment is not. Double literal slashes fail WiX ICE03.
  expect(wix).toContain(String.raw`Software\\{{manufacturer}}\RackTop`)
  expect(wix).not.toContain(String.raw`\\RackTop`)
  expect(wix).toContain('<MajorUpgrade Schedule="afterInstallInitialize"')
})

it('uses the old uninstaller in update mode and limits legacy cleanup to owned application files', () => {
  const upgrade = nsis.slice(nsis.indexOf('  reinst_uninstall:'), nsis.indexOf('  reinst_done:'))
  expect(upgrade).toContain('StrCpy $R1 "$R1 /UPDATE"')
  expect(upgrade).not.toContain('${IfThen} $UpdateMode = 1')
  expect(hooks).toContain('!insertmacro IsShortcutTarget')
  expect(hooks).toContain('$ModelGymLegacyDisplayName == "RackTop"')
  expect(hooks).toContain('Delete "$ModelGymLegacyInstallDir\\RackTop.exe"')
  expect(hooks).not.toMatch(/(?:RmDir|Delete)\s+\/r/i)
  expect(hooks).not.toContain('$APPDATA')
  expect(hooks).not.toContain('$LOCALAPPDATA')
})

it('cannot build or fetch upstream RackTop update artifacts through any platform config', () => {
  expect(config.bundle.createUpdaterArtifacts).toBe(false)
  for (const file of ['src-tauri/tauri.conf.json', 'src-tauri/tauri.windows.conf.json', 'src-tauri/tauri.linux.conf.json']) {
    const platformConfig = JSON.parse(read(file))
    expect(platformConfig.plugins?.updater?.endpoints ?? []).toEqual([])
  }
  expect(read('src-tauri/src/lib.rs')).not.toContain('.plugin(tauri_plugin_updater::Builder::new().build())')
})

it('ships one synchronized application version without changing Cargo crate compatibility names', () => {
  const npm = JSON.parse(read('package.json'))
  const lock = JSON.parse(read('package-lock.json'))
  expect(config.version).toBe(npm.version)
  expect(lock.version).toBe(npm.version)
  expect(lock.packages[''].version).toBe(npm.version)
  expect(read('src-tauri/Cargo.toml')).toContain(`name = "racktop"\nversion = "${npm.version}"`)
  expect(read('src-tauri/Cargo.lock')).toContain(`name = "racktop"\nversion = "${npm.version}"`)
})

// Run: node scripts/export-modelgym-icons.mjs
// Uses the project's installed Tauri CLI and Node built-ins only.
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { inflateSync } from 'node:zlib'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const icons = join(root, 'src-tauri/icons')
const output = join(root, 'artifacts/modelgym-icons')
const source = readFileSync(join(icons, 'icon.svg'), 'utf8')
mkdirSync(output, { recursive: true })

function render(input, destination, sizes = []) {
  const args = [join(root, 'node_modules/@tauri-apps/cli/tauri.js'), 'icon', input, '--output', destination]
  for (const size of sizes) args.push('--png', String(size))
  const result = spawnSync(process.execPath, args, { cwd: root, encoding: 'utf8' })
  if (result.status !== 0) throw new Error(result.stderr || result.stdout || 'Tauri icon export failed')
}

// Replace desktop variants, including Windows ICO and Store logos.
const desktop = join(output, 'desktop')
render(join(icons, 'icon.svg'), desktop)
// The CLI emits ICNS representations in hash-map order; keep asset diffs stable.
const icnsPath = join(desktop, 'icon.icns')
const icns = readFileSync(icnsPath)
const representations = []
for (let offset = 8; offset < icns.length;) {
  const length = icns.readUInt32BE(offset + 4)
  if (length < 8 || offset + length > icns.length) throw new Error('Invalid ICNS representation')
  representations.push(icns.subarray(offset, offset + length))
  offset += length
}
representations.sort((a, b) => Buffer.compare(a.subarray(0, 4), b.subarray(0, 4)))
writeFileSync(icnsPath, Buffer.concat([icns.subarray(0, 8), ...representations]))
for (const entry of readdirSync(desktop, { withFileTypes: true })) {
  if (entry.isFile()) copyFileSync(join(desktop, entry.name), join(icons, entry.name))
}
writeFileSync(join(root, 'public/icon.svg'), source)
writeFileSync(join(icons, 'icon-macos.svg'), source)
for (const [name, size] of [['icon-linux', 256], ['32x32-linux', 32], ['128x128-linux', 128], ['128x128@2x-linux', 256]]) {
  const input = join(icons, size === 256 ? '128x128@2x.png' : `${size}x${size}.png`)
  copyFileSync(input, join(icons, `${name}.png`))
}

// Decode the RGBA PNG emitted by Tauri to the raw pixels embedded by Rust.
function rgbaFromPng(file) {
  const png = readFileSync(file)
  const chunks = []
  let width, height
  for (let offset = 8; offset < png.length;) {
    const length = png.readUInt32BE(offset)
    const type = png.toString('ascii', offset + 4, offset + 8)
    const data = png.subarray(offset + 8, offset + 8 + length)
    if (type === 'IHDR') {
      width = data.readUInt32BE(0)
      height = data.readUInt32BE(4)
      if (data[8] !== 8 || data[9] !== 6 || data[12] !== 0) throw new Error('Expected an 8-bit non-interlaced RGBA PNG')
    }
    if (type === 'IDAT') chunks.push(data)
    offset += length + 12
  }
  const packed = inflateSync(Buffer.concat(chunks))
  const stride = width * 4
  const pixels = Buffer.alloc(stride * height)
  const paeth = (a, b, c) => {
    const p = a + b - c
    const distances = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)]
    return distances[0] <= distances[1] && distances[0] <= distances[2] ? a : distances[1] <= distances[2] ? b : c
  }
  for (let y = 0; y < height; y++) {
    const filter = packed[y * (stride + 1)]
    for (let x = 0; x < stride; x++) {
      const i = y * stride + x
      const left = x >= 4 ? pixels[i - 4] : 0
      const up = y ? pixels[i - stride] : 0
      const corner = y && x >= 4 ? pixels[i - stride - 4] : 0
      const predictors = [0, left, up, Math.floor((left + up) / 2), paeth(left, up, corner)]
      if (filter > 4) throw new Error(`Unsupported PNG filter: ${filter}`)
      pixels[i] = (packed[y * (stride + 1) + x + 1] + predictors[filter]) & 255
    }
  }
  if (width !== 36 || height !== 36) throw new Error('Tray icon must be 36×36')
  return pixels
}

const tray = join(icons, 'tray')
const color = source.replace('width="1024" height="1024"', 'width="36" height="36"')
const glyph = source.match(/  <g transform=[\s\S]*?<\/g>/)[0].replace('stroke="white"', 'stroke="black"')
const mask = `<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 1024 1024">
  <defs><mask id="cutout" maskUnits="userSpaceOnUse" x="0" y="0" width="1024" height="1024">
    <rect x="72" y="72" width="880" height="880" rx="247.5" fill="white"/>
${glyph}
  </mask></defs>
  <rect width="1024" height="1024" fill="black" mask="url(#cutout)"/>
</svg>
`
for (const [name, svg] of [['color', color], ['macos-template', mask]]) {
  const path = join(tray, `${name}.svg`)
  writeFileSync(path, svg)
  const destination = join(output, name)
  render(path, destination, [36])
  const png = join(destination, '36x36.png')
  if (!existsSync(png)) throw new Error('Tauri did not create the tray PNG')
  copyFileSync(png, join(tray, `${name}.png`))
  writeFileSync(join(tray, `${name}.rgba`), rgbaFromPng(png))
}
console.log('Exported ModelGym desktop, installer, favicon and tray icons.')

# LiveTranslate Windows one-command packaging (SelfServe P0-A1).
#
# Runs the full release chain on a clean machine:
#   1. sync dev tools           6. locate Inno Setup (ISCC)
#   2. variant requirements    7. compile setup.exe
#   3. main app spec           8. frozen --smoke
#   4. bundle tools\uv.exe     9. artifact list + sha256
#   5. bundle tools\python
#
# Usage:  pwsh -File scripts/package_windows.ps1 [-Version 0.1.0] [-SkipSmoke] [-SetupOnly]
# The same script runs in CI (release.yml) — no divergence between local and CI.
#
# Engines ship on-demand — the single installer bundles the app + embedded
# toolchain ONLY (base deps, tools\uv.exe, tools\python). The user installs,
# opens the app, and the runtime engine install (core/uv_runner.install_variant)
# pulls the pinned variant requirements (runtime/requirements/*.txt, generated
# in step 2) into data\engines on their hardware/network. No offline/preload
# component, no portable zip / sidecar.
# -SetupOnly builds ONLY the installer: sync → requirements → main spec →
#   uv.exe → iscc (skips smoke).

param(
    [string]$Version = "",
    [switch]$SkipSmoke,
    [switch]$SetupOnly,
    [switch]$QuickOnedir
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Version) {
    $Version = (uv run python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])").Trim()
}
Write-Host "== LiveTranslate packaging v$Version ==" -ForegroundColor Cyan

function Get-RemoteFile {
    # Try URLs in order until one yields a file of at least $MinBytes (flaky
    # networks truncate downloads); cleans up failed attempts.
    param([string[]]$Urls, [string]$OutFile, [long]$MinBytes)
    foreach ($u in $Urls) {
        try {
            Write-Host "  fetching $u" -ForegroundColor Yellow
            Invoke-WebRequest -Uri $u -OutFile $OutFile -TimeoutSec 300 -ErrorAction Stop
            if ((Test-Path $OutFile) -and (Get-Item $OutFile).Length -ge $MinBytes) { return $true }
        } catch {
            # fall through to the next URL
        }
        Remove-Item $OutFile -Force -ErrorAction SilentlyContinue
    }
    return $false
}

# The clean-build sync (step 1) prunes engine extras from the DEV venv.
# Remember whether they were installed so they can be restored afterwards —
# packaging must not break the developer's working environment (measured
# footgun: funasr/faster_whisper missing after every package run).
$script:HadEngines = (Test-Path ".venv\Lib\site-packages\funasr")
trap {
    if ($script:HadEngines -and -not (Test-Path ".venv\Lib\site-packages\funasr")) {
        Write-Host "packaging failed — restoring engine extras..." -ForegroundColor Yellow
        uv sync --extra engine-funasr --extra engine-whisper | Out-Null
    }
    break
}

# --- 0. local quick onedir (fast test path) ---------------------------------
# A frozen onedir only for local smoke-testing: no variant requirements, no
# Inno setup, no tool-bundling, no sha256. It does NOT re-sync the dev venv,
# so the developer's engine extras are never pruned. Requires an existing dev
# environment (uv sync --group dev) with pyinstaller available.
if ($QuickOnedir) {
    # The spec bundles runtime/requirements as datas; ensure the anchor dir
    # exists so a clean checkout (never full-packaged) still builds.
    $null = New-Item -ItemType Directory -Force "$Root\runtime\requirements"
    Write-Host "[quick] building onedir only (skips variant requirements / Inno / smoke)" -ForegroundColor Cyan
    & uv run pyinstaller packaging/livetranslate.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "quick onedir bundled spec failed" }
    Write-Host "== quick onedir ready: dist\LiveTranslate\ ==" -ForegroundColor Green
    return
}

# --- 1. dev tools -----------------------------------------------------------
Write-Host "[1/9] uv sync (dev group)" -ForegroundColor Cyan
uv sync --locked --group dev
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

# --- 2. variant requirements (embedded uv engine installs, P1-B2) -----------
# The frozen bundle ships the pinned variant requirements with it (spec datas
# runtime/requirements -> collect into _MEIPASS). Regenerate them BEFORE the
# main spec so they always land in the bundle; the runtime engine install
# (core/uv_runner.install_variant) reads these on the user machine and never
# re-resolves. Per-variant failures are soft (skipped), not fatal — the
# manager only offers variants whose requirements file exists.
Write-Host "[2/9] generate variant requirements (cpu, cu126)" -ForegroundColor Cyan
uv run python scripts/build_runtime_variants.py
if ($LASTEXITCODE -ne 0) { throw "build_runtime_variants failed" }

# --- 3. PyInstaller main onedir ---------------------------------------------
Write-Host "[3/9] build main app onedir" -ForegroundColor Cyan
uv run pyinstaller packaging/livetranslate.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "main spec failed" }

# --- 4. bundled uv (engine runtime installer, SelfServe P1-B2) --------------
# Frozen builds install engine variants with the embedded uv; it must ship as
# <install root>\tools\uv.exe (the onedir and Inno install both resolve it
# there via core/uv_runner.uv_binary()).
Write-Host "[4/9] bundle tools\uv.exe" -ForegroundColor Cyan
$toolsDir = Join-Path $Root "dist\tools"
$null = New-Item -ItemType Directory -Force $toolsDir
$uvExe = Join-Path $toolsDir "uv.exe"
if (-not (Test-Path $uvExe)) {
    # Prefer the dev machine's own uv (version-identical with the toolchain,
    # offline); fall back to a pinned download (CI path). Keep the pinned
    # version in sync with .github/workflows/release.yml.
    $localUv = Get-Command uv -ErrorAction SilentlyContinue
    if ($localUv -and (Test-Path $localUv.Source)) {
        Write-Host "  copying local uv $((& $localUv.Source --version) -join ' ')..." -ForegroundColor Yellow
        Copy-Item $localUv.Source $uvExe
    } else {
        $uvVersion = "0.12.5"
        Write-Host "  downloading uv $uvVersion (pinned)..." -ForegroundColor Yellow
        $uvZip = Join-Path $toolsDir "uv.zip"
        $uvTmp = Join-Path $toolsDir "uv-tmp"
        Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $uvZip
        if (Test-Path $uvTmp) { Remove-Item $uvTmp -Recurse -Force }
        Expand-Archive -Path $uvZip -DestinationPath $uvTmp
        Move-Item (Join-Path $uvTmp "uv.exe") $uvExe -Force
        Remove-Item $uvZip -Force
        Remove-Item $uvTmp -Recurse -Force
    }
}
& $uvExe --version

# --- 5. bundled CPython (uv's declared interpreter, SelfServe P1-B2) ---------
# The frozen app ships tools\python\python.exe and uv venv always uses this
# concrete path — the runtime never discovers/downloads a Python on the user
# machine (eliminates managed-install discovery, version-alias junction trust
# / os error 448 and no-local-python failures). Source: the uv-managed
# CPython 3.12 that `uv sync` (step 1) just ensured exists; copy the REAL
# versioned dir, never the alias junction.
Write-Host "[5/9] bundle tools\python (uv-managed CPython 3.12)" -ForegroundColor Cyan
$pyRoot = Join-Path $env:APPDATA "uv\python"
$pyDir = Get-ChildItem $pyRoot -Directory -Filter "cpython-3.12.*-windows-x86_64-none" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $pyDir) { throw "uv-managed CPython 3.12 not found under $pyRoot (step 1 sync should have installed it)" }
$pyTarget = Join-Path $toolsDir "python"
if (Test-Path $pyTarget) { Remove-Item $pyTarget -Recurse -Force }
Copy-Item $pyDir.FullName $pyTarget -Recurse
if (-not (Test-Path (Join-Path $pyTarget "python.exe"))) { throw "bundled python.exe missing after copy" }
& (Join-Path $pyTarget "python.exe") --version

# --- 6. Inno Setup compiler -------------------------------------------------
Write-Host "[6/9] locate Inno Setup (ISCC)" -ForegroundColor Cyan
# Keep one canonical string path: Get-Command yields ApplicationInfo(.Source)
# while the candidate scan yields plain strings/FileInfo — mixing them broke
# .Source access (empty on FileInfo).
$isccExe = $null
$cmd = Get-Command iscc -ErrorAction SilentlyContinue
if ($cmd) { $isccExe = $cmd.Source }
if (-not $isccExe) {
    $candidates = @(
        "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$Root\.tools\innosetup\ISCC.exe"
    )
    $isccExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $isccExe) {
    Write-Host "  ISCC not found — installing Inno Setup 6..." -ForegroundColor Yellow
    # Prefer winget (Windows 10/11); fall back to a direct silent install
    # (per-user, no admin) with multi-source download: jrsoftware.org direct
    # links return truncated files on flaky networks (measured: 10KB), so
    # the GitHub release via gh-proxy goes first.
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    $installed = $false
    if ($winget) {
        winget install --id JRSoftware.InnoSetup -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) { $installed = $true }
        else { Write-Host "  winget failed — falling back to direct install" -ForegroundColor Yellow }
    }
    if (-not $installed) {
        $null = New-Item -ItemType Directory -Force "$Root\.tools"
        $setup = "$Root\.tools\innosetup-installer.exe"
        $ok = Get-RemoteFile -Urls @(
            "https://gh-proxy.com/https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe",
            "https://github.com/jrsoftware/issrc/releases/download/is-6_7_3/innosetup-6.7.3.exe",
            "https://jrsoftware.org/download.php/is.exe"
        ) -OutFile $setup -MinBytes 1MB
        if (-not $ok) { throw "Inno Setup download failed from all sources — install manually: winget install JRSoftware.InnoSetup" }
        Start-Process -FilePath $setup -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART","/CURRENTUSER","/DIR=$Root\.tools\innosetup" -Wait
    }
    # Re-scan after install (winget lands in Program Files, the direct
    # installer in .tools\).
    $candidates = @(
        "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$Root\.tools\innosetup\ISCC.exe"
    )
    $isccExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $isccExe) { throw "Inno Setup install failed. Install it manually: winget install JRSoftware.InnoSetup" }
}
Write-Host "  using $isccExe"

# The per-user Inno install omits the unofficial ChineseSimplified.isl, but
# installer.iss declares it — fetch it when missing (multi-source, size-checked).
$islPath = Join-Path (Split-Path $isccExe) "Languages\ChineseSimplified.isl"
if (-not (Test-Path $islPath)) {
    Write-Host "  fetching Languages\ChineseSimplified.isl..." -ForegroundColor Yellow
    $ok = Get-RemoteFile -Urls @(
        "https://gh-proxy.com/https://raw.githubusercontent.com/jrsoftware/issrc/is-6_7_3/Files/Languages/Unofficial/ChineseSimplified.isl",
        "https://raw.githubusercontent.com/jrsoftware/issrc/is-6_7_3/Files/Languages/Unofficial/ChineseSimplified.isl"
    ) -OutFile $islPath -MinBytes 10KB
    if (-not $ok) { throw "ChineseSimplified.isl download failed — fetch it manually or drop the chinesesimplified language from installer.iss" }
}

# --- 7. Inno installer ------------------------------------------------------
Write-Host "[7/9] compile setup.exe" -ForegroundColor Cyan
& $isccExe "/DMyAppVersion=$Version" "packaging/installer.iss"
if ($LASTEXITCODE -ne 0) { throw "iscc failed" }

# --- 8. frozen smoke ---------------------------------------------------------
if (-not $SkipSmoke -and -not $SetupOnly) {
    Write-Host "[8/9] frozen smoke (offscreen, auto-quit)" -ForegroundColor Cyan
    & "dist\LiveTranslate\LiveTranslate.exe" --smoke
    if ($LASTEXITCODE -ne 0) { throw "--smoke failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "[8/9] smoke skipped" -ForegroundColor Yellow
}

# --- 9. artifact list + sha256 -------------------------------------------------
Write-Host "[9/9] artifacts" -ForegroundColor Cyan
Get-ChildItem "dist" -File |
    Where-Object { $_.Name -notmatch "\.sha256$" } |
    ForEach-Object {
        $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
        "$($_.Name)  $([math]::Round($_.Length/1MB,1)) MB  sha256:$hash"
        Set-Content -Path "$($_.FullName).sha256" -Value "$hash" -NoNewline -Encoding ascii
    }
Write-Host "== done ==" -ForegroundColor Green

# Restore the developer's engine extras pruned by the clean-build sync.
if ($script:HadEngines -and -not (Test-Path ".venv\Lib\site-packages\funasr")) {
    Write-Host "[post] restoring engine extras (dev venv)..." -ForegroundColor Cyan
    uv sync --extra engine-funasr --extra engine-whisper
    if ($LASTEXITCODE -ne 0) { Write-Host "  restore failed — run: uv sync --extra engine-funasr --extra engine-whisper" -ForegroundColor Yellow }
}

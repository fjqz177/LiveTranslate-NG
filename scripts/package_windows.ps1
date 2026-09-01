# LiveTranslate Windows one-command packaging (SelfServe P0-A1).
#
# Runs the full release chain on a clean machine:
#   1. sync dev tools            4. compile setup.exe
#   2. main app spec             5. frozen --smoke
#   3. locate Inno Setup (ISCC)  6. artifact list + sha256
#
# Usage:  pwsh -File scripts/package_windows.ps1 [-Version 0.1.0] [-SkipSmoke] [-SetupOnly]
# The same script runs in CI (release.yml) — no divergence between local and CI.
#
# Full-install model (2026-09-01): engine dependencies (torch / faster-whisper /
# funasr) ship with the app via pyappify, so there is no embedded uv, no bundled
# CPython and no runtime/on-demand engine install — the installer bundles the app
# onedir only. No offline/preload component, no portable zip / sidecar.
# -SetupOnly builds ONLY the installer: sync → main spec → iscc (skips smoke).

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
    Write-Host "[quick] building onedir only (skips Inno / smoke)" -ForegroundColor Cyan
    & uv run pyinstaller packaging/livetranslate.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "quick onedir bundled spec failed" }
    Write-Host "== quick onedir ready: dist\LiveTranslate\ ==" -ForegroundColor Green
    return
}

# --- 1. dev tools -----------------------------------------------------------
Write-Host "[1/6] uv sync (dev group)" -ForegroundColor Cyan
uv sync --locked --group dev
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

# --- 2. PyInstaller main onedir ---------------------------------------------
Write-Host "[2/6] build main app onedir" -ForegroundColor Cyan
uv run pyinstaller packaging/livetranslate.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "main spec failed" }

# --- 3. Inno Setup compiler -------------------------------------------------
Write-Host "[3/6] locate Inno Setup (ISCC)" -ForegroundColor Cyan
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

# --- 4. Inno installer ------------------------------------------------------
Write-Host "[4/6] compile setup.exe" -ForegroundColor Cyan
& $isccExe "/DMyAppVersion=$Version" "packaging/installer.iss"
if ($LASTEXITCODE -ne 0) { throw "iscc failed" }

# --- 5. frozen smoke ---------------------------------------------------------
if (-not $SkipSmoke -and -not $SetupOnly) {
    Write-Host "[5/6] frozen smoke (offscreen, auto-quit)" -ForegroundColor Cyan
    & "dist\LiveTranslate\LiveTranslate.exe" --smoke
    if ($LASTEXITCODE -ne 0) { throw "--smoke failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "[5/6] smoke skipped" -ForegroundColor Yellow
}

# --- 6. artifact list + sha256 -------------------------------------------------
Write-Host "[6/6] artifacts" -ForegroundColor Cyan
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

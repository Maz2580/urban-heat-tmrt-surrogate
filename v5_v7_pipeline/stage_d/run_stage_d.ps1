# Stage D on the Windows box - repackage Stage C labels into V5 training tensors.
#
#   .\run_stage_d.ps1                       # default: solar channels FIXED, shadow zeros
#   .\run_stage_d.ps1 -DryRun               # validate everything, write nothing
#   .\run_stage_d.ps1 -SolarChannels zeros -OutSuffix _zeros    # the ablation corpus
#
# Producing BOTH corpora is the recommended path. Stage D is cheap and this box
# is free now; deciding later means needing the machine again. The ablation
# answers something actionable: if the solar-channel fix moves the model, that
# argues for also fixing the backwards shadow function; if it does not, that work
# drops down the list.
#
# Single process on purpose. Stage D is I/O bound - it reads a raster and writes
# an npz. Running it six ways in parallel would contend on disk and buy nothing,
# which is the lesson Stage C's worker scaling already taught us: throughput
# plateaued at ~27.5 blocks/hour regardless of worker count.
#
# ####################################################################
# THIS FILE IS DELIBERATELY PURE ASCII. DO NOT ADD em-dashes, arrows,
# degree signs or emoji - not even in comments.
#
# Windows PowerShell 5.1 reads .ps1 as CP1252 unless the file carries a UTF-8
# BOM. A UTF-8 em-dash (E2 80 94) then decodes to three CP1252 characters whose
# last is 0x94 = U+201D, a RIGHT SMART QUOTE - and PowerShell 5.1 accepts smart
# quotes as string delimiters. One em-dash inside a Write-Host string therefore
# closes the string early and the rest of the file fails to parse, reporting
# errors on unrelated lines far below.
# ####################################################################

param(
  [switch]$DryRun,
  [ValidateSet("computed","zeros")]           [string]$SolarChannels = "computed",
  [ValidateSet("zeros","solweig","computed")] [string]$ShadowChannel = "zeros",
  [string]$OutSuffix = "",
  [double]$MaxInvalidFrac = 0.5,
  [int]$EdgeBufferPx = 0
)

# Reject anything that could escape the stage_d folder or produce a surprising
# path. A typo here would scatter 38 GB somewhere unexpected on a shared machine.
if ($OutSuffix -and $OutSuffix -notmatch '^[A-Za-z0-9_.-]+$') {
  Write-Host "OutSuffix must be letters, digits, dot, dash or underscore." -ForegroundColor Red
  exit 1
}

$root   = if ($env:URBAN_HEAT_ROOT) { $env:URBAN_HEAT_ROOT } else { throw "Set URBAN_HEAT_ROOT to the working folder" }
$python = "$root\venv\Scripts\python.exe"
$pkg    = "$root\stage_d"
$stageC = "$root\stage_c\out"
$outDir = "$root\stage_d\out$OutSuffix"

# Python writes non-ASCII (degree signs, dashes) in its report. When stdout is a
# console that is fine, but Tee-Object makes it a pipe, so Python falls back to
# the cp1252 locale encoding and raises UnicodeEncodeError on the first one.
# Force UTF-8 for the child process.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# --- preflight ------------------------------------------------------------
if (-not (Test-Path $python))        { Write-Host "no venv at $python" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $stageC))        { Write-Host "no Stage C output at $stageC" -ForegroundColor Red; exit 1 }
if (-not (Test-Path "$pkg\donors"))  { Write-Host "no donor pack at $pkg\donors" -ForegroundColor Red; exit 1 }

$done = (Get-ChildItem $stageC -Recurse -Filter DONE.json).Count
Write-Host "Stage C patches complete: $done" -ForegroundColor Cyan
if ($done -lt 482) {
  Write-Host "WARNING: expected 482. Stage D will only package what exists." -ForegroundColor Yellow
  Write-Host "Ctrl+C now if Stage C is still running." -ForegroundColor Yellow
  Start-Sleep -Seconds 10
}

$donors = (Get-ChildItem "$pkg\donors" -Filter *.npz).Count
Write-Host "donors available: $donors" -ForegroundColor Cyan
if ($donors -lt 482) {
  Write-Host "WARNING: expected 482 donors. Patches without one will be skipped." -ForegroundColor Yellow
}

# ~5 MB per sample x 16 samples per patch = ~80 MB per patch
$needGB = [math]::Round($done * 0.08, 1)
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Write-Host "disk: need ~$needGB GB, free $freeGB GB" -ForegroundColor Cyan
if ($freeGB -lt $needGB * 1.3) {
  Write-Host "REFUSING: not enough headroom." -ForegroundColor Red; exit 1
}

Write-Host "config: solar=$SolarChannels  shadow=$ShadowChannel  out=$outDir" -ForegroundColor Cyan

# Mixing two corpora in one folder would be invisible afterwards - the files are
# named identically whichever way the solar channels were built.
if (Test-Path $outDir) {
  $existing = (Get-ChildItem $outDir -Filter *.npz -ErrorAction SilentlyContinue).Count
  if ($existing -gt 0 -and -not $DryRun) {
    Write-Host "REFUSING: $outDir already holds $existing npz files." -ForegroundColor Red
    Write-Host "Use a different -OutSuffix, or delete that folder first." -ForegroundColor Red
    Write-Host "The two corpora produce IDENTICAL filenames, so mixing them is" -ForegroundColor Yellow
    Write-Host "unrecoverable without regenerating." -ForegroundColor Yellow
    exit 1
  }
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# --- run ------------------------------------------------------------------
$argsList = @(
  "$pkg\stage_d.py",
  "--stage-c-dir", $stageC,
  "--out-dir",     $outDir,
  "--donor-pack",  "$pkg\donors",
  "--solar-channels", $SolarChannels,
  "--shadow-channel", $ShadowChannel,
  "--max-invalid-frac", $MaxInvalidFrac,
  "--edge-buffer-px",   $EdgeBufferPx
)
if ($DryRun) { $argsList += "--dry-run" }

$log = "$outDir\stage_d.log"
Write-Host ""
Write-Host "running; log -> $log" -ForegroundColor Green
& $python -u @argsList 2>&1 | Tee-Object -FilePath $log

# $LASTEXITCODE is the last NATIVE command's code, and Tee-Object is a cmdlet,
# so this is python's status, not the pipe's. That differs from bash, where
# `cmd | tail` reports tail's code - the trap that once made an OOM-killed run
# look like a clean exit 0. Verified by reading the value, not by assuming it.
Write-Host ""
if ($LASTEXITCODE -eq 0) {
  Write-Host "Stage D finished clean." -ForegroundColor Green
} else {
  # Non-zero also means "errors were recorded", not necessarily "nothing worked".
  Write-Host "Stage D exited $LASTEXITCODE. Read the summary above before rerunning." -ForegroundColor Yellow
}
$written = (Get-ChildItem $outDir -Filter *.npz -ErrorAction SilentlyContinue).Count
Write-Host "files written: $written" -ForegroundColor Gray

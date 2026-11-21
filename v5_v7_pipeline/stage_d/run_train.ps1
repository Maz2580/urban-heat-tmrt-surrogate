# Train Urban Heat V5 on the Windows box.
#
#   .\stage_d\run_train.ps1 -Smoke        # 2 epochs, tiny: proves the pipeline
#   .\stage_d\run_train.ps1               # the real run, corpus = out
#   .\stage_d\run_train.ps1 -Zeros        # the ablation, corpus = out_zeros
#   .\stage_d\run_train.ps1 -Resume       # continue after an interruption
#
# ####################################################################
# PURE ASCII ONLY - see setup_train_env.ps1 for why.
# ####################################################################
#
# The model is written to stage_d\models\ under a NEW name and is NOT
# registered anywhere. Live inference reads urban_heat_data/models/ and nothing
# here touches it. Registration happens only after renders are approved.

param(
  [switch]$Smoke,
  [switch]$Zeros,
  [switch]$Resume,
  [int]$Epochs = 100,
  [int]$BatchSize = 12,
  [int]$Workers = 4
)

$root   = if ($env:URBAN_HEAT_ROOT) { $env:URBAN_HEAT_ROOT } else { throw "Set URBAN_HEAT_ROOT to the working folder" }
$py     = "$root\venv_train\Scripts\python.exe"
$pkg    = "$root\stage_d"
$corpus = if ($Zeros) { "$pkg\out_zeros" } else { "$pkg\out" }
$tag    = if ($Zeros) { "v5_zeros" } else { "v5" }
$outDir = "$pkg\models"

if (-not (Test-Path $py)) {
  Write-Host "no training venv. Run .\stage_d\setup_train_env.ps1 first." -ForegroundColor Red
  exit 1
}
if (-not (Test-Path "$corpus\manifest.parquet")) {
  Write-Host "no manifest at $corpus\manifest.parquet" -ForegroundColor Red
  Write-Host "download it and copy it into BOTH out and out_zeros." -ForegroundColor Red
  exit 1
}

$n = (Get-ChildItem $corpus -Filter *.npz).Count
Write-Host "corpus : $corpus  ($n npz files)" -ForegroundColor Cyan
Write-Host "tag    : $tag" -ForegroundColor Cyan
if ($n -lt 7712 -and -not $Smoke) {
  Write-Host "WARNING: expected 7712 npz files, found $n." -ForegroundColor Yellow
}

# Same encoding guard as Stage D: Tee-Object makes stdout a pipe, so Python
# falls back to cp1252 and dies on any non-ASCII character.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$argsList = @(
  "$pkg\train_v5.py",
  "--corpus", $corpus,
  "--out-dir", $outDir,
  "--tag", $tag,
  "--epochs", $Epochs,
  "--batch-size", $BatchSize,
  "--num-workers", $Workers
)
if ($Smoke)  { $argsList += "--smoke" }
if ($Resume) { $argsList += "--resume" }

$log = "$outDir\train_$tag.log"
Write-Host "log    : $log" -ForegroundColor Cyan
Write-Host ""

# Append, not overwrite: a resumed run should not erase the earlier history.
& $py -u @argsList 2>&1 | Tee-Object -FilePath $log -Append

Write-Host ""
if ($LASTEXITCODE -eq 0) {
  Write-Host "training finished." -ForegroundColor Green
  Write-Host "checkpoints in $outDir" -ForegroundColor Gray
  Write-Host "NOT registered for live inference - that needs render approval." -ForegroundColor Yellow
} else {
  Write-Host "training exited $LASTEXITCODE. Re-run with -Resume to continue" -ForegroundColor Yellow
  Write-Host "from the last completed epoch." -ForegroundColor Yellow
}

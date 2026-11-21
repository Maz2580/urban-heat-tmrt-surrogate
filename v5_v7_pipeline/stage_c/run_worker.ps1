# One Stage C worker.  Usage:  .\run_worker.ps1 0 6
#
# Exists because the full python command line is ~400 characters and PowerShell
# truncates long pasted commands mid-line (observed: "rclone.exe --config" ->
# "flag needs an argument"). Everything long lives here instead.
#
# Each worker takes shard i of n, so no two workers touch the same patch, and
# each gets its own scratch dir. Resume is per patch via DONE.json, so a worker
# can be killed and restarted with the same arguments at any time.

param(
  [Parameter(Mandatory=$true)][int]$i,
  [int]$n = 6,
  [int]$threads = 4
)

$root = if ($env:URBAN_HEAT_ROOT) { $env:URBAN_HEAT_ROOT } else { throw "Set URBAN_HEAT_ROOT to the working folder" }
$py   = "$root\venv\Scripts\python.exe"

if (-not (Test-Path $py)) { Write-Error "python not found at $py"; exit 1 }

Write-Host "worker $i/$n  threads=$threads" -ForegroundColor Cyan

& $py -u "$root\stage_c\stage_c_generate.py" `
  --root       "$root\stage_c" `
  --out        "$root\stage_c\out" `
  --scratch    "$root\scratch\w$i" `
  --patch-list "$root\stage_c\patch_list_B_default_lt70.json" `
  --shard      "$i/$n" `
  --threads    $threads `
  --allow-cpu

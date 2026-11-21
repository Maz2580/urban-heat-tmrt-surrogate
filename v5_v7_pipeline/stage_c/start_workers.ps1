# Launch N Stage C workers, each in its own window.  Usage:  .\start_workers.ps1 2
#
# Separate visible windows on purpose: this is a SHARED machine, so anyone should
# be able to see what is running and close it. Killing a window stops that worker
# only; resume is per patch, so at most one patch of work is lost and re-running
# picks it straight back up.
#
# ⚠️ START SMALL. On our other server, three concurrent workers exhausted RAM and
# triggered a GLOBAL OOM that killed unrelated live processes — the kernel picks
# victims by score, not by culprit. Run 2 first, watch memory, then scale.

param(
  [int]$n = 2,
  [int]$threads = 4
)

$root = if ($env:URBAN_HEAT_ROOT) { $env:URBAN_HEAT_ROOT } else { throw "Set URBAN_HEAT_ROOT to the working folder" }

$freeGB = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 1)
Write-Host "free RAM before launch: $freeGB GB" -ForegroundColor Yellow
if ($freeGB -lt 8) {
  Write-Host "REFUSING: under 8 GB free. Close something or run fewer workers." -ForegroundColor Red
  exit 1
}

for ($i = 0; $i -lt $n; $i++) {
  Start-Process powershell -ArgumentList @(
    "-NoExit", "-File", "$root\stage_c\run_worker.ps1", "$i", "$n", "$threads"
  )
  Start-Sleep -Seconds 20   # stagger: keeps the wall/aspect process pools from
                            # all firing at once in the first ~22 s of each run
}

Write-Host "launched $n workers (shards 0..$($n-1) of $n)" -ForegroundColor Green
Write-Host "progress:  dir $root\stage_c\out | measure" -ForegroundColor Gray

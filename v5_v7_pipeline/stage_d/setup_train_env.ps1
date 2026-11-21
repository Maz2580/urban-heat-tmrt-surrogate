# Build a SEPARATE venv for V5 training, with CUDA PyTorch for the RTX 5090.
#
#   .\stage_d\setup_train_env.ps1
#
# ####################################################################
# PURE ASCII ONLY. Windows PowerShell 5.1 reads .ps1 as CP1252 unless the file
# has a UTF-8 BOM, and a UTF-8 em-dash decodes to a character PS treats as a
# string delimiter. One em-dash in a string breaks the whole file.
# ####################################################################
#
# WHY A SECOND VENV
# The Stage C/D venv has CPU-only torch ON PURPOSE: SOLWEIG gained nothing from
# a GPU (A100 100s ~ T4 103s ~ CPU 60s per date) and CPU-only sidestepped
# CUDA 13.3 on Blackwell. That environment produced everything we verified
# today. Replacing its torch to train would put a validated environment at risk
# for no reason, and disk is free. So: venv_train, separate, disposable.
#
# Training is the opposite case from SOLWEIG: dense matrix work with no Python
# in the inner loop, so the GPU is the whole point.

param(
  [string]$CudaTag = "cu130"     # cu130 carries torch 2.13; cu128's newest is 2.11
)

$root = if ($env:URBAN_HEAT_ROOT) { $env:URBAN_HEAT_ROOT } else { throw "Set URBAN_HEAT_ROOT to the working folder" }
$venv = "$root\venv_train"
$py   = "$venv\Scripts\python.exe"

if (-not (Test-Path $venv)) {
  Write-Host "creating $venv" -ForegroundColor Cyan
  python -m venv $venv
}

& $py -m pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# ORDER MATTERS. Installing CUDA torch first and the rest afterwards SILENTLY
# REPLACES it: the second pip run does not see the CUDA index, and
# segmentation-models-pytorch requires a newer torch than that index carries,
# so pip resolves torch from PyPI - which is the CPU build - and uninstalls the
# CUDA one. The log says "Successfully installed" either way; the only symptom
# is torch.cuda.is_available() == False at the end.
#
# So: everything else FIRST, CUDA torch LAST, and nothing after it.
# ------------------------------------------------------------------
Write-Host "step 1/2: dependencies (may pull a CPU torch; that is fine)" -ForegroundColor Cyan
& $py -m pip install segmentation-models-pytorch pandas pyarrow numpy

Write-Host ""
Write-Host "step 2/2: replacing torch with the $CudaTag build (~3 GB)" -ForegroundColor Cyan
& $py -m pip uninstall -y torch torchvision
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"

# Prove the replacement actually stuck before running the GPU test, so a
# resolver surprise is caught here rather than looking like a driver problem.
$tv = & $py -c "import torch; print(torch.__version__)"
Write-Host "torch now: $tv" -ForegroundColor Cyan
if ($tv -notmatch "\+cu") {
  Write-Host "FAILED: torch is still a CPU build ($tv)." -ForegroundColor Red
  Write-Host "Something re-resolved it from PyPI. Do not install anything after" -ForegroundColor Red
  Write-Host "this step; re-run the script instead." -ForegroundColor Red
  exit 1
}

# ------------------------------------------------------------------
# Verify with a REAL kernel launch, not just torch.cuda.is_available().
#
# is_available() returns True whenever a CUDA driver is present, even if the
# build has no kernels for this GPU's architecture. The failure only appears
# when a kernel actually launches, as:
#     no kernel image is available for execution on the device
# which reads like a broken install rather than a version mismatch. So run a
# matmul and force a sync.
# ------------------------------------------------------------------
Write-Host ""
Write-Host "verifying (this is the check that can actually fail)" -ForegroundColor Cyan
& $py -c @"
import torch, sys
print('torch           :', torch.__version__)
print('cuda available  :', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('RESULT: NO CUDA - training would take days. Stop here.'); sys.exit(1)
print('device          :', torch.cuda.get_device_name())
cap = torch.cuda.get_device_capability()
print('compute cap     :', f'sm_{cap[0]}{cap[1]}')
try:
    a = torch.randn(4096, 4096, device='cuda')
    b = a @ a
    torch.cuda.synchronize()
    print('kernel launch   : OK  (mean %.4f)' % float(b.mean()))
except Exception as e:
    print('kernel launch   : FAILED')
    print(' ', type(e).__name__, str(e)[:200])
    print('RESULT: this torch build has no kernels for this GPU.')
    print('        Re-run:  .\\stage_d\\setup_train_env.ps1 -CudaTag cu128')
    sys.exit(2)
try:
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                 in_channels=23, classes=3, activation=None).cuda()
    x = torch.randn(2, 23, 512, 512, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = m(x)
    torch.cuda.synchronize()
    print('model fwd bf16  : OK  out', tuple(y.shape))
except Exception as e:
    print('model fwd bf16  : FAILED'); print(' ', type(e).__name__, str(e)[:200]); sys.exit(3)
print()
print('RESULT: environment is ready for training.')
"@

if ($LASTEXITCODE -ne 0) {
  Write-Host "setup FAILED (exit $LASTEXITCODE) - read the message above." -ForegroundColor Red
  exit $LASTEXITCODE
}
Write-Host ""
Write-Host "ready. Next:  .\stage_d\run_train.ps1" -ForegroundColor Green

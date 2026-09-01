param([string]$Python = "")

$packageRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($Python)) {
    $localPython = Join-Path $packageRoot ".venv\Scripts\python.exe"
    $sourcePython = "D:\navwareset_scene01_clean\.venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { $sourcePython }
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

Push-Location $packageRoot
try {
    & $Python -m py_compile run_online.py src\online_v2\runtime.py src\online_v2\pipeline.py src\online_v2\identity.py src\online_v2\support.py src\online_v2\registration.py src\online_v2\recovery.py scripts\audit_persistent_identity.py scripts\audit_static_registration.py scripts\audit_soft_ownership.py scripts\audit_suppressed_point_recovery.py scripts\audit_occupancy_suppression.py scripts\audit_fn_root_causes.py scripts\audit_a2_correspondence.py scripts\audit_empirical_pixel_offset.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m pytest -q tests\test_online.py
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

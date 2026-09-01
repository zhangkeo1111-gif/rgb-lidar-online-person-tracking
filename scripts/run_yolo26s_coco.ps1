param([Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs)
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = 'D:\navwareset_scene01_clean\.venv\Scripts\python.exe'
& $python (Join-Path $repoRoot 'run_online.py') --detector yolo26s-coco @ExtraArgs
exit $LASTEXITCODE

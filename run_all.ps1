$ErrorActionPreference = "Stop"

$ProjectPython = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw "Missing .venv. Run setup_windows.ps1 first."
}

New-Item -ItemType Directory -Force -Path "out\videos", "out\logs" | Out-Null

& $ProjectPython run_tracker.py `
    "clips\9055.mp4" `
    "out\videos\9055_tracked.mp4" `
    --targets "targets.json" `
    --config "configs\default.json" `
    --log "out\logs\9055_tracked.jsonl"

& $ProjectPython run_tracker.py `
    "clips\iBbVocyi7jM8QqB_VID_20260509144917376_rallies.mp4" `
    "out\videos\indoor_tracked.mp4" `
    --targets "targets.json" `
    --config "configs\default.json" `
    --log "out\logs\indoor_tracked.jsonl"

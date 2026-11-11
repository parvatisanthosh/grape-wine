# Overnight run of ORBIT evaluation v2 (notes/evaluation_v2_plan.md).
#
# Runs the tuning sweep, then passes A, B and C on the test trace, then the analysis.
# Every step resumes where it stopped, so running this again after an interruption
# continues the work instead of repeating it.
#
# Before starting: plug in the laptop, set sleep to "Never" when plugged in, close every
# other app (including VS Code, Docker, ChatGPT and the Claude app), and pause OneDrive.
#
#   cd C:\Users\parva\OneDrive\Desktop\open-vino\orbit-llm
#   powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1
#
# Progress and errors go to results\evaluation_v2\overnight.log.

param(
    [double]$SweepFreeGB = 2.0,  # the sweep runs one configuration at a time (3B model: ~2.6 GB)
    [double]$PassFreeGB = 2.0,   # passes run one system at a time (--block-size): ~3.4 GB at peak
    [int]$WaitMinutes = 30       # how long to wait for memory before giving up
)

# "Continue": Python's warnings go to stderr, and stricter settings would treat them as
# fatal. Failures are caught from the exit code instead.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$results = Join-Path $PSScriptRoot "results\evaluation_v2"
New-Item -ItemType Directory -Force $results | Out-Null
$log = Join-Path $results "overnight.log"

function Say($text) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $text
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function FreeGB {
    # Available memory: free plus cache that Windows can hand back at once.
    [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue / 1024, 1)
}

function WaitForMemory($needed) {
    for ($minute = 0; $minute -le $WaitMinutes; $minute++) {
        $free = FreeGB
        if ($free -ge $needed) { return $free }
        if ($minute -eq 0) { Say "Waiting for $needed GB of free memory (now $free GB). Close other apps." }
        Start-Sleep -Seconds 60
    }
    return FreeGB
}

function Step($name, [double]$needed, [string[]]$arguments) {
    $free = WaitForMemory $needed
    if ($free -lt $needed) {
        Say "STOPPED before $name`: only $free GB free after waiting $WaitMinutes minutes, $needed GB needed. Close apps and run this script again."
        exit 1
    }
    Say "START $name (free RAM $free GB)"
    # The Python guard refuses to start on battery, so a pulled plug stops the run cleanly.
    & python @arguments 2>&1 | ForEach-Object {
        $text = "$_"
        Add-Content -Path $log -Value $text -Encoding utf8
        if ($text -match "^\[|Sweeping|Best fixed|done|STOP|rror|battery") { Write-Host $text }
    }
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Say "FAILED: $name (exit code $code). See the log above."
        exit $code
    }
    Say "DONE $name"
}

Say "Overnight run started"
Step "tuning sweep" $SweepFreeGB @("evaluate_v2.py", "sweep", "--trace", "traces/tuning.jsonl")
Step "pass A" $PassFreeGB @("evaluate_v2.py", "compare", "--trace", "traces/test.jsonl", "--pass", "A", "--block-size", "25")
Step "pass B" $PassFreeGB @("evaluate_v2.py", "compare", "--trace", "traces/test.jsonl", "--pass", "B", "--block-size", "25")
Step "pass C" $PassFreeGB @("evaluate_v2.py", "compare", "--trace", "traces/test.jsonl", "--pass", "C", "--block-size", "25")

$analysis = Join-Path $results "analysis.txt"
& python evaluate_v2.py analyze (Join-Path $results "pass_A.jsonl") (Join-Path $results "pass_B.jsonl") (Join-Path $results "pass_C.jsonl") 2>&1 |
    ForEach-Object { "$_" } | Out-File -FilePath $analysis -Encoding utf8
Say "ALL DONE. Analysis: $analysis"

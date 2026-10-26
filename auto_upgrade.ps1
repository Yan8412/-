<#
.SYNOPSIS
  Weekly La Liga retune: walk-forward search, then promote a challenger only if it wins on recent out-of-sample matches.

.DESCRIPTION
  Does not replace latest_next.json or latest.json. The daily auto_update.ps1
  picks up a new champion the next time it trains or predicts, because those
  commands read data\models\registry.json unless --ignore-champion is set.

  Exit codes: 0 = the search finished (promoted or not), 1 = the search failed.
  One summary line per run is appended to logs\auto_upgrade.log.

  Task Scheduler (weekly, for example Sunday 06:00 local):
    Program:   powershell.exe
    Arguments: -NoProfile -ExecutionPolicy Bypass -File "D:\laliga-predictor\auto_upgrade.ps1"
    Start in:  D:\laliga-predictor

  Cross-platform entry, from the repo root:
    python -m laliga upgrade --trials 12 --holdout 80 --min-promote 40
#>
[CmdletBinding()]
param(
    [int]$Trials = 12,
    [int]$Holdout = 80,
    [int]$MinPromote = 40,
    [ValidateSet('bootstrap', 'strict')]
    [string]$Rule = 'bootstrap',
    [int]$MinTrain = 320,
    [int]$RefitEvery = 14
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'
$TmpDir = Join-Path $Root 'tmp'
New-Item -ItemType Directory -Force -Path $LogDir, $TmpDir | Out-Null

$SummaryLog = Join-Path $LogDir 'auto_upgrade.log'
$RunLog = Join-Path $LogDir ('upgrade-' + (Get-Date -Format 'yyyyMMdd') + '.log')

$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:LALIGA_DATA_DIR = Join-Path $Root 'data'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Started = Get-Date

function Write-Summary([string]$Status, [string]$Detail) {
    $utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $seconds = [int]((Get-Date) - $Started).TotalSeconds
    $line = "$utc status=$Status duration=${seconds}s"
    if ($Detail) { $line += " detail=`"$Detail`"" }
    Add-Content -LiteralPath $SummaryLog -Value $line -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Summary 'FAIL' "missing $Python"
    exit 1
}

$LockPath = Join-Path $LogDir 'auto_upgrade.lock'
try {
    $Lock = [System.IO.File]::Open($LockPath, 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    Write-Summary 'SKIP' 'another auto_upgrade run is still going'
    exit 1
}

try {
    $arguments = @(
        '-m', 'laliga', 'upgrade',
        '--trials', "$Trials",
        '--holdout', "$Holdout",
        '--min-promote', "$MinPromote",
        '--rule', $Rule,
        '--min-train', "$MinTrain",
        '--refit-every', "$RefitEvery"
    )
    Add-Content -LiteralPath $RunLog -Value ("[{0}] >>> python {1}" -f (Get-Date -Format s), ($arguments -join ' ')) -Encoding UTF8
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Python @arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($output) { Add-Content -LiteralPath $RunLog -Value ($output -join [Environment]::NewLine) -Encoding UTF8 }
    Add-Content -LiteralPath $RunLog -Value ("[{0}] <<< exit {1}" -f (Get-Date -Format s), $code) -Encoding UTF8
    if ($code -ne 0) {
        Write-Summary 'FAIL' 'upgrade failed; champion unchanged unless the registry was already written'
        exit 1
    }
    $promoted = 'unknown'
    if ($output -match '已升为冠军') { $promoted = 'yes' }
    elseif ($output -match '未升冠军') { $promoted = 'no' }
    Write-Summary 'OK' "promoted=$promoted"
    exit 0
} finally {
    $Lock.Dispose()
}

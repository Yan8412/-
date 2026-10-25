<#
.SYNOPSIS
  Daily La Liga update: fetch -> freshness guard -> train -> predict (archived) -> history -> backtest.

.DESCRIPTION
  Every path is relative to the folder this script is in (for example
  D:\laliga-predictor). Python is .venv\Scripts\python.exe in that folder.
  TEMP/TMP point at .\tmp so nothing is written to C:.

  Exit codes: 0 = predictions updated, 2 = data STALE (predictions left as they
  were), 1 = a required step failed (predictions left as they were).

  One summary line per run is appended to logs\auto_update.log. Full output of
  each run goes to logs\run-YYYYMMDD.log (the last 30 are kept).

  Task Scheduler action:
    Program:   powershell.exe
    Arguments: -NoProfile -ExecutionPolicy Bypass -File "D:\laliga-predictor\auto_update.ps1"
    Start in:  D:\laliga-predictor

.PARAMETER Availability
  Also fetch lineups/sidelined and predict with --availability. Off by default:
  turn it on only after `compare-availability` reports recommend_default = true.

.PARAMETER SkipBacktest
  Skip the walk-forward backtest at the end.

.PARAMETER LatestJsonFormat
  auto (default): keep whatever shape data\predictions\latest.json already has.
  dashboard: {created_at, label, note, items} with score matrices.
  list: a copy of latest_next.json.
#>
[CmdletBinding()]
param(
    [switch]$Availability,
    [switch]$SkipBacktest,
    [ValidateSet('auto', 'dashboard', 'list')]
    [string]$LatestJsonFormat = 'auto',
    [int]$Seasons = 3
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'
$DataDir = Join-Path $Root 'data'
$PredDir = Join-Path $DataDir 'predictions'
$TmpDir = Join-Path $Root 'tmp'
New-Item -ItemType Directory -Force -Path $LogDir, $PredDir, $TmpDir | Out-Null

$SummaryLog = Join-Path $LogDir 'auto_update.log'
$RunLog = Join-Path $LogDir ('run-' + (Get-Date -Format 'yyyyMMdd') + '.log')
$LatestNext = Join-Path $PredDir 'latest_next.json'
$LatestNextCsv = Join-Path $PredDir 'latest_next.csv'
$LatestJson = Join-Path $PredDir 'latest.json'

$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:LALIGA_DATA_DIR = $DataDir
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Started = Get-Date
$Steps = [ordered]@{}

function Write-Run([string]$Text) {
    Add-Content -LiteralPath $RunLog -Value $Text -Encoding UTF8
}

function Write-Summary([string]$Status, [string]$Detail) {
    $utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    $seconds = [int]((Get-Date) - $Started).TotalSeconds
    $parts = @("$utc status=$Status")
    foreach ($key in $Steps.Keys) { $parts += "$key=$($Steps[$key])" }
    $parts += "duration=${seconds}s"
    if ($Detail) { $parts += "detail=`"$Detail`"" }
    Add-Content -LiteralPath $SummaryLog -Value ($parts -join ' ') -Encoding UTF8
}

function Invoke-Laliga([string]$Name, [string[]]$Arguments) {
    Write-Run ("[{0}] >>> {1}: python {2}" -f (Get-Date -Format s), $Name, ($Arguments -join ' '))
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Python @Arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($output) { Write-Run ($output -join [Environment]::NewLine) }
    Write-Run ("[{0}] <<< {1}: exit {2}" -f (Get-Date -Format s), $Name, $code)
    $Steps[$Name] = $code
    return @{ Code = $code; Output = ($output -join "`n") }
}

if (-not (Test-Path -LiteralPath $Python)) {
    $Steps['python'] = 'missing'
    Write-Summary 'FAIL' "missing $Python"
    exit 1
}

$LockPath = Join-Path $LogDir 'auto_update.lock'
try {
    $Lock = [System.IO.File]::Open($LockPath, 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    $Steps['lock'] = 'busy'
    Write-Summary 'SKIP' 'another auto_update run is still going'
    exit 1
}

try {
    Get-ChildItem -LiteralPath $LogDir -Filter 'run-*.log' |
        Sort-Object Name -Descending | Select-Object -Skip 30 | Remove-Item -Force -ErrorAction SilentlyContinue

    $fetch = Invoke-Laliga 'fetch' @('-m', 'laliga', 'fetch', '--seasons', "$Seasons")
    if ($fetch.Code -ne 0) {
        Write-Summary 'FAIL' 'fetch failed; predictions unchanged'
        exit 1
    }

    $fresh = Invoke-Laliga 'fresh' @('-m', 'laliga', 'check-freshness')
    if ($fresh.Code -eq 2) {
        $retry = Invoke-Laliga 'refetch' @('-m', 'laliga', 'fetch', '--seasons', "$Seasons", '--refresh')
        $fresh = Invoke-Laliga 'fresh2' @('-m', 'laliga', 'check-freshness')
    }
    if ($fresh.Code -eq 2) {
        $reason = ($fresh.Output -split "`n")[0]
        Write-Summary 'STALE' "$reason; predictions unchanged"
        exit 2
    }
    if ($fresh.Code -ne 0) {
        Write-Summary 'FAIL' 'freshness check failed; predictions unchanged'
        exit 1
    }

    $useAvailability = $false
    if ($Availability) {
        $avail = Invoke-Laliga 'availability' @('-m', 'laliga', 'fetch-availability')
        $coefficients = Join-Path $DataDir 'models\availability.json'
        $useAvailability = ($avail.Code -eq 0) -and (Test-Path -LiteralPath $coefficients)
        if (-not $useAvailability) { $Steps['availability_used'] = 'no' }
    }

    $train = Invoke-Laliga 'train' @('-m', 'laliga', 'train')
    if ($train.Code -ne 0) {
        Write-Summary 'FAIL' 'train failed; predictions unchanged'
        exit 1
    }

    $format = $LatestJsonFormat
    if ($format -eq 'auto') {
        $format = 'dashboard'
        if (Test-Path -LiteralPath $LatestJson) {
            $first = (Get-Content -LiteralPath $LatestJson -Raw -Encoding UTF8).TrimStart([char]0xFEFF).TrimStart()
            if ($first.StartsWith('[')) { $format = 'list' }
        }
    }

    $predictArgs = @('-m', 'laliga', 'predict', '--next', '--json', $LatestNext, '--csv', $LatestNextCsv)
    if ($format -eq 'dashboard') { $predictArgs += '--publish' }
    if ($useAvailability) { $predictArgs += '--availability' }
    $predict = Invoke-Laliga 'predict' $predictArgs
    if ($predict.Code -ne 0) {
        Write-Summary 'FAIL' 'predict failed; predictions unchanged'
        exit 1
    }
    if ($format -eq 'list') {
        Copy-Item -LiteralPath $LatestNext -Destination $LatestJson -Force
    }
    $count = 0
    try { $count = @(Get-Content -LiteralPath $LatestNext -Raw -Encoding UTF8 | ConvertFrom-Json).Count } catch { }
    $Steps['fixtures'] = $count
    if ($predict.Output -match '\u9884\u6d4b\u5b58\u6863\uff1a\u5199\u5165 (\d+) \u573a') { $Steps['archived'] = $Matches[1] }

    [void](Invoke-Laliga 'history' @('-m', 'laliga', 'history', '--limit', '0', '--json', (Join-Path $PredDir 'history.json')))

    if (-not $SkipBacktest) {
        [void](Invoke-Laliga 'backtest' @('-m', 'laliga', 'backtest', '--output', (Join-Path $PredDir 'backtest.json')))
    }

    $detail = "latest.json=$format"
    if ($useAvailability) { $detail += ' availability=on' }
    Write-Summary 'OK' $detail
    exit 0
} finally {
    $Lock.Dispose()
}

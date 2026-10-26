# Updating the Windows production copy (D:\laliga-predictor)

The laptop runs a plain folder copy (no git) at `D:\laliga-predictor`, with its
virtual environment at `D:\laliga-predictor\.venv` and the daily job
`auto_update.ps1`. These steps replace the code with this branch, keep `data\`,
`logs\`, `.venv\` and `.env` untouched, and keep every path on `D:`.

Run everything in **one Windows PowerShell window** (5.1 is fine), so the
variables and the D: temp folder from step 0 apply to every step. An
administrator window is needed only for the Task Scheduler lines (steps 1 and 9);
you can open an admin window for the whole procedure.

## What changes for production

| Before (laptop copy) | After (this branch) |
| --- | --- |
| Pre-PR #3 code | Current `main` plus this branch (Shin / power de-vig, RPS, football-data import, archive, availability flag) |
| `check_freshness.py` only on the laptop | `python -m laliga check-freshness` and a root `check_freshness.py` in the repo, same exit codes (0 OK, 2 STALE) |
| `fetch` replays the raw cache for the current season unless `--refresh` | `fetch` re-requests the league list and the current season every run; finished seasons stay cached |
| Each run overwrites predictions | Every `predict` also appends to `data\archive\predictions.jsonl` (append-only, nothing after kickoff) |
| — | Dashboard tab 历史记录 (History): predicted vs actual with running hit rates and baselines |

`data\predictions\latest_next.json` keeps exactly the same fields (`starting_at`
in UTC, `home_team`, `away_team`, `ft{home,draw,away}`, `ht{...}`,
`top_scores[{home_goals,away_goals,probability}]`, `expected_goals_ft/ht`, plus
the ids and season fields it already had). It is now written atomically
(temporary file, then rename), so the notifier never reads half a file.

`latest.json` keeps whatever shape it already has: the new `auto_update.ps1`
looks at the first character of the existing file and keeps a list (`[`) as a
list, or the dashboard object (`{`) as an object. Force it with
`-LatestJsonFormat list` or `-LatestJsonFormat dashboard`.

## Step by step

### 0. Set the variables (D: only)

```powershell
$App    = 'D:\laliga-predictor'
$Work   = 'D:\laliga-deploy'
$Stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$Branch = 'cursor/prediction-archive-availability-4db1'
New-Item -ItemType Directory -Force -Path $Work, "$Work\tmp", "$Work\pip-cache" | Out-Null
$env:TEMP = "$Work\tmp"
$env:TMP = "$Work\tmp"
$env:PIP_CACHE_DIR = "$Work\pip-cache"
```

### 1. Pause the scheduled task (admin shell)

```powershell
$Task = Get-ScheduledTask | Where-Object { ($_.Actions | ForEach-Object { $_.Arguments }) -match 'auto_update\.ps1' }
$Task | Format-Table TaskName, State
$Task | Disable-ScheduledTask
```

If nothing is listed, the task calls the script some other way; disable it by
name in Task Scheduler instead.

### 2. Back up the whole folder (code, data, logs)

```powershell
robocopy $App "$Work\backup-$Stamp" /E /XD .venv __pycache__ /NFL /NDL /NP
"robocopy exit $LASTEXITCODE (0-7 is success)"
```

### 3. Download this branch

```powershell
Invoke-WebRequest -UseBasicParsing "https://github.com/Yan8412/-/archive/refs/heads/$Branch.zip" -OutFile "$Work\src-$Stamp.zip"
Expand-Archive -LiteralPath "$Work\src-$Stamp.zip" -DestinationPath "$Work\src-$Stamp" -Force
$Src = (Get-ChildItem -LiteralPath "$Work\src-$Stamp" -Directory | Select-Object -First 1).FullName
$Src
```

### 4. Compare your laptop-only scripts before they are replaced

The repo now ships `auto_update.ps1` and `check_freshness.py` at the root, so the
copy in step 5 overwrites the laptop versions (the backup from step 2 keeps them).
If the old `auto_update.ps1` did anything extra, such as starting the notifier,
add it to the new one before re-enabling the task.

```powershell
fc.exe "$App\auto_update.ps1" "$Src\auto_update.ps1"
fc.exe "$App\check_freshness.py" "$Src\check_freshness.py"
```

### 5. Copy the code over, never touching data, logs, venv or .env

```powershell
robocopy $Src $App /E /XD data logs .venv tmp .git /XF .env /NFL /NDL /NP
"robocopy exit $LASTEXITCODE (0-7 is success)"
```

No file was deleted or renamed between the old laptop code and this branch, so
copying over the old tree leaves no stale modules behind.

### 6. Update dependencies inside the existing venv

```powershell
Set-Location -LiteralPath $App
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`xgboost==3.2.0` is new since the laptop copy. It has a `win_amd64` wheel for
Python 3.11–3.14. Do not run `playwright install`: the browser tests are not
needed on the laptop, and Playwright would download browsers into the user
profile on C:.

### 7. Verify

```powershell
Set-Location -LiteralPath $App
.\.venv\Scripts\python.exe -m laliga doctor
.\.venv\Scripts\python.exe -m pytest -q tests\test_archive.py tests\test_freshness.py tests\test_cli.py tests\test_availability.py
```

`doctor` prints the match table size, the saved model, whether
`latest_next.json` still has the notifier fields, the archive line counts, and
the freshness result. It exits 1 only when something is broken.

Then run the daily job once by hand and read its summary line:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$App\auto_update.ps1"
"exit $LASTEXITCODE (0 = updated, 2 = STALE, 1 = failed)"
Get-Content -LiteralPath "$App\logs\auto_update.log" -Tail 3
.\.venv\Scripts\python.exe -m laliga doctor
```

A good run ends with a line like
`2026-10-15T05:00:12Z status=OK fetch=0 fresh=0 train=0 predict=0 fixtures=10 archived=10 history=0 backtest=0 duration=95s detail="latest.json=dashboard"`.
The full output of each run is in `logs\run-YYYYMMDD.log`.

### 8. Optional: reconstruct the 2026/27 matches already played

The live archive starts empty on the first run. To fill the History tab for
earlier 2026/27 matches with a clearly labelled walk-forward reconstruction (not
live predictions; it is kept in a separate file and separate hit rates):

```powershell
.\.venv\Scripts\python.exe -m laliga backfill-archive --season-id 27965
```

The same thing is the 回测重建本赛季 button on the 历史记录 page.

### 9. Re-enable the task (admin shell)

```powershell
$Task | Enable-ScheduledTask
```

The action should be `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\laliga-predictor\auto_update.ps1"`
with "Start in" set to `D:\laliga-predictor`.

## Rolling back

```powershell
robocopy "$Work\backup-$Stamp" $App /E /XD .venv /NFL /NDL /NP
```

This restores the old code and the data as it was at step 2. The archive file
written in the meantime would be replaced by the backup copy; copy
`data\archive\predictions.jsonl` aside first if you want to keep it.

## Checks that need the live SportMonks token (not possible in CI)

1. Availability data on the plan. The docs list `lineups` and `sidelined` as
   included in every base plan, Starter included. Confirm with the real token:

   ```powershell
   .\.venv\Scripts\python.exe -m laliga fetch-availability
   ```

   Exit 0 with non-zero "有伤停记录" / "两队首发都齐" counts means the data is
   there. Exit 1 with the "HTTP 403" message means the token cannot read it.

2. The evaluation that decides whether availability should be the default:

   ```powershell
   .\.venv\Scripts\python.exe -m laliga compare-availability --min-train 320
   ```

   It prints log loss, RPS, Brier, accuracy and top-3 for the frequency
   baseline, Dixon–Coles, and Dixon–Coles with the availability adjustment, with
   sample sizes and paired bootstrap intervals, and writes
   `data\predictions\availability_comparison.json`. It also writes
   `data\models\availability.json` with `recommend_default`. Only when that is
   `true`, add `-Availability` to the scheduled task's arguments. Without the
   switch the daily job stays plain Dixon–Coles.

3. That the current-season refetch picks up results: after a matchday,
   `check-freshness` should report the latest result within a day or two.

## Weekly upgrade (optional)

`auto_upgrade.ps1` does not write `latest_next.json`. It runs
`python -m laliga upgrade` and only replaces `data\models\registry.json` when
the challenger beats the champion on matches the search never saw. Schedule it
weekly, separate from the daily task:

```powershell
$Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -File "D:\laliga-predictor\auto_upgrade.ps1"' -WorkingDirectory 'D:\laliga-predictor'
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 6:00am
Register-ScheduledTask -TaskName 'LaLiga weekly upgrade' -Action $Action -Trigger $Trigger
```

The next daily `auto_update.ps1` reads the champion. Roll back with:

```powershell
.\.venv\Scripts\python.exe -m laliga upgrade --rollback
```

Without a live token, `python -m laliga probe` cannot be checked here. After
the token is set, run it once and read which of markets 1, 6, 12, 26, 28, 29,
30, 31, 57 and 80, which bookmakers, and which xG fields actually come back.

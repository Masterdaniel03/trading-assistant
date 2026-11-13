# Registers the daily run with Windows Task Scheduler.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1            # install / update
#   powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove    # remove it
#
# Runs `python main.py daily` Monday to Friday at 14:30 local time (17:30 New York
# when this PC is on Pacific time), after the day's closing prices are final.
# The command itself skips holidays. If the PC is off or asleep at 14:30, the run
# starts as soon as it is back on; orders sent late still wait for the next open,
# and the bot's own checks refuse to trade on out-of-date prices.
param([switch]$Remove, [string]$At = "14:30")

$name = "TradingAssistantDaily"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output "Removed scheduled task '$name' (if it existed)."
    return
}

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python not found at $python. Create the .venv first." }

$action = New-ScheduledTaskAction -Execute $python -Argument "main.py daily" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -Description "Trading assistant: fetch prices, research notes, paper trades (paper account only). Log: logs\daily_run.log" `
    -Force | Out-Null

$task = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Write-Output "Installed '$name': weekdays at $At, next run $($info.NextRunTime). State: $($task.State)."

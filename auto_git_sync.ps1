
$ErrorActionPreference = "Stop"

# Project directory: folder containing this script
$RepoRoot = $PSScriptRoot

# Automatically detected branch
$Branch = (& git -C $RepoRoot branch --show-current).Trim()

if ($LASTEXITCODE -ne 0 -or -not $Branch) {
    throw "Could not detect Git branch."
}

# Verify GitHub remote
$Remote = & git -C $RepoRoot remote get-url origin

if ($LASTEXITCODE -ne 0) {
    throw "GitHub remote 'origin' is not configured."
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "       GIT AUTO-SYNC STARTED" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Project : $RepoRoot"
Write-Host "Branch  : $Branch"
Write-Host "Remote  : $Remote"
Write-Host "Checking for changes every 3 seconds."
Write-Host "Changes are committed after 8 seconds of inactivity."
Write-Host "Press CTRL+C to stop."
Write-Host ""

$LastStatus = ""
$QuietSince = Get-Date

while ($true) {
    try {
        $StatusLines = @(
            & git -C $RepoRoot status --porcelain --untracked-files=all
        )

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Could not check Git status." -ForegroundColor Red
            Start-Sleep -Seconds 5
            continue
        }

        $CurrentStatus = $StatusLines -join "`n"

        if ($CurrentStatus) {

            if ($CurrentStatus -ne $LastStatus) {
                $LastStatus = $CurrentStatus
                $QuietSince = Get-Date

                Write-Host "Project changes detected..." -ForegroundColor Yellow
            }

            $Elapsed = ((Get-Date) - $QuietSince).TotalSeconds

            if ($Elapsed -ge 8) {

                Write-Host "Preparing automatic commit..." -ForegroundColor Cyan

                & git -C $RepoRoot add -A

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Git staging failed." -ForegroundColor Red
                    Start-Sleep -Seconds 5
                    continue
                }

                & git -C $RepoRoot diff --cached --quiet
                $DiffExit = $LASTEXITCODE

                if ($DiffExit -eq 1) {

                    $CommitMessage = "Auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

                    & git -C $RepoRoot commit -m $CommitMessage

                    if ($LASTEXITCODE -eq 0) {

                        Write-Host "Commit created successfully." -ForegroundColor Green

                        & git -C $RepoRoot push origin $Branch

                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "GitHub updated successfully!" -ForegroundColor Green
                        }
                        else {
                            Write-Host "Push failed. The local commit is safe; check your connection or GitHub authentication." -ForegroundColor Red
                        }
                    }
                    else {
                        Write-Host "Commit failed. Check the Git output above." -ForegroundColor Red
                    }
                }
                elseif ($DiffExit -ne 0) {
                    Write-Host "Could not inspect staged changes." -ForegroundColor Red
                }

                $LastStatus = ""
                $QuietSince = Get-Date
            }
        }
        else {
            $LastStatus = ""
        }
    }
    catch {
        Write-Host "Auto-sync error: $_" -ForegroundColor Red
    }

    Start-Sleep -Seconds 3
}

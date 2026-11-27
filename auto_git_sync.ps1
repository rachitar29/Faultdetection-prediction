
$ErrorActionPreference = "Stop"

$ProjectPath = "D:\Rachita\faultdetection&prediction"
$Branch = "main"
$QuietSeconds = 8
$PollSeconds = 3

Set-Location -LiteralPath $ProjectPath

Write-Host "Automatic Git Sync Started" -ForegroundColor Green
Write-Host "Watching tracked files for changes..."
Write-Host "New files are NOT uploaded automatically."
Write-Host "Press Ctrl+C to stop."

$LastChange = $null
$LastSignature = ""

while ($true) {
    try {
        # Only inspect changes to files already tracked by Git.
        $Changes = @(git status --porcelain --untracked-files=no)

        if ($LASTEXITCODE -ne 0) {
            throw "Unable to read Git status."
        }

        $Signature = $Changes -join "`n"

        if ($Signature -ne $LastSignature) {
            $LastSignature = $Signature
            $LastChange = Get-Date
        }

        if ($Changes.Count -gt 0 -and $null -ne $LastChange) {
            $Elapsed = ((Get-Date) - $LastChange).TotalSeconds

            if ($Elapsed -ge $QuietSeconds) {
                Write-Host "`nChanges detected. Preparing commit..." -ForegroundColor Yellow

                git add -u

                if ($LASTEXITCODE -ne 0) {
                    throw "Git staging failed."
                }

                git diff --cached --quiet

                if ($LASTEXITCODE -eq 1) {
                    $CommitMessage = "Auto-sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
                    git commit -m $CommitMessage

                    if ($LASTEXITCODE -ne 0) {
                        throw "Git commit failed."
                    }

                    Write-Host "Commit created. Pushing to GitHub..." -ForegroundColor Cyan

                    git push origin $Branch

                    if ($LASTEXITCODE -ne 0) {
                        Write-Host "Push failed. Run git push origin main manually." -ForegroundColor Red
                    }
                    else {
                        Write-Host "Successfully pushed to GitHub." -ForegroundColor Green
                    }
                }
                elseif ($LASTEXITCODE -ne 0) {
                    throw "Unable to check staged changes."
                }

                $LastChange = $null
                $LastSignature = ""
            }
        }
        elseif ($Changes.Count -eq 0) {
            $LastChange = $null
            $LastSignature = ""
        }
    }
    catch {
        Write-Host "Sync error: $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Seconds $PollSeconds
}

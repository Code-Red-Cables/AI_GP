$p = Get-Process DCGame-Win64-Shipping -ErrorAction SilentlyContinue
if ($p) {
    $wshell = New-Object -ComObject WScript.Shell
    $wshell.AppActivate($p.Id)
    Start-Sleep -Milliseconds 500
    $wshell.SendKeys(" ")
    Write-Host "Sim reset sent to process $($p.Id)"
} else {
    Write-Warning "DCGame-Win64-Shipping process not found!"
}

param(
  [Parameter(Mandatory=$true)][string]$ArgsLine,
  [string]$OutName = "tune_last.txt",
  # Default: leave the already-running sim alone. Pass -Restart only when you
  # want a fresh FlightSim + race-start window during YOLO pre-warm.
  [switch]$Restart,
  [string]$FlightSim = $(
    if ($env:AIGP_FLIGHTSIM) { $env:AIGP_FLIGHTSIM }
    else { "C:\Users\trexx\OneDrive\Documents\AIGP_VQ1_3391\FlightSim.exe" }
  )
)
$ErrorActionPreference = "Continue"
Set-Location "D:\Code\Competitions\AI_GP"
$py = ".\winvenv\Scripts\python.exe"
$logDir = "logs\tuning"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$out = Join-Path $logDir $OutName

function Start-FlightSimFresh {
  param([string]$ExePath)
  if (-not (Test-Path $ExePath)) {
    Write-Host "FATAL: FlightSim not found at $ExePath"
    Write-Host "Set AIGP_FLIGHTSIM or pass -FlightSim <path>"
    exit 3
  }
  $workDir = Split-Path -Parent $ExePath
  Write-Host "=== Launching FlightSim (YOLO will warm next) ==="
  Write-Host "EXE: $ExePath"
  Get-Process FlightSim -ErrorAction SilentlyContinue | ForEach-Object {
    $mb = [math]::Round($_.WorkingSet64 / 1MB, 1)
    Write-Host "Killing old FlightSim PID $($_.Id) ($mb MB)"
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
  }
  Start-Sleep -Seconds 2
  Get-NetUDPEndpoint -LocalPort 14550,5600 -ErrorAction SilentlyContinue |
    ForEach-Object {
      $op = $_.OwningProcess
      if ($op -and $op -ne 0) {
        $p = Get-Process -Id $op -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -match "python") {
          Write-Host "Killing stuck $($p.ProcessName) PID=$op holding UDP $($_.LocalPort)"
          Stop-Process -Id $op -Force -ErrorAction SilentlyContinue
        }
      }
    }
  Start-Sleep -Seconds 1
  Start-Process -FilePath $ExePath -WorkingDirectory $workDir
  Write-Host ""
  Write-Host ">>> FlightSim launched. While YOLO warms (~10-20s):"
  Write-Host ">>>   log in and START A RACE in the sim window."
  Write-Host ">>> Do not leave the pad until the client arms."
  Write-Host ""
}

if ($Restart) {
  Start-FlightSimFresh -ExePath $FlightSim
} else {
  Write-Host "=== Using existing FlightSim (pass -Restart to relaunch) ==="
}

Write-Host "RUNNING: $py tools\tune_flight.py $ArgsLine"
Write-Host "LOG: $out"
$argList = @("tools\tune_flight.py") + ($ArgsLine -split '\s+')
& $py @argList 2>&1 | Tee-Object -FilePath $out
exit $LASTEXITCODE

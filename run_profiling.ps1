# PowerShell script to run profiling with Docker
# All outputs saved to D:\temp\neural_net_profiling

$OutputDir = "D:\temp\neural_net_profiling"

Write-Host "=== Neural Network Profiling Setup ===" -ForegroundColor Cyan
Write-Host ""

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    Write-Host "Creating output directory: $OutputDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
} else {
    Write-Host "Output directory exists: $OutputDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -f Dockerfile.profiling -t neural-net-profiler .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Starting Profiling Container ===" -ForegroundColor Cyan
Write-Host "Outputs will be saved to: $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "Inside the container, run:" -ForegroundColor Yellow
Write-Host "  ./profile_in_docker.sh" -ForegroundColor White
Write-Host ""
Write-Host "When done, exit with: exit" -ForegroundColor Yellow
Write-Host ""

# Run container with privileged mode for perf, and mount output directory
docker run --privileged `
    -v ${PWD}:/workspace `
    -v ${OutputDir}:/output `
    -it neural-net-profiler

Write-Host ""
Write-Host "=== Profiling Session Ended ===" -ForegroundColor Cyan
Write-Host "Check for output files in: $OutputDir" -ForegroundColor Green

# Check if flamegraph was generated
$FlamegraphPath = Join-Path $OutputDir "flamegraph.svg"
if (Test-Path $FlamegraphPath) {
    Write-Host ""
    Write-Host "Flamegraph generated successfully!" -ForegroundColor Green
    Write-Host "Open with: start $FlamegraphPath" -ForegroundColor White
    
    # Offer to open it
    $response = Read-Host "Open flamegraph now? (y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Start-Process $FlamegraphPath
    }
} else {
    Write-Host ""
    Write-Host "No flamegraph found. Did the profiling run successfully?" -ForegroundColor Yellow
}

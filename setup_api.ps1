# Vehicle Sales Forecasting API Setup and Launch Script
# This PowerShell script sets up the environment and launches the API server

Write-Host "🏭 Vehicle Sales Forecasting API Setup" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8 or higher." -ForegroundColor Red
    exit 1
}

# Check if pip is available
try {
    pip --version | Out-Null
    Write-Host "✅ pip is available" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please ensure pip is installed." -ForegroundColor Red
    exit 1
}

# Navigate to the Code directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = Join-Path $scriptPath "Code"

if (Test-Path $codeDir) {
    Set-Location $codeDir
    Write-Host "✅ Changed to Code directory" -ForegroundColor Green
} else {
    Write-Host "❌ Code directory not found at: $codeDir" -ForegroundColor Red
    exit 1
}

# Check if virtual environment should be created
$createVenv = Read-Host "Do you want to create a virtual environment? (recommended) [y/N]"

if ($createVenv -eq "y" -or $createVenv -eq "Y") {
    Write-Host "🔄 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv_api
    
    if (Test-Path "venv_api\Scripts\Activate.ps1") {
        Write-Host "✅ Virtual environment created" -ForegroundColor Green
        Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
        & "venv_api\Scripts\Activate.ps1"
        Write-Host "✅ Virtual environment activated" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Install dependencies
Write-Host "🔄 Installing API dependencies..." -ForegroundColor Yellow
$requirementsFile = "..\requirements_api.txt"

if (Test-Path $requirementsFile) {
    pip install -r $requirementsFile
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ requirements_api.txt not found at: $requirementsFile" -ForegroundColor Red
    exit 1
}

# Check if forecast data exists
$predictionsFile = "forecast_results\forecast_predictions.csv"
$metricsFile = "forecast_results\forecast_metrics.csv"

if ((Test-Path $predictionsFile) -and (Test-Path $metricsFile)) {
    Write-Host "✅ Forecast data found" -ForegroundColor Green
} else {
    Write-Host "📊 Forecast data not found" -ForegroundColor Yellow
    $runPipeline = Read-Host "Do you want to run the forecasting pipeline to generate data? [y/N]"
    
    if ($runPipeline -eq "y" -or $runPipeline -eq "Y") {
        Write-Host "🔄 Running forecasting pipeline..." -ForegroundColor Yellow
        Write-Host "This may take a few minutes..." -ForegroundColor Yellow
        
        python main_pipeline.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Forecasting pipeline completed successfully" -ForegroundColor Green
        } else {
            Write-Host "❌ Forecasting pipeline failed" -ForegroundColor Red
            Write-Host "Please check the error messages above" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Forecast data is required to run the API server" -ForegroundColor Red
        Write-Host "Please run the forecasting pipeline first: python main_pipeline.py" -ForegroundColor Yellow
        exit 1
    }
}

# Start the API server
Write-Host ""
Write-Host "🚀 Starting Vehicle Sales Forecasting API server..." -ForegroundColor Cyan
Write-Host "Server will be available at: http://localhost:5000" -ForegroundColor Green
Write-Host "API Documentation: ../API_Documentation.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "Available endpoints:" -ForegroundColor Yellow
Write-Host "  • GET /health - Health check" -ForegroundColor White
Write-Host "  • GET /api/v1/predictions - All model predictions" -ForegroundColor White
Write-Host "  • GET /api/v1/metrics - Model performance metrics" -ForegroundColor White
Write-Host "  • GET /api/v1/models - Models information" -ForegroundColor White
Write-Host "  • GET /api/v1/forecast/{model_name} - Specific model forecast" -ForegroundColor White
Write-Host "  • GET /api/v1/comparison - Model comparison" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Cyan

# Test the demo script option
$runDemo = Read-Host "Do you want to run the API demo script after starting the server? [y/N]"

if ($runDemo -eq "y" -or $runDemo -eq "Y") {
    Write-Host "Demo script will run in a separate window after the server starts" -ForegroundColor Yellow
    
    # Start API server in background and run demo
    Start-Process -FilePath "python" -ArgumentList "app.py" -NoNewWindow
    Start-Sleep -Seconds 3
    Start-Process -FilePath "python" -ArgumentList "api_demo.py"
} else {
    # Start API server
    python app.py
}

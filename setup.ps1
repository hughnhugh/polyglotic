#Requires -Version 5.1
<#
    One-time setup for vc-translator: locates a usable Python install, creates
    the project's venv (venv/) if it doesn't already exist, and installs every
    dependency. Safe to re-run - an existing venv is reused, not recreated, so
    re-running this after a `git pull` just catches up on any new packages.

    This does NOT install Python or NVIDIA drivers themselves - those are
    system-wide changes this script won't make for you; it just detects them
    and tells you what's missing.
#>

$ErrorActionPreference = "Stop"

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Write-Warn($text) {
    Write-Host "WARNING: $text" -ForegroundColor Yellow
}

function Exit-WithError($text) {
    Write-Host ""
    Write-Host "ERROR: $text" -ForegroundColor Red
    exit 1
}

# --- Locate Python -----------------------------------------------------
Write-Step "Looking for Python 3.12"

$pythonExe = $null
$pythonPrefixArgs = @()

try {
    $pyVersion = & py -3.12 --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = "py"
        $pythonPrefixArgs = @("-3.12")
        Write-Host "Found via 'py' launcher: $pyVersion"
    }
} catch {}

if (-not $pythonExe) {
    try {
        $pyVersion = & python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            if ($pyVersion -notmatch "3\.12") {
                Write-Warn "Found $pyVersion, not 3.12 - vc-translator is tested on 3.12. Continuing anyway; if package installs fail below, install Python 3.12 from https://www.python.org/downloads/ and re-run this script."
            } else {
                Write-Host "Found: $pyVersion"
            }
            $pythonExe = "python"
        }
    } catch {}
}

if (-not $pythonExe) {
    Exit-WithError "No Python install found. Install Python 3.12 from https://www.python.org/downloads/ (check 'Add python.exe to PATH' during install), then re-run this script."
}

# --- Check for an NVIDIA GPU (warn, don't block) ------------------------
Write-Step "Checking for an NVIDIA GPU"

try {
    $null = & nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NVIDIA GPU detected."
    } else {
        Write-Warn "'nvidia-smi' ran but returned an error - vc-translator requires an NVIDIA GPU with CUDA support. It will not run without one."
    }
} catch {
    Write-Warn "'nvidia-smi' not found - vc-translator requires an NVIDIA GPU with CUDA support (and its driver installed). It will not run without one."
}

# --- Create the venv (if it doesn't already exist) ----------------------
$venvPython = ".\venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    Write-Step "Reusing existing venv\"
} else {
    Write-Step "Creating venv\"
    & $pythonExe @pythonPrefixArgs -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Failed to create the venv."
    }
}

function Install-Packages($description, $pipArgs, $required = $true) {
    Write-Step $description
    & $venvPython -m pip install @pipArgs
    if ($LASTEXITCODE -ne 0) {
        if ($required) {
            Exit-WithError "Failed installing: $description"
        } else {
            Write-Warn "Failed installing (optional, continuing): $description"
        }
    }
}

Install-Packages "Upgrading pip" @("--upgrade", "pip")
Install-Packages "Installing PyTorch (CUDA build - this is a large download)" `
    @("torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128")
Install-Packages "Installing core dependencies" `
    @("faster-whisper", "PyAudioWPatch", "numpy", "opencc-python-reimplemented", "websockets")
Install-Packages "Installing GPU memory readout support (optional)" @("nvidia-ml-py") -required $false

# --- Translator: llama.cpp server + Hy-MT2 model --------------------------
# Fetched with Windows' built-in curl.exe rather than Python/huggingface_hub:
# it resumes a partial download (-C -) instead of starting over, which
# matters for a multi-GB file - huggingface_hub's downloader was observed
# dropping mid-file on a slow connection and discarding the partial.

$llamaBuild = "b11796"
$llamaBase = "https://github.com/ggml-org/llama.cpp/releases/download/$llamaBuild"
$modelUrl = "https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/main/Hy-MT2-7B-Q4_K_M.gguf"
$modelPath = "models\Hy-MT2-7B-Q4_K_M.gguf"
$modelSize = 4624648896

function Get-Download($url, $dest, $expectedSize = 0) {
    $part = "$dest.part"
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        & curl.exe -L -f -C - --retry 5 --retry-all-errors -o $part $url
        $have = if (Test-Path $part) { (Get-Item $part).Length } else { 0 }
        if ($LASTEXITCODE -eq 0 -and ($expectedSize -eq 0 -or $have -eq $expectedSize)) {
            Move-Item -Force $part $dest
            return
        }
        Write-Warn "Download interrupted ($have bytes so far) - resuming (attempt $attempt)..."
    }
    Exit-WithError "Failed downloading $url - re-run this script to resume where it left off."
}

Write-Step "Checking the GPU driver's CUDA version"
$cudaLine = (& nvidia-smi 2>$null | Select-String -Pattern "CUDA (UMD )?Version:\s*([0-9]+)\.").Matches
if ($cudaLine -and [int]$cudaLine[0].Groups[2].Value -lt 13) {
    Write-Warn "Your NVIDIA driver supports CUDA $($cudaLine[0].Groups[2].Value).x - the translator's llama.cpp build needs CUDA 13 (driver 580 or newer). Update from https://www.nvidia.com/download/index.aspx if translation fails to load."
}

if (Test-Path "bin\llama-server.exe") {
    Write-Step "Reusing existing llama.cpp server (bin\)"
} else {
    Write-Step "Downloading llama.cpp server ($llamaBuild, CUDA 13)"
    New-Item -ItemType Directory -Force bin | Out-Null
    foreach ($zip in @("llama-$llamaBuild-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip")) {
        $zipPath = Join-Path $env:TEMP $zip
        Get-Download "$llamaBase/$zip" $zipPath
        Expand-Archive -Force -Path $zipPath -DestinationPath bin
        Remove-Item $zipPath
    }
}

if ((Test-Path $modelPath) -and (Get-Item $modelPath).Length -eq $modelSize) {
    Write-Step "Reusing existing translation model ($modelPath)"
} else {
    Write-Step "Downloading the Hy-MT2-7B translation model (4.6GB - resumes if interrupted)"
    New-Item -ItemType Directory -Force models | Out-Null
    Get-Download $modelUrl $modelPath $modelSize
}

Write-Step "Setup complete"
Write-Host "Run vc-translator with:"
Write-Host "  .\venv\Scripts\python.exe translate_vc.py"
Write-Host ""
Write-Host "First launch downloads the Whisper and VAD models (~3GB) -"
Write-Host "this only happens once and needs an internet connection."

#Requires -Version 5.1
<#
    Setup for vc-translator on an 8GB NVIDIA GPU (an RTX 4060 Ti, say) -
    double-click setup.bat.

    All it needs is Windows and an NVIDIA driver. It gets its own Python 3.12
    if there isn't one (through uv), makes one environment (venv\) with every
    package, and fetches the llama.cpp server (bin\), the models (models\)
    and the word-lookup dictionaries (data\). Safe to re-run: what's already
    there is kept, and an interrupted download picks up where it stopped.

    Already have the models? Put them in models\ first (models\README.md
    says what goes where) and they're used instead of downloaded - or point
    setup at a folder laid out the same way, like another install's models\:
        setup.bat -ModelsFrom D:\vc-translator-models

    The models, and the GPU memory they take:
      default             Qwen3-ASR-0.6B + Hy-MT2-1.8B Q8_0    4.2GB
      -LargerSpeechModel  Qwen3-ASR-1.7B + Hy-MT2-1.8B Q4_K_M  6.2GB - better
                          on Cantonese, but tight on 8GB
      -Assistant          adds the local assistant (Qwen3.5-4B, ~3GB more -
                          for 12GB+ GPUs)
    -NoShortcut leaves the Start Menu alone.
#>

param([string]$ModelsFrom = "", [switch]$LargerSpeechModel, [switch]$Assistant, [switch]$NoShortcut)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$uvVersion = "0.13.7"
$llamaBuild = "b11796"

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

# Fetched with Windows' built-in curl.exe: it resumes a partial download
# (-C -) instead of starting over, which matters for a multi-GB file.
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
    Exit-WithError "Failed downloading $url - re-run setup to resume where it left off."
}

# models\<path>: kept if it's there (and the right size, when that's known),
# else copied from -ModelsFrom if it's there, else downloaded.
function Get-ModelFile($path, $url, $size = 0) {
    $dest = Join-Path "models" $path
    if ((Test-Path $dest) -and ($size -eq 0 -or (Get-Item $dest).Length -eq $size)) {
        return "kept"
    }
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    if ($ModelsFrom) {
        $source = Join-Path $ModelsFrom $path
        if ((Test-Path $source) -and ($size -eq 0 -or (Get-Item $source).Length -eq $size)) {
            Copy-Item -Force $source $dest
            return "copied from $ModelsFrom"
        }
    }
    Get-Download $url $dest $size
    return "downloaded"
}

function Get-Model($title, $files) {
    # $files: @(path in models\, url, size or 0), ...
    Write-Step $title
    $how = @($files | ForEach-Object { Get-ModelFile $_[0] $_[1] $_[2] }) | Select-Object -Unique
    Write-Host ($how -join ", ")
}

if ($ModelsFrom -and -not (Test-Path $ModelsFrom)) {
    Exit-WithError "-ModelsFrom $ModelsFrom doesn't exist."
}

# --- The GPU -------------------------------------------------------------
Write-Step "Checking the NVIDIA GPU"
$gpu = $null
try { $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1 } catch {}
if (-not $gpu) {
    Write-Warn "No NVIDIA GPU found (nvidia-smi didn't answer). vc-translator needs one, with its driver - get the latest from https://www.nvidia.com/download/index.aspx. Carrying on with the rest of setup."
} else {
    $gpuName, $gpuMiB = $gpu -split ",\s*"
    Write-Host "$gpuName - $([math]::Round([int]$gpuMiB / 1024)) GB"
    if ([int]$gpuMiB -lt 7500) {
        Write-Warn "Under 8GB of GPU memory - the models need 4.2GB, and Windows, the browser and Discord use some too."
    } elseif ($LargerSpeechModel -and [int]$gpuMiB -lt 11000) {
        Write-Warn "-LargerSpeechModel takes 6.2GB of this GPU's memory - tight beside a browser or a game."
    }
    $cuda = (& nvidia-smi 2>$null | Select-String -Pattern "CUDA (UMD )?Version:\s*([0-9]+)\.").Matches
    if ($cuda -and [int]$cuda[0].Groups[2].Value -lt 13) {
        Write-Warn "Your NVIDIA driver supports CUDA $($cuda[0].Groups[2].Value).x - the translator needs CUDA 13 (driver 580 or newer). Update from https://www.nvidia.com/download/index.aspx."
    }
}

# --- Python and the packages (through uv) ----------------------------------
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $PSScriptRoot "bin\uv\uv.exe"
    if (-not (Test-Path $uv)) {
        Write-Step "Getting uv $uvVersion (it installs Python and the packages)"
        New-Item -ItemType Directory -Force bin\uv | Out-Null
        $zip = Join-Path $env:TEMP "uv-$uvVersion.zip"
        Get-Download "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip" $zip
        Expand-Archive -Force -Path $zip -DestinationPath bin\uv
        Remove-Item $zip
    }
}

$venvPython = "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    Write-Step "Reusing venv\"
} else {
    Write-Step "Creating venv\ with Python 3.12 (downloaded if it isn't installed)"
    & $uv venv venv --python 3.12
    if ($LASTEXITCODE -ne 0) { Exit-WithError "Couldn't create venv\." }
}

Write-Step "Installing PyTorch with CUDA (~2.5GB the first time)"
& $uv pip install --python $venvPython "torch==2.11.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Exit-WithError "Failed installing PyTorch." }

Write-Step "Installing the app's packages"
& $uv pip install --python $venvPython -r requirements.txt
if ($LASTEXITCODE -ne 0) { Exit-WithError "Failed installing the packages in requirements.txt." }

Write-Step "Checking PyTorch can use the GPU"
& $venvPython -c "import torch; print(f'PyTorch {torch.__version__}: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA GPU - speech recognition will fail'))"

# --- The translator's server: llama.cpp ------------------------------------
if (Test-Path "bin\llama-server.exe") {
    Write-Step "Reusing the llama.cpp server (bin\)"
} else {
    Write-Step "Downloading the llama.cpp server ($llamaBuild, CUDA 13)"
    New-Item -ItemType Directory -Force bin | Out-Null
    foreach ($zip in @("llama-$llamaBuild-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip")) {
        $zipPath = Join-Path $env:TEMP $zip
        Get-Download "https://github.com/ggml-org/llama.cpp/releases/download/$llamaBuild/$zip" $zipPath
        Expand-Archive -Force -Path $zipPath -DestinationPath bin
        Remove-Item $zipPath
    }
}

# --- The models ----------------------------------------------------------------
$hf = "https://huggingface.co"

if ($LargerSpeechModel) {
    $speech = "Qwen3-ASR-1.7B"
    $weights = @(@("model-00001-of-00002.safetensors", 4220320824), @("model-00002-of-00002.safetensors", 478200688),
                 @("model.safetensors.index.json", 0))
    $translatorFile, $translatorSize = "Hy-MT2-1.8B-Q4_K_M.gguf", 1133080448
} else {
    $speech = "Qwen3-ASR-0.6B"
    $weights = @(,@("model.safetensors", 1876091704))  # leading comma: keeps a 1-element list of pairs from flattening
    $translatorFile, $translatorSize = "Hy-MT2-1.8B-Q8_0.gguf", 1908528192
}
$speechFiles = @("chat_template.json", "config.json", "generation_config.json", "merges.txt", "preprocessor_config.json",
                 "tokenizer_config.json", "vocab.json" | ForEach-Object { ,@($_, 0) }) + $weights
Get-Model "Speech recognition: $speech" @($speechFiles | ForEach-Object {
    ,@("$speech\$($_[0])", "$hf/Qwen/$speech/resolve/main/$($_[0])", $_[1]) })

Get-Model "Translation: $translatorFile" @(,@($translatorFile, "$hf/tencent/Hy-MT2-1.8B-GGUF/resolve/main/$translatorFile",
                                             $translatorSize))

Get-Model "Word links: awesome-align (1.1GB, runs on the CPU)" @("config.json", "vocab.txt", "tokenizer_config.json",
    "special_tokens_map.json", "pytorch_model.bin" | ForEach-Object {
    ,@("awesome-align-with-co\$_", "$hf/aneuraz/awesome-align-with-co/resolve/main/$_", 0) })

if ($Assistant) {
    Get-Model "Local assistant: Qwen3.5-4B (2.7GB)" @(,@("Qwen3.5-4B-Q4_K_M.gguf",
        "$hf/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf", 2740937888))
}

Write-Step "Speech detection: Silero VAD"
& $venvPython -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"
if ($LASTEXITCODE -ne 0) { Write-Warn "Couldn't fetch Silero VAD now - the app fetches it on its first start instead." }

# --- Word lookup: dictionaries (data\) --------------------------------------------
New-Item -ItemType Directory -Force data | Out-Null
foreach ($d in @(
    @("data\cedict.txt.gz", "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz", "CC-CEDICT, Chinese-English (4MB)"),
    @("data\vi-wiktionary.jsonl.gz", "https://kaikki.org/dictionary/Vietnamese/kaikki.org-dictionary-Vietnamese.jsonl.gz", "Vietnamese, from Wiktionary (13MB)"),
    @("data\word-frequency.txt", "https://raw.githubusercontent.com/fxsjy/jieba/master/jieba/dict.txt", "Chinese word frequencies (5MB)")
)) {
    if (-not (Test-Path $d[0])) {
        Write-Step "Dictionary: $($d[2])"
        Get-Download $d[1] $d[0]
    }
}

# --- The Start Menu ------------------------------------------------------------
if (-not $NoShortcut) {
    Write-Step "Adding vc-translator to the Start Menu"
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Programs")) "vc-translator.lnk"))
    $link.TargetPath = Join-Path $PSScriptRoot "venv\Scripts\pythonw.exe"
    $link.Arguments = '"' + (Join-Path $PSScriptRoot "translate_vc.py") + '"'
    $link.WorkingDirectory = $PSScriptRoot
    $link.Description = "Live captions and translation"
    $link.Save()
}

Write-Step "Setup complete"
Write-Host "Start vc-translator from the Start Menu, or double-click start.bat."
Write-Host "Models: $speech + $translatorFile - about $(if ($LargerSpeechModel) { '6.2' } else { '4.2' })GB of GPU memory."

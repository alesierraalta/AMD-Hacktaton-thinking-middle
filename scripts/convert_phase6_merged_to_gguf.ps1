param(
    [string]$MergedModelDir = "results/phase6_merged_model",
    [string]$LlamaCppDir = "llama.cpp",
    [string]$F16Gguf = "results/codepause-qwen15-merged-f16.gguf",
    [string]$Q4Gguf = "results/codepause-qwen15-merged-Q4_K_M.gguf",
    [string]$QuantType = "Q4_K_M"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $MergedModelDir)) {
    throw "Missing merged model directory: $MergedModelDir. Run: python scripts/merge_phase6_adapter.py"
}

if (-not (Test-Path -LiteralPath $LlamaCppDir)) {
    git clone https://github.com/ggml-org/llama.cpp $LlamaCppDir
}

python -m pip install -r "$LlamaCppDir/requirements.txt"
if (-not $?) {
    Write-Warning "Full llama.cpp requirements install failed. Installing minimal conversion dependency instead."
    python -m pip install sentencepiece
}

$quantizeCandidates = @(
    "$LlamaCppDir/build/bin/Release/llama-quantize.exe",
    "$LlamaCppDir/build/bin/llama-quantize.exe",
    "$LlamaCppDir/build/bin/llama-quantize"
)

$quantizeBin = $quantizeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $quantizeBin) {
    if (Get-Command cmake -ErrorAction SilentlyContinue) {
        cmake -S $LlamaCppDir -B "$LlamaCppDir/build"
        cmake --build "$LlamaCppDir/build" --config Release
    }
}

python "$LlamaCppDir/convert_hf_to_gguf.py" `
    $MergedModelDir `
    --outfile $F16Gguf `
    --outtype f16

$quantizeBin = $quantizeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $quantizeBin) {
    $prebuiltDir = Join-Path $env:TEMP "llama-b9090-bin-win-cpu-x64"
    $prebuiltZip = "$prebuiltDir.zip"
    $prebuiltQuantize = Join-Path $prebuiltDir "llama-quantize.exe"

    if (-not (Test-Path -LiteralPath $prebuiltQuantize)) {
        Invoke-WebRequest `
            -Uri "https://github.com/ggml-org/llama.cpp/releases/download/b9090/llama-b9090-bin-win-cpu-x64.zip" `
            -OutFile $prebuiltZip
        Expand-Archive -LiteralPath $prebuiltZip -DestinationPath $prebuiltDir -Force
    }

    if (-not (Test-Path -LiteralPath $prebuiltQuantize)) {
        throw "Missing llama-quantize binary after build/prebuilt fallback. Checked: $($quantizeCandidates -join ', '), $prebuiltQuantize"
    }

    $quantizeBin = $prebuiltQuantize
}

& $quantizeBin $F16Gguf $Q4Gguf $QuantType

"Quantized GGUF saved to $Q4Gguf"
"Import with: lms import $Q4Gguf"

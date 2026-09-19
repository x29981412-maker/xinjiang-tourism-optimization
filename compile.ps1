$ErrorActionPreference = "Stop"
$paperRoot = Join-Path $PSScriptRoot "paper"
Push-Location -LiteralPath $paperRoot
try {
    & latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX compilation failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

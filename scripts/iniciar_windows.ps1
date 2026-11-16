$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
Write-Host "=== VOICE OF CUSTOMER - TECH CHALLENGE FASE 4 ===" -ForegroundColor Cyan

$py = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @('-3.12', '-3.11', '-3')) {
        & py $version --version *> $null
        if ($LASTEXITCODE -eq 0) { $py = @('py', $version); break }
    }
}
if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) {
    & python --version *> $null
    if ($LASTEXITCODE -eq 0) { $py = @('python') }
}
if (-not $py) { throw 'Instale Python 3.11 ou 3.12: https://www.python.org/downloads/ (marque Add Python to PATH).' }

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host 'Criando ambiente Python...'
    if ($py.Length -gt 1) { & $py[0] $py[1] -m venv .venv } else { & $py[0] -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel criar o ambiente virtual.' }
}
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path '.venv\.dependencias_instaladas')) {
    Write-Host 'Instalando dependencias...'
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao atualizar pip.' }
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependencias.' }
    New-Item '.venv\.dependencias_instaladas' -ItemType File | Out-Null
}
if (-not (Test-Path 'artifacts\manifest.json')) {
    Write-Host 'Baixando a base e preparando embeddings (primeira execucao)...'
    & $python scripts\build_index.py
    if ($LASTEXITCODE -ne 0) { throw 'Erro ao construir indice. Baixe CSVs do Kaggle em data/raw e execute novamente.' }
}
Write-Host 'Iniciando aplicativo...'
& $python -m streamlit run app.py

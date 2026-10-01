param([string]$Python = 'python', [int]$Port = 8010)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw '请先配置 Docker Desktop 的 Linux 容器环境，详见 docs/milvus.md。'
}
Push-Location -LiteralPath $projectDirectory
$previousBackend = $env:RAG_BACKEND
$previousUri = $env:MILVUS_URI
try {
    docker compose -f compose.milvus.yml up -d --wait --wait-timeout 300
    if ($LASTEXITCODE -ne 0) { throw 'Milvus 启动失败，请查看 Docker 服务日志。' }
    # Only this launch process changes backend. Keep .env and its keys untouched.
    $env:RAG_BACKEND = 'milvus'
    $env:MILVUS_URI = 'http://127.0.0.1:19530'
    & $Python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'API 启动失败。请确认 Python 已安装项目依赖。' }
}
finally {
    $env:RAG_BACKEND = $previousBackend
    $env:MILVUS_URI = $previousUri
    Pop-Location
}

# Forward all arguments to the project CLI; reuse the D-drive environment from steps 1/2.
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskRuntime = Join-Path (Split-Path -Parent $taskRepo) 'tfm_runtime'
. (Join-Path $taskRuntime 'env.ps1')
Push-Location -LiteralPath $taskRepo
try {
    & (Join-Path $taskRuntime 'envs\tfm\Scripts\python.exe') -m harness @args
    $taskExit = $LASTEXITCODE
}
finally { Pop-Location }
exit $taskExit

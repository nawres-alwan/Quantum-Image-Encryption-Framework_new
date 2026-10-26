param(
    [string]$PythonExecutable = 'python',
    [switch]$ReplayRecordedKeys
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runArguments = @('--replicates', '3', '--chunk-size', '2048')
if ($ReplayRecordedKeys) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'private_keys'))) {
        throw 'Private research keys are not distributed with this public repository.'
    }
    $runArguments += '--replay-recorded-keys'
}
& $PythonExecutable (Join-Path $PSScriptRoot 'run_experiments.py') @runArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExecutable (Join-Path $PSScriptRoot 'analyze_results.py')
exit $LASTEXITCODE

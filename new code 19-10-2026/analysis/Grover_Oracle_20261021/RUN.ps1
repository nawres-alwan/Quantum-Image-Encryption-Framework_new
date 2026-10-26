param([string]$PythonExecutable = 'python')
$ErrorActionPreference = 'Stop'
& $PythonExecutable -X utf8 (Join-Path $PSScriptRoot 'test_qef_gsa_oracle.py')
if ($LASTEXITCODE -ne 0) { throw 'Oracle validation failed.' }
& $PythonExecutable -X utf8 (Join-Path $PSScriptRoot 'qef_gsa_oracle.py')
if ($LASTEXITCODE -ne 0) { throw 'Oracle experiments failed.' }

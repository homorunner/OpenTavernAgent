<#
.SYNOPSIS
    Builds the plugin (including the tavernplugin client) and installs it into the game.

.DESCRIPTION
    Thin wrapper over hearthstone-helper-plugin\deploy.ps1, which does the real work:
    it resolves MSBuild, builds HSLibrary2.dll, picks the UnityDoorstop proxy that
    matches Hearthstone.exe, installs both and verifies the hashes.

    Run from an elevated PowerShell session: C:\Program Files (x86) is not writable
    for a standard user.

.EXAMPLE
    .\deploy.ps1
.EXAMPLE
    .\deploy.ps1 -SkipBuild          # reinstall the DLL that is already in bin\Release
#>
[CmdletBinding()]
param(
    [string] $GameDir = 'C:\Program Files (x86)\Hearthstone',
    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Release',
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'

$deploy = Join-Path $PSScriptRoot '..\hearthstone-helper-plugin\deploy.ps1'
$deploy = [System.IO.Path]::GetFullPath($deploy)
if (-not (Test-Path -LiteralPath $deploy)) {
    throw "cannot find $deploy - the plugin framework is expected next to tavernplugin\"
}

$buildArgs = @{
    GameDir       = $GameDir
    Configuration = $Configuration
}
if ($SkipBuild) { $buildArgs['SkipBuild'] = $true }

& $deploy @buildArgs

# The plugin starts the server itself, and looks for it in
# <game folder>\tavernplugin\server, so ship the Python files next to the DLL.
$serverSrc = Join-Path $PSScriptRoot 'server'
$serverDst = Join-Path $GameDir 'tavernplugin\server'
if (Test-Path -LiteralPath $serverSrc) {
    New-Item -ItemType Directory -Force -Path $serverDst | Out-Null
    Copy-Item -Path (Join-Path $serverSrc '*.py') -Destination $serverDst -Force
    Write-Host "Installing  : server -> $serverDst"
}

Write-Host ''
Write-Host 'Next:'
Write-Host '  1. launch Hearthstone - the plugin starts the bridge server automatically'
Write-Host '  2. enter a Battlegrounds game; the overlay is top-right, F8 toggles it'
Write-Host '  3. logs: %USERPROFILE%\Documents\TavernPlugin\ (server.log, config.json)'
Write-Host '     and %USERPROFILE%\Documents\tavernplugin.log'

@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Qwen3.8 - Ollama local server launcher

REM ============================================================================
REM Qwen3.8 local Ollama server launcher for Windows 10/11
REM
REM Hardware target:
REM   NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
REM   Intel Core i7-13700H
REM   32 GB system RAM
REM
REM The model may be larger than VRAM. Ollama can split model execution between
REM GPU VRAM and system RAM/CPU. The exact split is shown by "ollama ps".
REM
REM SECURITY:
REM   Default bind = 127.0.0.1:11434 (LOCAL MACHINE ONLY)
REM   Use the third argument for a specific LAN IPv4 address when another PC
REM   needs to connect.
REM   0.0.0.0 is supported but listens on every IPv4 interface and is NOT
REM   the default.
REM
REM IMPORTANT:
REM   Quit the Ollama tray application and any existing "ollama serve" process
REM   before running this script. This launcher does NOT kill existing processes.
REM ============================================================================

REM ---------- Editable defaults ----------
set "QWEN_SOURCE=qwen3.8:27b"
set "QWEN_MODEL=qwen3.8-27b-local:latest"

REM 64K is a practical starting point for an 8 GB RTX 4060 + 32 GB RAM.
REM You can pass 100000 or 131072 as argument 1, but memory use rises sharply.
set "QWEN_CONTEXT=65536"

REM q4_0 uses less KV-cache memory than q8_0.
REM Change to q8_0 if you have enough memory and want higher KV precision.
set "QWEN_CACHE=q4_0"

set "QWEN_KEEP_ALIVE=30m"
set "QWEN_LOAD_TIMEOUT=15m"
set "QWEN_MODELFILE="

REM LOCAL-ONLY default.
set "QWEN_BIND_IP=127.0.0.1"

REM Do not use the Intel integrated GPU for Ollama inference.
set "OLLAMA_IGPU_ENABLE=0"
REM ---------------------------------------

REM ---------- Command-line overrides ----------
if /I "%~1"=="--help" goto help
if /I "%~1"=="/?" goto help
if /I "%~1"=="-h" goto help

if not "%~1"=="" set "QWEN_CONTEXT=%~1"
if not "%~2"=="" set "QWEN_SOURCE=%~2"
if not "%~3"=="" set "QWEN_BIND_IP=%~3"

REM Accept "localhost" as a friendly alias, then normalize it to IPv4 loopback.
if /I "%QWEN_BIND_IP%"=="localhost" set "QWEN_BIND_IP=127.0.0.1"

where powershell.exe >nul 2>&1
if errorlevel 1 goto no_powershell

REM ---------- Validate settings ----------
powershell.exe -NoLogo -NoProfile -Command ^
  "$ctxOk = $env:QWEN_CONTEXT -in @('32768','65536','100000','131072'); " ^
  "if (-not $ctxOk) { Write-Host 'Use context 32768, 65536, 100000, or 131072.'; exit 1 }; " ^
  "$namePattern = '^[A-Za-z0-9][A-Za-z0-9._:/-]*$'; " ^
  "if ($env:QWEN_SOURCE -notmatch $namePattern -or $env:QWEN_MODEL -notmatch $namePattern) { Write-Host 'Use a valid local Ollama model name.'; exit 1 }; " ^
  "if ($env:QWEN_SOURCE -match '(?i)cloud' -or $env:QWEN_MODEL -match '(?i)cloud') { Write-Host 'Choose a downloaded local model, without a cloud tag.'; exit 1 }; " ^
  "if ($env:QWEN_CACHE -notin @('q8_0','f16','q4_0')) { Write-Host 'Cache must be q8_0, f16, or q4_0.'; exit 1 }; " ^
  "if ($env:QWEN_SOURCE -ieq $env:QWEN_MODEL) { Write-Host 'The source and local preset must have different names.'; exit 1 }"
if errorlevel 1 goto failed

powershell.exe -NoLogo -NoProfile -Command ^
  "$ip = $null; " ^
  "if (-not [System.Net.IPAddress]::TryParse($env:QWEN_BIND_IP,[ref]$ip) -or " ^
  "$ip.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) { " ^
  "Write-Host 'The bind address must be an IPv4 address such as 127.0.0.1, 0.0.0.0, or 192.168.1.10.'; exit 1 }"
if errorlevel 1 goto failed

REM The server bind address and the client address are different concepts.
REM When bound to 0.0.0.0, use 127.0.0.1 for local CLI/API checks.
if "%QWEN_BIND_IP%"=="0.0.0.0" (
    set "QWEN_CLIENT_IP=127.0.0.1"
) else (
    set "QWEN_CLIENT_IP=%QWEN_BIND_IP%"
)
set "QWEN_LOCAL_API=http://%QWEN_CLIENT_IP%:11434"

REM ---------- Find Ollama ----------
set "QWEN_OLLAMA_EXE="
for /f "delims=" %%I in ('where ollama.exe 2^>nul') do if not defined QWEN_OLLAMA_EXE set "QWEN_OLLAMA_EXE=%%I"
if defined QWEN_OLLAMA_EXE goto found_ollama
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "QWEN_OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined QWEN_OLLAMA_EXE goto no_ollama

:found_ollama
REM ---------- Server environment ----------
REM These variables are inherited by the NEW "ollama serve" process.
set "OLLAMA_HOST=%QWEN_BIND_IP%:11434"
set "OLLAMA_NO_CLOUD=1"
set "OLLAMA_FLASH_ATTENTION=1"
set "OLLAMA_KV_CACHE_TYPE=%QWEN_CACHE%"
set "OLLAMA_CONTEXT_LENGTH=%QWEN_CONTEXT%"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_MAX_LOADED_MODELS=1"
set "OLLAMA_KEEP_ALIVE=%QWEN_KEEP_ALIVE%"
set "OLLAMA_LOAD_TIMEOUT=%QWEN_LOAD_TIMEOUT%"
set "OLLAMA_IGPU_ENABLE=0"

REM Keep local Ollama traffic away from inherited HTTP proxies.
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "NO_PROXY=localhost,127.0.0.1,::1,%QWEN_BIND_IP%,%QWEN_CLIENT_IP%"

REM ---------- Existing Ollama process check ----------
REM A running tray app can launch its own server with old settings.
tasklist /FI "IMAGENAME eq ollama app.exe" /NH 2>nul | findstr /I /C:"ollama app.exe" >nul
if not errorlevel 1 goto already_running

REM Check requested bind/port without stopping anything.
powershell.exe -NoLogo -NoProfile -Command ^
  "$listener = $null; " ^
  "try { " ^
  "  $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Parse($env:QWEN_BIND_IP),11434); " ^
  "  $listener.Server.ExclusiveAddressUse = $true; " ^
  "  $listener.Start(); " ^
  "  exit 0 " ^
  "} catch { " ^
  "  Write-Host ('Cannot bind ' + $env:QWEN_BIND_IP + ':11434 : ' + $_.Exception.Message); " ^
  "  exit 1 " ^
  "} finally { " ^
  "  if ($null -ne $listener) { $listener.Stop() } " ^
  "}"
if errorlevel 1 goto already_running

REM ---------- Display configuration ----------
echo.
echo ============================================================
echo Qwen3.8 Ollama Server
echo ============================================================
echo Source model:       %QWEN_SOURCE%
echo Agent model name:   %QWEN_MODEL%
echo Context tokens:     %QWEN_CONTEXT%
echo KV cache:           %QWEN_CACHE%
echo Parallel requests:  1
echo Loaded models:      1 maximum
echo Idle retention:     %QWEN_KEEP_ALIVE%
echo Load timeout:       %QWEN_LOAD_TIMEOUT%
echo Server bind:        %QWEN_BIND_IP%:11434
echo Local client API:   %QWEN_LOCAL_API%
echo Intel iGPU:         disabled for Ollama
echo ============================================================
echo.

REM ---------- Start server ----------
echo Starting Ollama server in a separate window...
start "Ollama server - Qwen3.8" "%QWEN_OLLAMA_EXE%" serve
if errorlevel 1 goto failed

REM The server inherited OLLAMA_HOST above.
REM The current batch uses a connectable address for client operations.
set "OLLAMA_HOST=%QWEN_CLIENT_IP%:11434"

REM ---------- Wait for server ----------
echo Waiting for Ollama API to become ready...
powershell.exe -NoLogo -NoProfile -Command ^
  "$deadline = [DateTime]::UtcNow.AddSeconds(60); " ^
  "do { " ^
  "  try { " ^
  "    $v = Invoke-RestMethod -Uri ($env:QWEN_LOCAL_API + '/api/version') -TimeoutSec 3 -ErrorAction Stop; " ^
  "    if ($v.version) { Write-Host ('Ollama server version: ' + $v.version); exit 0 } " ^
  "  } catch {}; " ^
  "  Start-Sleep -Milliseconds 500 " ^
  "} while ([DateTime]::UtcNow -lt $deadline); " ^
  "Write-Host 'Timed out waiting for Ollama.'; exit 1"
if errorlevel 1 goto server_failed

REM ---------- Verify source model exists ----------
echo.
echo Checking installed source model...
"%QWEN_OLLAMA_EXE%" show "%QWEN_SOURCE%"
if errorlevel 1 goto missing_model

REM ---------- Create local context preset ----------
REM FROM reuses the installed model blobs; it does not duplicate model weights.
set "QWEN_MODELFILE=%TEMP%\qwen3.8-local-%RANDOM%-%RANDOM%.Modelfile"
>"%QWEN_MODELFILE%" (
    echo FROM %QWEN_SOURCE%
    echo PARAMETER num_ctx %QWEN_CONTEXT%
)
if errorlevel 1 goto failed

echo.
echo Creating or refreshing local preset: %QWEN_MODEL%
"%QWEN_OLLAMA_EXE%" create "%QWEN_MODEL%" -f "%QWEN_MODELFILE%"
if errorlevel 1 goto create_failed

del /q "%QWEN_MODELFILE%" >nul 2>&1
set "QWEN_MODELFILE="

REM ---------- Warm-load the model ----------
echo.
echo Loading Qwen into memory...
echo This can take a while because the model may use both GPU and CPU/RAM.
"%QWEN_OLLAMA_EXE%" run "%QWEN_MODEL%" "Reply with only: OK"
if errorlevel 1 goto load_failed

REM ---------- Verify placement ----------
echo.
echo ============================================================
echo Current model placement:
echo ============================================================
"%QWEN_OLLAMA_EXE%" ps
echo.
echo PROCESSOR shows where the model is loaded.
echo   100%% GPU  = fully loaded in GPU VRAM.
echo   100%% CPU  = fully loaded in system RAM.
echo   CPU + GPU  = hybrid CPU/GPU offloading.
echo.
echo Recommended starting point for your 8 GB RTX 4060:
echo   Context = 65536
echo   KV cache = q4_0
echo.
echo Larger contexts such as 100000 or 131072 use substantially more memory.
echo.

REM ---------- Endpoint information ----------
if "%QWEN_BIND_IP%"=="127.0.0.1" goto local_endpoint
if "%QWEN_BIND_IP%"=="0.0.0.0" goto wildcard_endpoint

echo Server endpoint for clients:
echo   http://%QWEN_BIND_IP%:11434
echo.
echo If the coding agent runs on another PC, use the GPU machine address above.
echo Make sure Windows Firewall permits TCP 11434 from that client PC.
goto endpoint_done

:local_endpoint
echo Server endpoint:
echo   %QWEN_LOCAL_API%
echo.
echo This is LOCAL ONLY. Another PC cannot connect to 127.0.0.1.
goto endpoint_done

:wildcard_endpoint
echo Server is listening on all IPv4 interfaces.
echo Use the GPU machine's LAN IPv4 address from the list below:
powershell.exe -NoLogo -NoProfile -Command ^
  "try { " ^
  "  $ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop | " ^
  "    Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.AddressState -eq 'Preferred' }); " ^
  "  if ($ips.Count -eq 0) { Write-Host 'Use ipconfig to find this machine''s LAN IPv4 address.' }; " ^
  "  foreach ($ip in $ips) { Write-Host ('  ' + $ip.InterfaceAlias + ': http://' + $ip.IPAddress + ':11434') } " ^
  "} catch { Write-Host 'Use ipconfig to find this machine''s LAN IPv4 address.' }"
echo.
echo Restrict firewall access to the coding PC if you use 0.0.0.0.
goto endpoint_done

:endpoint_done
echo.
echo Test the local API on this GPU machine:
echo   curl.exe %QWEN_LOCAL_API%/api/tags
echo.
echo OpenAI-compatible endpoint:
echo   %QWEN_LOCAL_API%/v1
echo.
echo Model name for your coding agent:
echo   %QWEN_MODEL%
echo.
echo Keep the separate Ollama server window open while using the agent.
echo To stop the server, press Ctrl+C in that server window.
echo.
echo To release model memory from another CMD window on this machine:
echo   set "OLLAMA_HOST=%QWEN_CLIENT_IP%:11434"
echo   ollama stop %QWEN_MODEL%
echo.
echo ============================================================
echo Qwen3.8 is ready.
echo ============================================================
pause >nul
exit /b 0

:already_running
echo.
echo Ollama is already running, port 11434 is occupied, or the bind IP is unavailable.
echo New environment settings require a fresh Ollama server process.
echo.
echo 1. Quit Ollama from the Windows system tray.
echo 2. Close any existing "ollama serve" window.
echo 3. Run this batch file again.
echo.
echo This script intentionally does not stop existing processes automatically.
goto failed

:missing_model
echo.
echo The local source model could not be opened:
echo   %QWEN_SOURCE%
echo.
echo No download was requested by this script.
echo Install or transfer the model first, then run this batch again.
echo.
echo Models currently visible to this server:
"%QWEN_OLLAMA_EXE%" list
goto failed

:no_ollama
echo.
echo Ollama was not found.
echo Install Ollama for Windows, then run this batch file again.
goto failed

:no_powershell
echo.
echo Windows PowerShell was not found.
echo This launcher requires Windows PowerShell 5.1.
goto failed

:server_failed
echo.
echo Ollama did not become ready within 60 seconds.
echo Check the separate Ollama server window for the actual error.
goto failed

:create_failed
echo.
echo The local preset could not be created.
echo Check the Ollama error printed above.
goto failed

:load_failed
echo.
echo The model failed to load or run.
echo.
echo For a memory error, close the server and retry with:
echo   run_qwen3_8_24gb_fixed.bat 32768 qwen3.8:27b 127.0.0.1
echo or:
echo   run_qwen3_8_24gb_fixed.bat 65536 qwen3.8:27b 127.0.0.1
echo.
echo Then check the placement with: ollama ps
goto failed

:failed
if defined QWEN_MODELFILE del /q "%QWEN_MODELFILE%" >nul 2>&1
echo.
echo Launcher exited with an error.
pause
exit /b 1

:help
echo.
echo Usage:
echo   run_qwen3_8_24gb_fixed.bat [context] [installed-model] [bind-IPv4]
echo.
echo Context options:
echo   32768    32K
echo   65536    64K (default)
echo   100000   100K
echo   131072   128K
echo.
echo Default source model: qwen3.8:27b
echo Default preset name:  qwen3.8-27b-local:latest
echo Default bind:         127.0.0.1:11434 (local only)
echo Default KV cache:     q4_0
echo.
echo Examples:
echo   run_qwen3_8_24gb_fixed.bat
echo   run_qwen3_8_24gb_fixed.bat 65536 qwen3.8:27b 127.0.0.1
echo   run_qwen3_8_24gb_fixed.bat 65536 qwen3.8:27b localhost
echo   run_qwen3_8_24gb_fixed.bat 65536 qwen3.8:27b 192.168.1.10
echo   run_qwen3_8_24gb_fixed.bat 65536 qwen3.8:27b 0.0.0.0
echo.
echo Notes:
echo   127.0.0.1 = only this PC can connect.
echo   192.168.x.x = bind only to that LAN interface.
echo   0.0.0.0 = listen on all IPv4 interfaces.
echo.
echo Quit the existing Ollama tray app/server before launching this file.
echo The model must already be installed. This script never runs "ollama pull".
exit /b 0
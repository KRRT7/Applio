@echo off
setlocal
title Applio

if not exist .venv\Scripts\python.exe (
    echo Please run 'uv sync' first to set up the environment.
    pause
    exit /b 1
)

set HIP_VISIBLE_DEVICES="0"
set ZLUDA_COMGR_LOG_LEVEL=1
SET DISABLE_ADDMM_CUDA_LT=1
zluda\zluda.exe -- .venv\Scripts\python.exe app.py --open
echo.
pause
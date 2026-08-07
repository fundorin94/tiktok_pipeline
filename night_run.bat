@echo off
rem Overnight full pipeline run for a case. Survives Claude Code / terminal exit.
rem Usage: night_run.bat [case_id] [max_parts]   (default: test3, all parts)
set CASE=%1
if "%CASE%"=="" set CASE=test3
set PYTHONIOENCODING=utf-8
set PIPELINE_MAX_PARTS=%2
if "%PIPELINE_MAX_PARTS%"=="" set PIPELINE_MAX_PARTS=0
rem 3rd arg "fast" switches image generation to SD 1.5 (~10s/frame) for sync/script iteration
set PIPELINE_FAST_IMAGES=0
if "%3"=="fast" set PIPELINE_FAST_IMAGES=1
cd /d "%~dp0"
echo [%date% %time%] night run started for %CASE% > night_run.log
"venv\Scripts\python.exe" -u run_pipeline.py --case-id %CASE% --stage script >> night_run.log 2>&1
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -u run_pipeline.py --case-id %CASE% --stage archive >> night_run.log 2>&1
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -u run_pipeline.py --case-id %CASE% --stage voiceover >> night_run.log 2>&1
if errorlevel 1 goto :fail
"venv\Scripts\python.exe" -u run_pipeline.py --case-id %CASE% --stage video >> night_run.log 2>&1
if errorlevel 1 goto :fail
rem Titles, captions and hashtags -- without this the run leaves stale metadata
rem from a previous script behind, which then goes out with the videos.
"venv\Scripts\python.exe" -u run_pipeline.py --case-id %CASE% --stage metadata >> night_run.log 2>&1
if errorlevel 1 goto :fail
echo [%date% %time%] NIGHT RUN COMPLETE >> night_run.log
exit /b 0
:fail
echo [%date% %time%] NIGHT RUN FAILED (see above) >> night_run.log
exit /b 1

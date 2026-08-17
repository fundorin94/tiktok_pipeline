@echo off
rem Overnight full pipeline run for a case. Survives Claude Code / terminal exit.
rem Usage: night_run.bat [case_id] [max_parts] [fast] [animate]
rem   e.g. night_run.bat test3 0 fast animate  -- all parts, SD 1.5 stills, motion
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
rem Animate the generated stills before assembly. 4th arg "animate" turns it
rem on: it is the slowest step by far (~7 min per shot), so it stays opt-in.
rem Already-generated clips are skipped, so a re-run resumes rather than redoes.
rem Scene openers only by default: animating every frame of a 6-part case is
rem ~160 clips, about 19 hours, which no overnight run finishes. Pass
rem "animate-all" to do the lot anyway.
if /i "%4"=="animate" (
  "venv\Scripts\python.exe" -u tools\animate_frames.py --case-id %CASE% --max-parts %PIPELINE_MAX_PARTS% --per-part 8 >> night_run.log 2>&1
  if errorlevel 1 goto :fail
)
if /i "%4"=="animate-all" (
  "venv\Scripts\python.exe" -u tools\animate_frames.py --case-id %CASE% --max-parts %PIPELINE_MAX_PARTS% --all-frames >> night_run.log 2>&1
  if errorlevel 1 goto :fail
)
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

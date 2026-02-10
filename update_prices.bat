@echo off
REM ========================================================================
REM Daily Automated Updates - Prices + Vol/Skew
REM Runs at 4 AM via Windows Task Scheduler
REM ========================================================================

echo.
echo ========================================================================
echo DAILY DATA UPDATE
echo Started: %date% %time%
echo ========================================================================
echo.

REM Navigate to dashboard folder
cd C:\Users\AdamKelleher\ags_book_streamlit

REM Wait for OneDrive to sync (if needed)
echo Waiting 60 seconds for OneDrive sync...
timeout /t 60 /nobreak >nul

REM ========================================================================
REM STEP 1: Update Prices from HertzDR.xlsm
REM ========================================================================
echo.
echo [STEP 1/2] Updating futures prices...
echo ========================================================================

REM Check if HertzDR.xlsm exists
if not exist "C:\Users\AdamKelleher\OneDrive - Prime Trading\DR files\HertzDR.xlsm" (
    echo ERROR: HertzDR.xlsm not found!
    echo Check OneDrive sync status
    echo %date% %time% - ERROR: HertzDR.xlsm not found >> update_errors.log
    goto vol_update
)

REM Run price update
python update_from_hertz.py

if %errorlevel% equ 0 (
    echo SUCCESS: Price update completed
    echo %date% %time% - Price update SUCCESS >> update_log.txt
) else (
    echo ERROR: Price update failed
    echo %date% %time% - Price update FAILED >> update_errors.log
)

REM ========================================================================
REM STEP 2: Update Vol/Skew Data
REM ========================================================================
:vol_update
echo.
echo [STEP 2/2] Updating vol/skew data...
echo ========================================================================

REM Check if end of day snapshot exists
if not exist "eod_vol_snap.csv" (
    echo WARNING: eod_vol_snap.csv not found - skipping vol/skew update
    echo %date% %time% - WARNING: eod_vol_snap.csv not found >> update_log.txt
    goto end
)

REM Run vol/skew update
python update_volskew_clean.py eod_vol_snap.csv

if %errorlevel% equ 0 (
    echo SUCCESS: Vol/skew update completed
    echo %date% %time% - Vol/skew update SUCCESS >> update_log.txt
    
    REM Rename processed file (so we know it was used)
    ren eod_vol_snap.csv eod_vol_snap_%date:~-4,4%%date:~-10,2%%date:~-7,2%.csv
    echo Renamed to: eod_vol_snap_%date:~-4,4%%date:~-10,2%%date:~-7,2%.csv
) else (
    echo ERROR: Vol/skew update failed
    echo %date% %time% - Vol/skew update FAILED >> update_errors.log
)

REM ========================================================================
REM Completion
REM ========================================================================
:end
echo.
echo ========================================================================
echo DAILY UPDATE COMPLETED
echo Finished: %date% %time%
echo ========================================================================
echo.
echo Check update_log.txt for details
echo Check update_errors.log if any errors occurred
echo.

REM Keep window open if run manually (not from Task Scheduler)
if "%1"=="" pause
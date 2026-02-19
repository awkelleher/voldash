@echo off
REM ========================================================================
REM 30-Day Cache Refresh - IV Percentiles + Skew Percentiles
REM Regenerates: cache/iv_percentiles/ (parquet snapshots + timeseries)
REM              cache/median_iv.csv
REM              cache/median_skew.csv
REM              cache/iv_percentile_dist.csv
REM              cache/skew_percentile_dist.csv
REM
REM Schedule via Windows Task Scheduler: run every 30 days
REM ========================================================================

echo.
echo ========================================================================
echo 30-DAY CACHE REFRESH
echo Started: %date% %time%
echo ========================================================================
echo.

REM Navigate to dashboard folder
cd C:\Users\AdamKelleher\ags_book_streamlit

REM ========================================================================
REM STEP 1: Force recompute all IV + skew percentiles
REM ========================================================================
echo.
echo [STEP 1/1] Recomputing IV + skew percentiles (forced)...
echo ========================================================================

python scripts\iv_percentiles_precompute.py --force

if %errorlevel% equ 0 (
    echo SUCCESS: Cache refresh completed
    echo %date% %time% - 30-day cache refresh SUCCESS >> logs\update_log.txt
) else (
    echo ERROR: Cache refresh failed
    echo %date% %time% - 30-day cache refresh FAILED >> logs\update_errors.log
)

REM ========================================================================
REM Completion
REM ========================================================================
echo.
echo ========================================================================
echo 30-DAY CACHE REFRESH COMPLETED
echo Finished: %date% %time%
echo ========================================================================
echo.
echo Files updated:
echo   cache\iv_percentiles\snapshots\*.parquet
echo   cache\iv_percentiles\timeseries\*.parquet
echo   cache\iv_percentiles\metadata\*.json
echo   cache\median_iv.csv
echo   cache\median_skew.csv
echo   cache\iv_percentile_dist.csv
echo   cache\skew_percentile_dist.csv
echo.

REM Keep window open if run manually (not from Task Scheduler)
if "%1"=="" pause

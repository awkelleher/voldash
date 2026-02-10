# Vol Dashboard - Quick Reference Guide

## Updating Current Market Vols (Intraday)

**Use Case:** You copy/paste current IVs from trading software into Excel throughout the day, and want to update the dashboard without doing a full data save.

### Method 1: Quick Manual Entry (Fastest - 30 seconds)

```bash
python update_live_vols.py
```

Then enter current IVs:
```
SOY front month IV (%): 12.5
MEAL front month IV (%): 16.8
OIL front month IV (%): 24.3
CORN front month IV (%): 14.2
WHEAT front month IV (%): 22.5
```

Press Enter to skip any commodity you don't want to update.

**What it does:**
- Creates/updates `live_vols.json` file
- Dashboard shows 🔴 LIVE DATA indicator
- Overrides historical data for front month only

**Refresh dashboard:**
- Press `R` in browser
- Or wait 10 seconds for auto-refresh

**Clear live data:**
- Click "Clear Live Data" button in dashboard sidebar
- Or delete `live_vols.json` file
- Dashboard reverts to historical data

### Method 2: Direct JSON Edit (If Comfortable)

Create/edit `live_vols.json` in your dashboard folder:
```json
{
  "timestamp": "2026-01-30T14:30:00",
  "vols": {
    "SOY": 12.5,
    "MEAL": 16.8,
    "OIL": 24.3,
    "CORN": 14.2,
    "WHEAT": 22.5
  }
}
```

Save → Refresh dashboard (press R)

---

## Pushing Project to GitHub

### First Time Setup (One Time Only)

**1. Create GitHub repository:**
- Go to https://github.com
- Click "+" → "New repository"
- Name it: `vol-dashboard` (or whatever you want)
- Make it Private
- Don't initialize with README (you already have one)
- Click "Create repository"

**2. Link your local folder to GitHub:**

```bash
# Navigate to your dashboard folder
cd C:\path\to\vol-dashboard

# Initialize git (if not already done)
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial vol dashboard commit"

# Connect to GitHub (replace with your repo URL)
git remote add origin https://github.com/yourusername/vol-dashboard.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Daily Push Workflow (After Making Changes)

```bash
# Check what changed
git status

# Add all changes
git add .

# Commit with a message
git commit -m "Added variance ratios to dashboard"

# Push to GitHub
git push
```

### Working Remotely (From Another Computer)

**First time on new computer:**
```bash
git clone https://github.com/yourusername/vol-dashboard.git
cd vol-dashboard
pip install -r requirements.txt
```

**Get latest changes:**
```bash
git pull
```

**After making changes remotely:**
```bash
git add .
git commit -m "Fixed bug in price analysis"
git push
```

**Back at original computer:**
```bash
git pull  # Get changes from remote work
```

### Important Notes:

- `.gitignore` file prevents data files (CSV) from being uploaded
- Only code (.py files) and docs (.md files) go to GitHub
- Data files stay on local machines
- If you get merge conflicts, GitHub will tell you - easy to fix

---

## Updating Futures Price Information

**Use Case:** New overnight prices are available in HertzDR.xlsm file each morning.

### Daily Morning Routine

**1. Ensure HertzDR.xlsm has updated:**
- File location: `C:\Users\AdamKelleher\OneDrive - Prime Trading\DR files\HertzDR.xlsm`
- Should have yesterday's closing prices
- Make sure Excel is CLOSED (file can't be read if Excel has it open)

**2. Run update script:**
```bash
python update_from_hertz.py
```

**What it does:**
- Reads all 6 commodity sheets from HertzDR.xlsm
- Parses OHLC data for all contract months
- Handles both datetime and Excel serial number formats (1970 bug fixed)
- Adds new data to `all_commodity_prices.csv`
- Removes duplicates (keeps newest data)
- Creates automatic backup: `all_commodity_prices_backup_YYYYMMDD.csv`

**Expected output:**
```
======================================================================
UPDATE PRICES FROM HERTZDR.XLSM
======================================================================

Reading from: C:/Users/AdamKelleher/OneDrive - Prime Trading/DR files/HertzDR.xlsm
  ✓ File opened
  Parsing SOY...
    ✓ 62,376 records
  Parsing MEAL...
    ✓ 62,349 records
  [etc...]

✓ Total new data: 362,824 records
  Date range: 2007-01-03 to 2026-01-30

  ✓ New dates added: 2026-01-30

✓ Updated data saved to all_commodity_prices.csv

======================================================================
✅ PRICE UPDATE COMPLETE
======================================================================
```

**3. Refresh dashboard:**
- If dashboard is running: Press `R` in browser
- Or restart: `streamlit run vol_dashboard.py`

### Alternative: Custom File Path

If HertzDR.xlsm is in a different location:
```bash
python update_from_hertz.py "C:\Different\Path\HertzDR.xlsm"
```

### Troubleshooting:

**"File not found":**
- Check file path in script (line ~233)
- Update `default_path` if location changed
- Or provide path as argument (see above)

**"No data extracted":**
- Make sure Excel is closed
- Check that HertzDR.xlsm has data in the commodity sheets
- File might be corrupted - try re-opening in Excel

**"Permission denied":**
- Close Excel if it has the file open
- Check OneDrive sync isn't locking the file

### Data Files Generated:

- `all_commodity_prices.csv` - Master price file (~7MB, 19 years of data)
- `all_commodity_prices_backup_YYYYMMDD.csv` - Daily backup
- These files are NOT committed to GitHub (too large, in .gitignore)

---

## End of Day Full Vol/Skew Update

**Use Case:** After market close, you save your full vol/skew snapshot from Excel and want to add it to historical data.

### Process:

**1. In Excel: Export vol/skew tab to CSV**
- Save your daily snapshot
- Export to CSV (e.g., `volskew_20260130.csv`)

**2. Run update script:**
```bash
python update_data.py volskew_20260130.csv
```

**What it does:**
- Parses wide-format vol/skew data (all commodities side-by-side)
- Adds to `historical_vol_skew_all_commodities.csv`
- Creates automatic backup with timestamp
- Removes duplicates (keeps newest)

**3. Clear live data (if used during the day):**
- Click "Clear Live Data" in dashboard
- Or delete `live_vols.json`

---

## Summary: Typical Trading Day

**Morning (9:00 AM):**
```bash
python update_from_hertz.py          # Get overnight prices
streamlit run vol_dashboard.py        # Start dashboard
```

**During Market (9:30 AM - 2:30 PM):**
```bash
python update_live_vols.py           # Quick IV updates as market moves
# Enter current IVs
# Press R in dashboard
```

**End of Day (After 2:30 PM):**
```bash
python update_data.py today_volskew.csv    # Save full snapshot
# Clear live data in dashboard
```

---

## File Locations

**Code (in GitHub):**
- `vol_dashboard.py`, `vol_analysis.py`, etc.
- `requirements.txt`, `.gitignore`, `README.md`

**Data (NOT in GitHub, local only):**
- `historical_vol_skew_all_commodities.csv` (~12MB)
- `all_commodity_prices.csv` (~7MB)
- `continuous_prices_with_rv.csv` (generated)
- `variance_ratios_*.csv` (generated)
- `live_vols.json` (temporary)

**Source Data:**
- `C:\Users\AdamKelleher\OneDrive - Prime Trading\DR files\HertzDR.xlsm`

---

## Quick Commands Cheat Sheet

```bash
# Start dashboard
streamlit run vol_dashboard.py

# Update prices
python update_from_hertz.py

# Quick IV update
python update_live_vols.py

# End of day vol/skew save
python update_data.py today_snapshot.csv

# Push to GitHub
git add .
git commit -m "Your message"
git push

# Get updates from GitHub
git pull

# Check git status
git status
```

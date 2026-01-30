python update_data.py today_data.csv
```

**What it does:**
- ✅ Parses the new CSV
- ✅ Adds it to your historical dataset
- ✅ Removes duplicates (keeps newest data)
- ✅ Creates automatic backup with timestamp
- ✅ Saves updated dataset

**4. Refresh dashboard:**
   - If dashboard is already running, press `R` in browser
   - Or just restart: `streamlit run vol_dashboard.py`

---

## Data Flow Summary
```
Exegy → Excel → Export CSV → update_data.py → historical CSV → Dashboard
"""
Intraday Vol Update - Simple Manual Entry

For quick intraday updates when you just need to refresh current market IVs
without doing a full data save.

This creates a small temporary file that the dashboard reads for "live" data.
"""

import pandas as pd
from datetime import datetime
import json


def update_live_vols():
    """Interactive prompt to update current market vols"""
    
    print("="*70)
    print("INTRADAY VOL UPDATE")
    print("="*70)
    print("Enter current market IVs (press Enter to skip a commodity)\n")
    
    commodities = ['SOY', 'MEAL', 'OIL', 'CORN', 'WHEAT']
    live_data = {
        'timestamp': datetime.now().isoformat(),
        'vols': {}
    }
    
    for commodity in commodities:
        try:
            vol_str = input(f"{commodity} front month IV (%): ").strip()
            if vol_str:
                vol = float(vol_str)
                live_data['vols'][commodity] = vol
                print(f"  ✓ {commodity}: {vol:.2f}%")
        except ValueError:
            print(f"  ⚠️  Invalid input, skipping {commodity}")
    
    if len(live_data['vols']) > 0:
        # Save to JSON file
        with open('data/live_vols.json', 'w') as f:
            json.dump(live_data, f, indent=2)
        
        print(f"\n✓ Saved {len(live_data['vols'])} live vols to data/live_vols.json")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nRefresh your dashboard to see updates!")
    else:
        print("\n❌ No data entered")
    
    print("="*70)


if __name__ == "__main__":
    update_live_vols()

#!/usr/bin/env python3
"""Example: Batch process multiple XES files."""

import sys
sys.path.insert(0, '..')

from pathlib import Path
from xestools.modules import i20_loader
from xestools.modules.scan import Scan
from xestools.modules.cli_export import scan_to_dataframe
import pandas as pd

# Find all .nxs files
files = list(Path('../xestools/data/vtc').glob('*.nxs'))

# Process each
all_data = []
for f in files:
    scan = Scan()
    try:
        snum = i20_loader.add_scan_from_nxs(scan, str(f))
        df = scan_to_dataframe(scan, snum, channel='upper')
        df['filename'] = f.name
        all_data.append(df)
        print(f"✓ Processed {f.name}: {len(df)} points")
    except Exception as e:
        print(f"✗ Failed {f.name}: {e}")

# Combine
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv('all_xes_data.csv', index=False)
    print(f"\n✅ Processed {len(files)} files → {len(combined)} total points")
    print(f"   Saved to: all_xes_data.csv")
else:
    print("No files processed successfully")

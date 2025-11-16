#!/usr/bin/env python3
"""Simple example: Export RXES to CSV."""

import sys
sys.path.insert(0, '..')

from xestools.modules import i20_loader
from xestools.modules.scan import Scan
from xestools.modules.cli_export import scan_to_dataframe

# Load RXES scan
scan = Scan()
snum = i20_loader.add_scan_from_nxs(scan, '../xestools/data/rxes/279517_1.nxs')

# Export to DataFrame
df = scan_to_dataframe(scan, snum, channel='upper')

# Save to CSV
df.to_csv('rxes_export.csv', index=False)
print(f"Exported {len(df)} data points to rxes_export.csv")
print(f"Columns: {list(df.columns)}")

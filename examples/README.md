# CLI Export Examples

This directory contains example scripts showing how to use the CLI export functionality.

## Requirements

```bash
pip install pandas  # For DataFrame export
pip install xarray  # For xarray/NetCDF export (optional)
```

## Examples

- `export_to_csv.py` - Basic CSV export
- `batch_analysis.py` - Process multiple files

## Usage

```bash
cd examples/
python3 export_to_csv.py
python3 batch_analysis.py
```

## Quick Start

```python
from i20_xes.modules import i20_loader
from i20_xes.modules.scan import Scan
from i20_xes.modules.cli_export import scan_to_dataframe

# Load data
scan = Scan()
snum = i20_loader.add_scan_from_nxs(scan, 'scan.nxs')

# Export to pandas DataFrame
df = scan_to_dataframe(scan, snum, channel='upper')

# Now use full pandas capabilities
df.to_csv('data.csv')
df.groupby('bragg')['intensity'].mean()
```

## Export Formats

- **CSV**: Simple, widely compatible
- **Parquet**: Efficient binary format
- **HDF5**: Full data preservation with metadata
- **NetCDF**: Scientific standard (requires xarray)

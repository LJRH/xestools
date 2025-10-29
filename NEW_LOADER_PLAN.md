# I20 Loader Refactoring Plan
**Date**: October 29, 2025  
**Purpose**: Unified ASCII and NeXus loader architecture with shared grid processing

---

## 📊 DATA FLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                       │
├───────────────────────────────────┬─────────────────────────────────────────┤
│   NeXus (.nxs)                   │         ASCII (.dat)                     │
│   ┌─────────────────────┐        │   ┌──────────────────────────┐          │
│   │ HDF5 2D arrays:     │        │   │ Flat text data:          │          │
│   │ - bragg1WithOffset  │        │   │ Energy    XESEnergy  ... │          │
│   │ - XESEnergyUpper    │        │   │ 9658.0    8632.0     ... │          │
│   │ - FFI1_medipix1     │        │   │ 9658.0    8632.3     ... │          │
│   │ - I1 (monitor)      │        │   │ 9658.0    8632.6     ... │          │
│   │                     │        │   │ ...                      │          │
│   │ Shape: (17, 51)     │        │   │ 867 rows × 7 columns     │          │
│   │ Already 2D meshes   │        │   │ Needs reshaping          │          │
│   └──────────┬──────────┘        │   └────────────┬─────────────┘          │
│              │                    │                │                        │
└──────────────┼────────────────────┴────────────────┼────────────────────────┘
               │                                     │
               ▼                                     ▼
      ┌─────────────────┐               ┌──────────────────────────┐
      │ Extract arrays  │               │ parse_i20_ascii_metadata │
      │ from HDF5       │               │ ┌──────────────────────┐ │
      │                 │               │ │ Parse header lines   │ │
      │ bragg_2d        │               │ │ Extract command      │ │
      │ emission_2d     │               │ │ Detect scan type     │ │
      │ intensity_2d    │               │ │ Find columns         │ │
      │ monitor_2d      │               │ └──────────────────────┘ │
      └────────┬────────┘               │                          │
               │                        │ load_i20_ascii_data      │
               │                        │ ┌──────────────────────┐ │
               │                        │ │ np.genfromtxt()      │ │
               │                        │ │ Skip comments        │ │
               │                        │ │ Parse tab-delimited  │ │
               │                        │ └──────────────────────┘ │
               │                        └───────────┬──────────────┘
               │                                    │
               │                        Extract columns:
               │                        bragg (1D), emission (1D),
               │                        intensity (1D), monitor (1D)
               │                                    │
               └────────────┬───────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────────────────────┐
          │         SHARED PROCESSING PIPELINE                  │
          │                                                     │
          │  ┌───────────────────────────────────────────────┐ │
          │  │ 1. validate_rxes_data()                       │ │
          │  │    - Check shape compatibility                │ │
          │  │    - Verify finite data fraction > 50%        │ │
          │  │    - Calculate data ranges                    │ │
          │  │    - Return validation metadata               │ │
          │  └───────────────────────────────────────────────┘ │
          │                      ▼                              │
          │  ┌───────────────────────────────────────────────┐ │
          │  │ 2. analyze_grid_structure()                   │ │
          │  │    - Find unique axis values                  │ │
          │  │    - Auto-detect outer/inner axes             │ │
          │  │    - Count repetition patterns                │ │
          │  │    - Check grid regularity                    │ │
          │  │    - Calculate completeness                   │ │
          │  └───────────────────────────────────────────────┘ │
          │                      ▼                              │
          │  ┌───────────────────────────────────────────────┐ │
          │  │ 3. normalize_grid_to_2d()                     │ │
          │  │                                               │ │
          │  │    IF 2D (NeXus):                             │ │
          │  │    ├─ Use reduce_axes_for() logic             │ │
          │  │    ├─ Transpose if needed                     │ │
          │  │    └─ Ensure: emission on rows, bragg on cols │ │
          │  │                                               │ │
          │  │    IF 1D (ASCII):                             │ │
          │  │    ├─ Reshape based on grid_info              │ │
          │  │    ├─ Handle outer/inner axis order           │ │
          │  │    └─ Transpose to standard orientation       │ │
          │  │                                               │ │
          │  │    Output: All arrays shape (n_emission, n_bragg) │
          │  └───────────────────────────────────────────────┘ │
          │                      ▼                              │
          │  ┌───────────────────────────────────────────────┐ │
          │  │ 4. create_rxes_scan_entry()                   │ │
          │  │    - Build standardized dict                  │ │
          │  │    - Add channel-specific keys                │ │
          │  │    - Include metadata                         │ │
          │  └───────────────────────────────────────────────┘ │
          │                                                     │
          └─────────────────────┬───────────────────────────────┘
                                ▼
                  ┌──────────────────────────┐
                  │  Scan Container          │
                  │  (Standardized Format)   │
                  │                          │
                  │  scan[number] = {        │
                  │    'path': str,          │
                  │    'braggOffset_2d': arr,│
                  │    'energy_upper_2d': arr│
                  │    'intensity_upper': arr│
                  │    'I1': arr,            │
                  │    'averaged': False,    │
                  │    'normalised': False   │
                  │  }                       │
                  └──────────────────────────┘
```

---

## 🎯 IMPLEMENTATION TASKS

### **Phase 1: Shared Core Functions** (Priority: HIGH)

#### Task 1.1: `validate_rxes_data()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `reduce_axes_for()` (after line ~72)

```python
def validate_rxes_data(
    bragg: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
) -> dict:
    """
    Validate RXES data arrays and return metadata.
    
    Works with both 1D (flattened) and 2D (pre-shaped) arrays.
    
    Args:
        bragg: Incident energy (Ω) - 1D or 2D
        emission: Emission energy (ω) - 1D or 2D
        intensity: Detector intensity - 1D or 2D
        monitor: Optional I1 monitor - 1D or 2D
    
    Returns:
        dict with keys: is_1d, shape, n_points, bragg_range, 
                        emission_range, has_finite_data, finite_fraction
    
    Raises:
        ValueError: If arrays incompatible or insufficient data
    """
    bragg = np.asarray(bragg)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    
    # Check shape compatibility
    if bragg.shape != emission.shape or bragg.shape != intensity.shape:
        raise ValueError(
            f"Shape mismatch: bragg {bragg.shape}, "
            f"emission {emission.shape}, intensity {intensity.shape}"
        )
    
    is_1d = (bragg.ndim == 1)
    shape = bragg.shape
    n_points = bragg.size
    
    # Check for finite data
    finite_mask = np.isfinite(bragg) & np.isfinite(emission) & np.isfinite(intensity)
    finite_fraction = np.sum(finite_mask) / n_points if n_points > 0 else 0.0
    
    if finite_fraction < 0.5:
        raise ValueError(
            f"Insufficient finite data: only {finite_fraction*100:.1f}% valid"
        )
    
    # Get data ranges
    bragg_finite = bragg[finite_mask]
    emission_finite = emission[finite_mask]
    
    return {
        'is_1d': is_1d,
        'shape': shape,
        'n_points': n_points,
        'bragg_range': (float(np.min(bragg_finite)), float(np.max(bragg_finite))),
        'emission_range': (float(np.min(emission_finite)), float(np.max(emission_finite))),
        'has_finite_data': finite_fraction > 0,
        'finite_fraction': finite_fraction,
    }
```

**Tests to add**:
- Test with 1D arrays (ASCII case)
- Test with 2D arrays (NeXus case)
- Test with mismatched shapes (should raise)
- Test with mostly NaN data (should raise)

---

#### Task 1.2: `analyze_grid_structure()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `validate_rxes_data()`

```python
def analyze_grid_structure(
    bragg: np.ndarray,
    emission: np.ndarray,
    outer_axis: Optional[str] = None,
    precision: float = 0.01,
) -> dict:
    """
    Analyze 2D grid structure of RXES data.
    
    Auto-detects outer/inner axis from repetition patterns.
    
    Args:
        bragg: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        outer_axis: 'bragg' or 'emission' or None (auto-detect)
        precision: Rounding precision in eV for unique values
    
    Returns:
        dict with keys: outer_axis, inner_axis, n_outer, n_inner,
                        outer_values, inner_values, outer_counts,
                        is_regular_grid, grid_completeness, needs_reshaping
    """
    bragg = np.asarray(bragg).ravel()
    emission = np.asarray(emission).ravel()
    
    # Round to precision
    bragg_rounded = np.round(bragg / precision) * precision
    emission_rounded = np.round(emission / precision) * precision
    
    # Get unique values
    bragg_unique = np.unique(bragg_rounded[np.isfinite(bragg_rounded)])
    emission_unique = np.unique(emission_rounded[np.isfinite(emission_rounded)])
    
    # Auto-detect outer axis if not specified
    if outer_axis is None:
        def count_max_consecutive(arr, unique_vals):
            """Count max consecutive occurrences for first few values."""
            max_counts = []
            for val in unique_vals[:min(5, len(unique_vals))]:
                matches = (arr == val)
                if not matches.any():
                    continue
                # Find runs of consecutive True values
                diff = np.diff(np.concatenate([[False], matches, [False]]).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) > 0:
                    max_counts.append(np.max(ends - starts))
            return np.mean(max_counts) if max_counts else 0
        
        bragg_consec = count_max_consecutive(bragg_rounded, bragg_unique)
        emission_consec = count_max_consecutive(emission_rounded, emission_unique)
        
        # More consecutive repetitions = outer axis (stays constant)
        outer_axis = 'bragg' if bragg_consec > emission_consec else 'emission'
    
    # Set parameters based on outer axis
    if outer_axis == 'bragg':
        outer_values = bragg_unique
        inner_values = emission_unique
        outer_arr = bragg_rounded
    else:
        outer_values = emission_unique
        inner_values = bragg_unique
        outer_arr = emission_rounded
    
    n_outer = len(outer_values)
    n_inner = len(inner_values)
    
    # Check regularity by counting outer value occurrences
    outer_counts = [np.sum(outer_arr == val) for val in outer_values]
    is_regular_grid = (len(set(outer_counts)) == 1)
    
    # Grid completeness
    expected_points = n_outer * n_inner
    actual_points = len(bragg)
    grid_completeness = actual_points / expected_points if expected_points > 0 else 0.0
    
    return {
        'outer_axis': outer_axis,
        'inner_axis': 'emission' if outer_axis == 'bragg' else 'bragg',
        'n_outer': n_outer,
        'n_inner': n_inner,
        'outer_values': outer_values,
        'inner_values': inner_values,
        'outer_counts': outer_counts,
        'is_regular_grid': is_regular_grid,
        'grid_completeness': grid_completeness,
        'needs_reshaping': (actual_points == expected_points and is_regular_grid),
    }
```

**Tests to add**:
- Test with bragg outer (most common case)
- Test with emission outer (reversed case)
- Test auto-detection algorithm
- Test with irregular grids
- Test with sparse grids

---

#### Task 1.3: `normalize_grid_to_2d()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `analyze_grid_structure()`

```python
def normalize_grid_to_2d(
    bragg: np.ndarray,
    emission: np.ndarray,
    intensity: np.ndarray,
    monitor: Optional[np.ndarray] = None,
    grid_info: Optional[dict] = None,
) -> dict:
    """
    Normalize RXES data to standard 2D grid format.
    
    Standard output: emission on rows, bragg on columns
    Shape: (n_emission, n_bragg)
    
    Handles:
    - Already 2D arrays (NeXus) - validate and transpose if needed
    - 1D arrays (ASCII) - reshape to 2D
    
    Args:
        bragg: Incident energy (1D or 2D)
        emission: Emission energy (1D or 2D)
        intensity: Detector intensity (1D or 2D)
        monitor: Optional I1 monitor (1D or 2D)
        grid_info: Optional pre-computed from analyze_grid_structure()
    
    Returns:
        dict with keys: bragg_2d, emission_2d, intensity_2d, 
                        monitor_2d, method, warnings
    """
    bragg = np.asarray(bragg)
    emission = np.asarray(emission)
    intensity = np.asarray(intensity)
    if monitor is not None:
        monitor = np.asarray(monitor)
    
    warnings_list = []
    
    # Analyze grid if not provided
    if grid_info is None:
        grid_info = analyze_grid_structure(bragg, emission)
    
    # Case 1: Already 2D (from NeXus)
    if bragg.ndim == 2:
        # Use existing reduce_axes_for to determine orientation
        y_omega, x_Omega, transposed = reduce_axes_for(emission, bragg)
        
        if transposed:
            bragg_2d = bragg.T
            emission_2d = emission.T
            intensity_2d = intensity.T
            monitor_2d = monitor.T if monitor is not None else None
        else:
            bragg_2d = bragg
            emission_2d = emission
            intensity_2d = intensity
            monitor_2d = monitor
        
        method = 'direct'
    
    # Case 2: 1D arrays - need reshaping (ASCII)
    elif bragg.ndim == 1:
        if not grid_info['is_regular_grid']:
            warnings_list.append(
                f"Irregular grid: outer counts vary {set(grid_info['outer_counts'])}"
            )
        
        if grid_info['grid_completeness'] < 0.95:
            warnings_list.append(
                f"Sparse grid: {grid_info['grid_completeness']*100:.1f}% filled"
            )
        
        if not grid_info['needs_reshaping']:
            raise ValueError(
                "Cannot reshape incomplete grid. "
                f"Expected {grid_info['n_outer'] * grid_info['n_inner']} points, "
                f"got {len(bragg)}"
            )
        
        # Reshape based on detected structure
        n_outer = grid_info['n_outer']
        n_inner = grid_info['n_inner']
        shape_2d = (n_outer, n_inner)
        
        if grid_info['outer_axis'] == 'bragg':
            # bragg outer, emission inner
            # Reshape then transpose to standard format
            bragg_2d = bragg.reshape(shape_2d).T
            emission_2d = emission.reshape(shape_2d).T
            intensity_2d = intensity.reshape(shape_2d).T
            monitor_2d = monitor.reshape(shape_2d).T if monitor is not None else None
        else:
            # emission outer, bragg inner (already standard format)
            bragg_2d = bragg.reshape(shape_2d)
            emission_2d = emission.reshape(shape_2d)
            intensity_2d = intensity.reshape(shape_2d)
            monitor_2d = monitor.reshape(shape_2d) if monitor is not None else None
        
        method = 'reshape'
    
    else:
        raise ValueError(f"Unexpected array dimensions: {bragg.ndim}D")
    
    return {
        'bragg_2d': bragg_2d,
        'emission_2d': emission_2d,
        'intensity_2d': intensity_2d,
        'monitor_2d': monitor_2d,
        'method': method,
        'warnings': warnings_list,
    }
```

**Tests to add**:
- Test with 2D NeXus-style arrays
- Test with 1D ASCII-style arrays (bragg outer)
- Test with 1D ASCII-style arrays (emission outer)
- Test transpose logic correctness
- Test with incomplete grids (should raise)

---

#### Task 1.4: `create_rxes_scan_entry()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `normalize_grid_to_2d()`

```python
def create_rxes_scan_entry(
    grids: dict,
    channel: str,
    path: str,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Create standardized Scan entry dict for RXES data.
    
    Args:
        grids: Output from normalize_grid_to_2d()
        channel: 'upper' or 'lower'
        path: Source file path
        metadata: Optional additional metadata
    
    Returns:
        dict suitable for scan.add_scan()
    """
    use_upper = channel.lower().startswith('u')
    
    entry = {
        'path': path,
        'braggOffset_2d': grids['bragg_2d'],
        'averaged': False,
        'normalised': False,
    }
    
    # Add channel-specific data
    if use_upper:
        entry['energy_upper_2d'] = grids['emission_2d']
        entry['intensity_upper'] = grids['intensity_2d']
    else:
        entry['energy_lower_2d'] = grids['emission_2d']
        entry['intensity_lower'] = grids['intensity_2d']
    
    # Add monitor if available
    if grids['monitor_2d'] is not None:
        entry['I1'] = grids['monitor_2d']
    
    # Add optional metadata
    if metadata:
        entry.update(metadata)
    
    return entry
```

---

### **Phase 2: ASCII-Specific Functions** (Priority: HIGH)

#### Task 2.1: `parse_i20_ascii_metadata()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add in new section "# -------- ASCII Metadata Parsing --------"

```python
def parse_i20_ascii_metadata(path: str) -> dict:
    """
    Parse I20 beamline ASCII file metadata from header.
    
    Extracts:
    - Scan command to determine scan type (RXES vs XES)
    - Outer/inner axis from command structure
    - Column names from header line
    - Detector channel (upper/lower)
    
    Args:
        path: Path to .dat file
    
    Returns:
        dict with keys: command, scan_type, outer_axis, inner_axis,
                        columns, channel, detector, sample_name, date
    
    Raises:
        ValueError: If file format invalid or scan type unclear
    """
    import re
    
    metadata = {
        'command': None,
        'scan_type': None,
        'outer_axis': None,
        'inner_axis': None,
        'columns': [],
        'channel': None,
        'detector': None,
        'sample_name': None,
        'date': None,
    }
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Stop at data section
            if not line.startswith('#'):
                break
            
            # Remove leading '# '
            content = line[1:].strip()
            
            # Extract command
            if content.startswith('command:'):
                metadata['command'] = content[8:].strip()
            
            # Extract sample name
            elif content.startswith('Sample name:'):
                metadata['sample_name'] = content[12:].strip()
            
            # Extract date
            elif content.startswith('Instrument:') and 'Date:' in content:
                match = re.search(r'Date:\s*(.+)$', content)
                if match:
                    metadata['date'] = match.group(1).strip()
            
            # Column headers (tab-delimited, no leading #)
            elif 'Energy' in content and 'XESEnergy' in content:
                # Split on tabs or multiple spaces
                cols = re.split(r'\t+|\s{2,}', content)
                metadata['columns'] = [c.strip() for c in cols if c.strip()]
    
    # Analyze command to determine scan type
    if metadata['command']:
        cmd = metadata['command']
        
        # Check for bragg scan parameters
        has_bragg_scan = 'bragg1WithOffset' in cmd and \
                        bool(re.search(r'bragg1WithOffset\s+[\d.]+\s+[\d.]+\s+[\d.]+', cmd))
        
        # Check for emission scan parameters (nested scan notation)
        has_emission_scan = bool(re.search(r'XESEnergy(?:Upper|Lower)\s+\[Range', cmd))
        
        if has_bragg_scan and has_emission_scan:
            metadata['scan_type'] = 'RXES'
            
            # Determine outer axis from command structure
            # First scanned parameter in command is outer loop
            bragg_pos = cmd.find('bragg1WithOffset')
            emission_match = re.search(r'XESEnergy(?:Upper|Lower)', cmd)
            emission_pos = emission_match.start() if emission_match else float('inf')
            
            if bragg_pos < emission_pos:
                metadata['outer_axis'] = 'bragg'
                metadata['inner_axis'] = 'emission'
            else:
                metadata['outer_axis'] = 'emission'
                metadata['inner_axis'] = 'bragg'
        
        elif has_emission_scan:
            metadata['scan_type'] = 'XES'
        elif has_bragg_scan:
            metadata['scan_type'] = 'XES'  # Only bragg varies
        else:
            raise ValueError(f"Cannot determine scan type from command: {cmd}")
    
    # Detect channel from column names
    if metadata['columns']:
        if 'XESEnergyUpper' in metadata['columns']:
            metadata['channel'] = 'upper'
            if 'FFI1_medipix1' in metadata['columns']:
                metadata['detector'] = 'medipix1'
        elif 'XESEnergyLower' in metadata['columns']:
            metadata['channel'] = 'lower'
            if 'FFI1_medipix2' in metadata['columns']:
                metadata['detector'] = 'medipix2'
    
    # Validate required fields
    if not metadata['columns']:
        raise ValueError("Could not parse column headers from file")
    if not metadata['channel']:
        raise ValueError("Could not determine detector channel (upper/lower)")
    if not metadata['scan_type']:
        raise ValueError("Could not determine scan type (RXES/XES)")
    
    return metadata
```

**Tests to add**:
- Test with actual I20 RXES file
- Test channel detection (upper/lower)
- Test scan type detection (RXES/XES)
- Test outer/inner axis detection
- Test with malformed headers (should raise)

---

#### Task 2.2: `load_i20_ascii_data()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `parse_i20_ascii_metadata()`

```python
def load_i20_ascii_data(path: str, metadata: dict) -> np.ndarray:
    """
    Load numeric data from I20 ASCII file.
    
    Args:
        path: Path to .dat file
        metadata: Output from parse_i20_ascii_metadata()
    
    Returns:
        2D numpy array: shape (n_rows, n_columns)
    
    Raises:
        ValueError: If data loading fails
    """
    try:
        # Try tab-delimited first (I20 standard)
        data = np.genfromtxt(
            path,
            comments='#',
            delimiter='\t',
            dtype=float,
            invalid_raise=False,
        )
    except Exception as e:
        # Fallback to whitespace-delimited
        try:
            data = np.genfromtxt(
                path,
                comments='#',
                delimiter=None,
                dtype=float,
                invalid_raise=False,
            )
        except Exception as e2:
            raise ValueError(f"Failed to load data from {path}: {e2}")
    
    # Ensure 2D
    data = np.atleast_2d(data)
    
    # Validate number of columns
    if data.shape[1] != len(metadata['columns']):
        import warnings
        warnings.warn(
            f"Column count mismatch: expected {len(metadata['columns'])}, "
            f"got {data.shape[1]}"
        )
    
    return data
```

**Tests to add**:
- Test with tab-delimited file
- Test with space-delimited file
- Test with mixed delimiters
- Test column count validation

---

#### Task 2.3: `add_scan_from_i20_ascii()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Add after `load_i20_ascii_data()`

```python
def add_scan_from_i20_ascii(
    scan: Scan,
    path: str,
    scan_number: Optional[Any] = None,
) -> Any:
    """
    Load I20 beamline ASCII file into Scan container.
    
    Uses shared grid processing pipeline for both RXES and XES.
    
    Args:
        scan: Scan container
        path: Path to .dat file
        scan_number: Optional scan number (auto-assigned if None)
    
    Returns:
        scan_number used
    
    Raises:
        ValueError: If file invalid or processing fails
    """
    # Step 1: Parse metadata
    metadata = parse_i20_ascii_metadata(path)
    
    # Step 2: Load data
    data = load_i20_ascii_data(path, metadata)
    
    # Step 3: Extract columns
    bragg = data[:, 0]  # Energy column (bragg1WithOffset)
    
    # Find emission column
    emission_col_name = f"XESEnergy{metadata['channel'].capitalize()}"
    try:
        emission_col_idx = metadata['columns'].index(emission_col_name)
    except ValueError:
        raise ValueError(f"Column {emission_col_name} not found in {metadata['columns']}")
    emission = data[:, emission_col_idx]
    
    # Find intensity column
    intensity_col_name = f"FFI1_{metadata['detector']}"
    try:
        intensity_col_idx = metadata['columns'].index(intensity_col_name)
    except ValueError:
        raise ValueError(f"Column {intensity_col_name} not found in {metadata['columns']}")
    intensity = data[:, intensity_col_idx]
    
    # Find monitor column
    try:
        monitor_col_idx = metadata['columns'].index('I1')
        monitor = data[:, monitor_col_idx]
    except ValueError:
        monitor = None
    
    # Auto-assign scan number
    if scan_number is None:
        scan_number = scan.next_index()
    
    # Step 4: Process based on scan type
    if metadata['scan_type'] == 'XES':
        # Simple 1D - filter and store
        ok = np.isfinite(emission) & np.isfinite(intensity)
        scan.add_scan(scan_number, {
            'energy': emission[ok],
            'intensity': intensity[ok],
            'monitor': monitor[ok] if monitor is not None else None,
            'path': path,
            'scan_type': 'XES',
            'channel': metadata['channel'],
            'source': 'ascii',
        })
    
    elif metadata['scan_type'] == 'RXES':
        # Use shared pipeline
        
        # Validate
        validate_rxes_data(bragg, emission, intensity, monitor)
        
        # Analyze grid
        grid_info = analyze_grid_structure(
            bragg, emission,
            outer_axis=metadata.get('outer_axis')
        )
        
        # Normalize to 2D
        grids = normalize_grid_to_2d(
            bragg, emission, intensity, monitor,
            grid_info=grid_info
        )
        
        # Create entry
        entry = create_rxes_scan_entry(
            grids,
            channel=metadata['channel'],
            path=path,
            metadata={'source': 'ascii', **metadata}
        )
        
        scan.add_scan(scan_number, entry)
    
    else:
        raise ValueError(f"Unknown scan type: {metadata['scan_type']}")
    
    return scan_number
```

**Tests to add**:
- Test loading actual I20 RXES file
- Test loading actual I20 XES file
- Test scan number auto-assignment
- Compare output with NeXus version of same data

---

### **Phase 3: Refactor NeXus Loader** (Priority: MEDIUM)

#### Task 3.1: Update `add_scan_from_nxs()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Replace existing function (lines 75-137)

**Changes**:
1. Keep HDF5 extraction logic (lines 93-118)
2. After extraction, process each channel through shared pipeline:
   - Call `validate_rxes_data()`
   - Call `analyze_grid_structure()`
   - Call `normalize_grid_to_2d()`
3. Build entry using processed grids
4. Store in Scan container

**See ASCII art diagram for flow**

**Key code changes**:
```python
# After extracting arrays from HDF5...

# Process upper channel if available
if energy_upper is not None and upper_int is not None:
    validate_rxes_data(bragg_off, energy_upper, upper_int, i1)
    grid_info = analyze_grid_structure(bragg_off, energy_upper)
    grids = normalize_grid_to_2d(bragg_off, energy_upper, upper_int, i1, grid_info)
    
    # Add to entry
    entry['energy_upper_2d'] = grids['emission_2d']
    entry['intensity_upper'] = grids['intensity_2d']
    entry['braggOffset_2d'] = grids['bragg_2d']
    entry['I1'] = grids['monitor_2d']

# Similar for lower channel...
```

---

### **Phase 4: Integration & Testing** (Priority: HIGH)

#### Task 4.1: Update `xes_from_ascii()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Replace existing function (lines 186-197)

**Changes**:
```python
def xes_from_ascii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 1D XES from ASCII file.
    
    Supports both:
    - I20 beamline format (.dat)
    - Simple 2-column format
    """
    # Try I20 format first
    try:
        metadata = parse_i20_ascii_metadata(path)
        if metadata['scan_type'] == 'XES':
            scan = Scan()
            snum = add_scan_from_i20_ascii(scan, path)
            entry = scan[snum]
            return entry['energy'], entry['intensity']
    except Exception:
        pass  # Fall through to simple format
    
    # Fallback: simple 2-column format
    data = np.genfromtxt(path, comments="#", delimiter=None, dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("ASCII XES must have >= 2 columns (energy, intensity)")
    x = data[:, 0]
    y = data[:, 1]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]
```

---

#### Task 4.2: Update `xes_from_path()`
**File**: `i20_xes/modules/i20_loader.py`  
**Location**: Update existing function (lines 213-227)

**Changes**: Add RXES ASCII handling

---

#### Task 4.3: Add Integration Tests
**File**: New file `tests/test_i20_loader_unified.py`

**Test cases**:
1. Load RXES from NeXus → verify grid structure
2. Load RXES from ASCII → verify grid structure
3. Compare NeXus vs ASCII output for same scan
4. Test grid analysis with different outer axes
5. Test validation catches bad data
6. Test sparse grid handling

---

## 📝 IMPLEMENTATION CHECKLIST

### Phase 1: Core Functions
- [ ] Implement `validate_rxes_data()`
- [ ] Implement `analyze_grid_structure()`
- [ ] Implement `normalize_grid_to_2d()`
- [ ] Implement `create_rxes_scan_entry()`
- [ ] Add docstrings with examples
- [ ] Add unit tests for each function

### Phase 2: ASCII Functions
- [ ] Implement `parse_i20_ascii_metadata()`
- [ ] Implement `load_i20_ascii_data()`
- [ ] Implement `add_scan_from_i20_ascii()`
- [ ] Test with actual I20 ASCII file
- [ ] Add error handling for edge cases

### Phase 3: NeXus Refactor
- [ ] Refactor `add_scan_from_nxs()` to use shared pipeline
- [ ] Test with existing NeXus files
- [ ] Verify backward compatibility
- [ ] Check GUI still works with refactored loader

### Phase 4: Integration
- [ ] Update `xes_from_ascii()`
- [ ] Update `xes_from_path()`
- [ ] Add integration tests
- [ ] Test in actual GUI
- [ ] Verify RXES normalization still works
- [ ] Verify XES background extraction still works

### Phase 5: Documentation
- [ ] Update `ASCII_LOADER_NOTES.md` with new implementation
- [ ] Update `README.md` with ASCII support status
- [ ] Update `PLAN.md` to mark ASCII loader as complete
- [ ] Add code comments explaining grid processing logic

---

## 🚨 CRITICAL REQUIREMENTS

1. **Backward Compatibility**: Existing NeXus files must load identically
2. **Data Validation**: All inputs validated before processing
3. **Error Messages**: Clear, actionable error messages
4. **Testing**: Each function tested independently
5. **Documentation**: Inline comments explaining grid logic
6. **Performance**: Avoid unnecessary copies of large arrays

---

## 🎯 SUCCESS CRITERIA

✅ ASCII RXES files load successfully  
✅ Output matches NeXus version of same scan  
✅ Grid structure correctly auto-detected  
✅ Both upper and lower channels work  
✅ XES background extraction works with ASCII  
✅ GUI displays ASCII RXES maps correctly  
✅ All existing tests still pass  
✅ New tests achieve >90% coverage  

---

## 📊 TESTING MATRIX

| Data Source | Scan Type | Channel | Expected Result |
|-------------|-----------|---------|-----------------|
| NeXus | RXES | Upper | Load & display 2D map |
| NeXus | RXES | Lower | Load & display 2D map |
| ASCII | RXES | Upper | Load & display 2D map |
| ASCII | RXES | Lower | Load & display 2D map |
| ASCII | XES | Upper | Load & display 1D spectrum |
| ASCII | XES | Lower | Load & display 1D spectrum |
| ASCII | Simple 2-col | N/A | Load & display 1D spectrum |

---

## 🔄 ROLLBACK PLAN

If issues arise:
1. Core functions are additions - can be disabled
2. NeXus loader changes can be reverted to original
3. ASCII functions are new - can be removed
4. Git commit strategy: one function per commit for easy rollback

---

**END OF PLAN**

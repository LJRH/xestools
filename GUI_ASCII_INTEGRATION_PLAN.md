# GUI Integration Plan: Enable ASCII RXES Loading

**Date**: October 29, 2025  
**Status**: Analysis Complete - Ready for Implementation  
**Priority**: HIGH - Complete the ASCII loader integration

---

## 📊 CURRENT STATE ANALYSIS

### ✅ What's Already Working:

1. **Backend Loader Functions** (i20_loader.py):
   - ✅ `add_scan_from_i20_ascii()` - Loads ASCII RXES into Scan container
   - ✅ `parse_i20_ascii_metadata()` - Detects RXES vs XES, channel, axes
   - ✅ `xes_from_ascii()` - Updated to try I20 format first
   - ✅ `xes_from_path()` - Works for ASCII files
   - ✅ Shared grid processing pipeline tested and functional

2. **GUI XES Loading** (main_gui.py):
   - ✅ `on_xes_load_scans()` (lines 444-538) - Already calls `xes_from_ascii()`
   - ✅ Works for both simple 2-column and I20 ASCII format
   - ✅ Properly handles channel selection
   - ✅ No changes needed here!

3. **GUI XES Normalisation Loading** (main_gui.py):
   - ✅ `on_rxes_normalise()` (lines 1259-1285) - Uses `xes_from_path()`
   - ✅ Already supports ASCII files
   - ✅ No changes needed here!

4. **GUI Wide XES Loading** (main_gui.py):
   - ✅ `on_xes_load_wide()` (lines 877-899) - Uses `xes_from_path()`
   - ✅ Already supports ASCII files
   - ✅ No changes needed here!

### ❌ What's Missing:

**ONLY ONE FUNCTION NEEDS UPDATE:**

**`_load_rxes_scan()` (main_gui.py lines 296-313)**
- Currently only loads `.nxs` files
- Needs to also support ASCII `.dat/.txt/.csv` files
- Should call `add_scan_from_i20_ascii()` for ASCII files

---

## 🎯 REQUIRED CHANGES

### **Change 1: Update File Dialog Filters**

**Location**: `main_gui.py` line 297

**Current Code**:
```python
def _load_rxes_scan(self):
    filters = ["NeXus scans (*.nxs *.h5 *.hdf *.hdf5)", "All files (*)"]
    path, _ = QFileDialog.getOpenFileName(self, "Load RXES scan", "", ";;".join(filters))
```

**New Code**:
```python
def _load_rxes_scan(self):
    filters = [
        "RXES scans (*.nxs *.h5 *.hdf *.hdf5 *.dat *.txt *.csv)",
        "NeXus (*.nxs *.h5 *.hdf *.hdf5)",
        "ASCII (*.dat *.txt *.csv)",
        "All files (*)"
    ]
    path, _ = QFileDialog.getOpenFileName(self, "Load RXES scan", "", ";;".join(filters))
```

**Impact**: User can now select ASCII files in the file dialog

---

### **Change 2: Add ASCII Loading Logic**

**Location**: `main_gui.py` lines 301-313

**Current Code**:
```python
try:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nxs", ".h5", ".hdf", ".hdf5") and i20_loader.is_probably_detector_hdf(path):
        QMessageBox.warning(self, "Detector HDF selected",
                            "This looks like a raw detector file. Please pick the scan .nxs/.h5.")
        return
    scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
    self.current_scan_number = scan_number
    self.status.showMessage(f"Loaded RXES: {path}", 5000)
    self.tabs.setCurrentIndex(0)
    self.refresh_rxes_view()
except Exception as e:
    QMessageBox.critical(self, "Load error", f"Failed to load RXES:\n{path}\n\n{e}")
```

**New Code**:
```python
try:
    ext = os.path.splitext(path)[1].lower()
    
    # Handle NeXus files
    if ext in (".nxs", ".h5", ".hdf", ".hdf5"):
        if i20_loader.is_probably_detector_hdf(path):
            QMessageBox.warning(self, "Detector HDF selected",
                                "This looks like a raw detector file. Please pick the scan .nxs/.h5.")
            return
        scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
    
    # Handle ASCII files
    elif ext in (".dat", ".txt", ".csv"):
        scan_number = i20_loader.add_scan_from_i20_ascii(self.scan, path)
    
    # Unknown extension - try NeXus as default
    else:
        scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
    
    self.current_scan_number = scan_number
    self.status.showMessage(f"Loaded RXES: {path}", 5000)
    self.tabs.setCurrentIndex(0)
    self.refresh_rxes_view()
    
except Exception as e:
    QMessageBox.critical(self, "Load error", f"Failed to load RXES:\n{path}\n\n{e}")
```

**Impact**: 
- ASCII RXES files now load through the new loader
- Proper error handling maintained
- NeXus files still work as before

---

## 🧪 TESTING STRATEGY

### **Test Case 1: Load ASCII RXES via GUI**
```
1. Launch GUI: python3 main.py
2. Click "Load" button
3. Select "RXES scan (.nxs)"
4. Navigate to i20_xes/data/rxes/XES_ZnZSM5_air500_279192_1.dat
5. File should load successfully
6. RXES tab should show 2D map
7. Channel selection (Upper/Lower) should work
8. View mode (Incident/Transfer) should work
```

**Expected Result**: ✅ ASCII RXES loads and displays correctly

---

### **Test Case 2: Verify NeXus Still Works**
```
1. Load NeXus RXES: i20_xes/data/rxes/279496_1.nxs
2. Verify it still loads correctly
3. Test all existing functionality
```

**Expected Result**: ✅ No regression in NeXus loading

---

### **Test Case 3: Compare ASCII vs NeXus**
```
Load same scan in both formats:
- ASCII: XES_ZnZSM5_air500_279192_1.dat
- NeXus: 279192_1.nxs (if available)

Compare:
- Grid dimensions
- Energy ranges
- Intensity values
- Visual appearance
```

**Expected Result**: ✅ Both formats produce identical results

---

### **Test Case 4: Error Handling**
```
1. Try to load invalid file
2. Try to load ASCII XES (1D) as RXES
3. Try to load file with missing channels
```

**Expected Result**: ✅ Clear error messages, no crashes

---

### **Test Case 5: Integration Features**
```
After loading ASCII RXES:
1. Add ROI lines → Should extract profiles
2. Change channel (Upper/Lower) → Should work if both present
3. Change view mode → Should transform correctly
4. Load XES for normalization → Should normalize map
5. Save profiles as CSV → Should save correctly
```

**Expected Result**: ✅ All RXES features work with ASCII data

---

## 📝 IMPLEMENTATION CHECKLIST

- [ ] Update `_load_rxes_scan()` file dialog filters
- [ ] Add ASCII loading branch in `_load_rxes_scan()`
- [ ] Test loading ASCII RXES via GUI
- [ ] Test loading NeXus RXES (no regression)
- [ ] Test channel selection with ASCII RXES
- [ ] Test view mode switching with ASCII RXES
- [ ] Test ROI extraction with ASCII RXES
- [ ] Test RXES normalization with ASCII RXES
- [ ] Test error handling with invalid files
- [ ] Update user documentation if needed

---

## 🔍 CODE LOCATIONS REFERENCE

### Files to Modify:
1. **`i20_xes/main_gui.py`**:
   - Line 297: Update filters
   - Lines 301-313: Add ASCII handling logic

### Files Already Complete (No Changes Needed):
1. ✅ `i20_xes/modules/i20_loader.py` - Backend complete
2. ✅ `i20_xes/main_gui.py:on_xes_load_scans()` - XES loading works
3. ✅ `i20_xes/main_gui.py:on_rxes_normalise()` - Normalisation works
4. ✅ `i20_xes/main_gui.py:on_xes_load_wide()` - Wide scan works

---

## 🚨 CRITICAL SUCCESS FACTORS

1. **Minimal Changes**: Only 2 small edits to one function
2. **No Regressions**: NeXus loading must continue to work
3. **Error Handling**: Clear messages for invalid files
4. **User Experience**: Seamless - users shouldn't notice difference between formats
5. **Testing**: All RXES features must work with ASCII data

---

## 🎯 EXPECTED OUTCOMES

After implementation:

✅ **Users can load RXES from ASCII files via GUI**  
✅ **No changes needed to XES loading (already works)**  
✅ **No changes needed to normalisation (already works)**  
✅ **No changes needed to wide scan loading (already works)**  
✅ **All existing functionality preserved**  
✅ **ASCII and NeXus data treated identically after loading**  

---

## 💡 IMPLEMENTATION NOTES

### Why So Simple?

The implementation is simple because:
1. **Backend is Complete**: All hard work done in i20_loader.py
2. **XES Already Works**: The pattern is already established
3. **Smart Architecture**: GUI just calls backend functions
4. **Type Detection**: Backend handles format detection automatically

### What Makes This Safe?

1. **Isolated Changes**: Only one function modified
2. **Clear Branching**: Separate paths for NeXus vs ASCII
3. **Fallback Logic**: Unknown extensions try NeXus (safe default)
4. **Error Handling**: Existing try/catch preserved
5. **No API Changes**: refresh_rxes_view() doesn't change

---

## 📊 COMPARISON: Before vs After

### Before:
```
File Dialog: Only shows .nxs files
Load Logic:  Only calls add_scan_from_nxs()
Result:      ASCII RXES cannot be loaded via GUI
```

### After:
```
File Dialog: Shows .nxs AND .dat/.txt/.csv files
Load Logic:  Calls add_scan_from_nxs() OR add_scan_from_i20_ascii()
Result:      Both NeXus and ASCII RXES load seamlessly
```

---

## 🔄 ROLLBACK PLAN

If issues arise:
1. Changes are in one function only
2. Easy to revert to previous version
3. No database or file format changes
4. Users can still use backend functions directly

---

**END OF ANALYSIS**

**RECOMMENDATION**: Proceed with implementation - changes are minimal, well-understood, and low-risk.


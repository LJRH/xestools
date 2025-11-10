# Silx Integration Plan for I20 XES/RXES Viewer

## Executive Summary

This plan outlines the integration of **silx** (the synchrotron toolkit) into the I20 XES/RXES viewer application, specifically utilizing `ImageView` for enhanced 2D RXES data visualization and `findcontours` for improved ROI extraction. The silx library provides professional-grade tools specifically designed for synchrotron data visualization and is widely used at European synchrotron facilities.

---

## 1. Current State Analysis

### Current Implementation (matplotlib-based)

**PlotWidget** (`i20_xes/widgets/plot_widget.py`):
- Uses matplotlib with FigureCanvas backend
- Dual subplot layout: main 2D image (ax_img) + 1D profile plot (ax_prof)
- Manual ROI line implementation with draggable vertical/horizontal lines
- Custom bandwidth bands using `axvspan`/`axhspan`
- Profile extraction via numpy masking and summation
- ~1080 lines of code with complex error handling

**Current Capabilities**:
- ✅ 2D RXES maps with pcolormesh/imshow
- ✅ Interactive ROI lines (vertical/horizontal)
- ✅ Adjustable bandwidth for integration
- ✅ Live profile updates
- ✅ Multiple extraction modes (incident/emission/transfer)
- ✅ XES overlay plotting

**Current Limitations**:
- ❌ No advanced image manipulation tools (zoom, pan, contrast)
- ❌ Manual colorbar management with cleanup issues
- ❌ No built-in ROI shapes beyond lines
- ❌ Limited interactive profile editing
- ❌ Complex matplotlib resource management
- ❌ No contour analysis capabilities

---

## 2. Silx Capabilities Overview

### silx.gui.plot.ImageView
Professional 2D image viewer with:
- **Built-in colorbar** with automatic management
- **Interactive colormap editor** (log scale, autoscale, gamma)
- **Zoom, pan, and aspect ratio controls**
- **Profile tools** (line profiles, cross profiles)
- **ROI manager** with multiple shape types
- **Histogram widget** for intensity distribution
- **Pixel position and value readout**
- **Image origin handling** (auto-flip for correct orientation)
- **Export capabilities** (save image, data)

### silx.image.shapes (findcontours)
Advanced contour analysis:
- **Level set extraction** at specified intensity values
- **Polygon ROI creation** from contours
- **Contour smoothing and simplification**
- **Area and perimeter calculations**
- **Multi-level contour extraction**
- **Integration with silx ROI system**

### silx.gui.plot.PlotWidget (1D)
For profile plots:
- **Interactive legends**
- **Curve styling tools**
- **Data inspection cursors**
- **Export to CSV/HDF5**
- **Multiple curve management**

---

## 3. Integration Architecture

### Approach: Hybrid Integration (Recommended)

Replace the matplotlib-based 2D image display with silx while maintaining some matplotlib for specialized workflows.

### Component Breakdown

```
┌─────────────────────────────────────────────┐
│          MainWindow (main_gui.py)           │
└─────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼───────────┐    ┌──────────▼──────────┐
│  Left Panel       │    │  Right Panel        │
│  (Controls)       │    │  (Visualization)    │
│                   │    │                     │
│  - IOPanel        │    │  NEW: SilxPlot      │
│  - RXESPanel      │    │  Widget             │
│  - XESPanel       │    │                     │
└───────────────────┘    └─────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
          ┌─────────▼─────────┐      ┌──────────▼─────────┐
          │ silx.ImageView    │      │ silx.Plot1D        │
          │ (RXES 2D maps)    │      │ (profiles)         │
          │                   │      │                    │
          │ - ColorBar        │      │ - Multi-curve      │
          │ - ROI Manager     │      │ - Interactive      │
          │ - Zoom/Pan        │      │                    │
          └───────────────────┘      └────────────────────┘
```

---

## 4. Detailed Implementation Plan

### Phase 1: Setup and Basic Integration (Week 1)

#### 1.1 Install silx
```bash
pip install silx[full]  # Full installation with OpenGL support
# or
pip install silx  # Minimal installation
```

#### 1.2 Create New Widget: `SilxPlotWidget`

**Location**: `i20_xes/widgets/silx_plot_widget.py`

**Features**:
- Wrapper around `silx.gui.plot.ImageView` and `silx.gui.plot.Plot1D`
- Maintain same API as current PlotWidget for compatibility
- Signal compatibility: `lines_changed` → `roiChanged`

**Key Classes to Use**:
```python
from silx.gui import qt
from silx.gui.plot import ImageView, Plot1D
from silx.gui.plot.items.roi import RectangleROI, HorizontalLineROI, VerticalLineROI
from silx.gui.plot.tools.roi import RegionOfInterestManager
```

#### 1.3 Basic Structure
```python
class SilxPlotWidget(QWidget):
    """Enhanced plot widget using silx for 2D RXES visualization."""
    
    roi_changed = QtCore.Signal()  # Replaces lines_changed
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layout: image view on top, profile plot below
        layout = QVBoxLayout(self)
        
        # Main 2D image viewer
        self.imageView = ImageView(parent=self)
        self.imageView.setKeepDataAspectRatio(True)
        self.imageView.setColormap("viridis")
        
        # Profile plot
        self.profilePlot = Plot1D(parent=self)
        self.profilePlot.setGraphTitle("ROI Profiles")
        self.profilePlot.setGraphXLabel("Energy (eV)")
        self.profilePlot.setGraphYLabel("Integrated counts")
        
        # Add to layout with sizing
        layout.addWidget(self.imageView, 3)  # 75% height
        layout.addWidget(self.profilePlot, 1)  # 25% height
        
        # ROI manager
        self.roiManager = RegionOfInterestManager(self.imageView)
        self.roiManager.sigRoiChanged.connect(self._on_roi_changed)
```

---

### Phase 2: RXES 2D Visualization (Week 2)

#### 2.1 Migrate 2D Plotting

**Replace**: `PlotWidget._plot_2d_dataset()` (lines 504-551)

**With**: Silx-based implementation

```python
def plot_2d_dataset(self, dataset: DataSet):
    """Plot 2D RXES map using silx ImageView."""
    
    Z = dataset.z
    if Z is None:
        raise ValueError("No Z data for 2D plot")
    
    # Set image data
    if dataset.x2d is not None and dataset.y2d is not None:
        # Curvilinear grid (energy transfer mode)
        origin = (dataset.x2d[0,0], dataset.y2d[0,0])
        scale = self._calculate_scale(dataset.x2d, dataset.y2d)
        self.imageView.setImage(Z, origin=origin, scale=scale)
    else:
        # Regular grid (incident energy mode)
        if dataset.x is not None and dataset.y is not None:
            origin = (dataset.x.min(), dataset.y.min())
            scale = (
                (dataset.x.max() - dataset.x.min()) / Z.shape[1],
                (dataset.y.max() - dataset.y.min()) / Z.shape[0]
            )
            self.imageView.setImage(Z, origin=origin, scale=scale)
        else:
            self.imageView.setImage(Z)
    
    # Set axis labels
    self.imageView.setGraphXLabel(dataset.xlabel or "X")
    self.imageView.setGraphYLabel(dataset.ylabel or "Y")
    self.imageView.setGraphTitle(os.path.basename(dataset.source) or "Data")
```

**Benefits**:
- ✅ Automatic colorbar management (no more cleanup issues!)
- ✅ Built-in zoom/pan/reset tools
- ✅ Interactive colormap adjustment
- ✅ Proper aspect ratio handling

---

### Phase 3: ROI Extraction with Silx (Week 3)

#### 3.1 Replace Manual Line ROIs

**Current**: Custom matplotlib lines with manual dragging (lines 855-906)

**New**: Silx ROI system

```python
def set_line_orientation(self, orientation: str):
    """Set ROI line orientation (vertical/horizontal)."""
    self._line_orientation = orientation
    
    # Clear existing ROIs
    self.roiManager.clear()
    
    # Add appropriate ROI type
    if orientation == "vertical":
        roi = VerticalLineROI()
        roi.setName("ROI 1")
        roi.setPosition(self._get_default_position())
    else:
        roi = HorizontalLineROI()
        roi.setName("ROI 1")
        roi.setPosition(self._get_default_position())
    
    self.roiManager.addRoi(roi)
    roi.sigRegionChanged.connect(self._on_roi_changed)

def add_line(self):
    """Add another ROI line."""
    rois = self.roiManager.getRois()
    if len(rois) >= 3:
        return  # Max 3 lines
    
    if self._line_orientation == "vertical":
        roi = VerticalLineROI()
    else:
        roi = HorizontalLineROI()
    
    roi.setName(f"ROI {len(rois) + 1}")
    roi.setPosition(self._get_next_position())
    self.roiManager.addRoi(roi)
```

---

### Phase 4: Profile Extraction (Week 4)

#### 4.1 Integrate with Silx Profile Tools

```python
def _on_roi_changed(self):
    """Handle ROI changes and update profiles."""
    self.update_profiles()
    self.roi_changed.emit()

def update_profiles(self):
    """Extract profiles from current ROIs and plot."""
    if self.dataset is None or self.dataset.kind != "2D":
        return
    
    Z = self.dataset.z
    X2D, Y2D = self._build_coordinate_grids(self.dataset)
    
    # Get all ROI lines (not bands)
    line_rois = [roi for roi in self.roiManager.getRois() 
                 if isinstance(roi, (VerticalLineROI, HorizontalLineROI))
                 and not roi.getName().endswith("_band")]
    
    if not line_rois:
        self.profilePlot.clear()
        return
    
    # Clear previous profiles
    self.profilePlot.clear()
    
    # Extract profiles for each ROI
    for idx, roi in enumerate(line_rois):
        x_profile, y_profile = self._extract_roi_profile(
            roi, Z, X2D, Y2D, self._bandwidth
        )
        
        if x_profile is not None and y_profile is not None:
            # Add to plot with distinct color
            self.profilePlot.addCurve(
                x_profile, y_profile,
                legend=roi.getName(),
                color=self._get_roi_color(idx)
            )
    
    self.profilePlot.replot()
```

---

### Phase 5: Contour Analysis Integration (Week 5)

#### 5.1 Add Contour-Based ROI Creation

```python
from silx.image import shapes

def add_contour_roi_from_level(self, intensity_level: float):
    """Create ROI from contour at specified intensity level."""
    if self.dataset is None or self.dataset.kind != "2D":
        return
    
    Z = self.dataset.z
    
    # Find contours at intensity level
    contours = shapes.find_contours(Z, level=intensity_level)
    
    if not contours:
        QMessageBox.warning(
            self, "No contours found",
            f"No contours found at intensity level {intensity_level}"
        )
        return
    
    # Use the largest contour (by area)
    largest_contour = max(contours, key=lambda c: shapes.polygon_area(c))
    
    # Convert to ROI
    from silx.gui.plot.items.roi import PolygonROI
    
    poly_roi = PolygonROI()
    poly_roi.setName(f"Contour @ {intensity_level:.2f}")
    
    # Convert contour points to data coordinates
    points = self._pixel_to_data_coords(largest_contour)
    poly_roi.setPoints(points)
    
    self.roiManager.addRoi(poly_roi)
    
    return poly_roi
```

---

## 5. Migration Strategy

### Phased Rollout (Recommended)

#### Phase 1: Parallel Implementation (Weeks 1-2)
- Create `SilxPlotWidget` alongside existing `PlotWidget`
- Add feature flag in MainWindow to switch between implementations
- Test basic 2D plotting with silx

```python
# In main_gui.py
USE_SILX = True  # Feature flag

if USE_SILX:
    from i20_xes.widgets.silx_plot_widget import SilxPlotWidget
    self.plot = SilxPlotWidget()
else:
    from i20_xes.widgets.plot_widget import PlotWidget
    self.plot = PlotWidget()
```

#### Phase 2: Feature Parity (Weeks 3-4)
- Implement all existing features in silx version
- Run side-by-side testing
- Fix edge cases and bugs

#### Phase 3: User Testing (Week 5)
- Beta testing with actual I20 data
- Gather feedback on usability
- Performance benchmarking

#### Phase 4: Full Migration (Week 6)
- Remove matplotlib PlotWidget
- Update documentation
- Clean up dependencies

---

## 6. API Compatibility Layer

To minimize changes to `main_gui.py`, maintain compatible method signatures:

### Current API → Silx API Mapping

| Current Method | Silx Equivalent | Notes |
|----------------|----------------|-------|
| `plot.plot(dataset)` | `imageView.setImage()` | Need wrapper |
| `plot.set_line_orientation()` | `roiManager` ROI type selection | Different paradigm |
| `plot.add_line()` | `roiManager.addRoi()` | Direct mapping |
| `plot.set_bandwidth()` | Custom band ROI logic | New implementation |
| `plot.get_line_positions()` | `[roi.getPosition() for roi in rois]` | Simple iteration |
| `plot.lines_changed` signal | `roiManager.sigRoiChanged` | Direct mapping |

---

## 7. Testing Strategy

### Unit Tests
```python
# tests/test_silx_plot_widget.py
def test_2d_plotting():
    """Test 2D RXES map plotting."""
    widget = SilxPlotWidget()
    dataset = create_test_2d_dataset()
    widget.plot(dataset)
    assert widget.imageView.getImage() is not None

def test_roi_addition():
    """Test ROI line addition."""
    widget = SilxPlotWidget()
    widget.set_line_orientation("vertical")
    widget.add_line()
    assert len(widget.roiManager.getRois()) == 1
```

---

## 8. Benefits Summary

### User-Facing Improvements
1. **Professional UI**: Industry-standard synchrotron visualization tools
2. **Better Interaction**: Built-in zoom, pan, colormap adjustment
3. **ROI Flexibility**: Support for multiple ROI shapes (not just lines)
4. **Contour Analysis**: New capability for feature extraction
5. **Performance**: OpenGL-accelerated rendering (with silx[full])
6. **Export**: Direct HDF5/image export from silx widgets

### Developer Benefits
1. **Less Code**: ~40% reduction in custom plotting code
2. **Maintenance**: No manual matplotlib resource cleanup
3. **Robustness**: Industry-tested at major synchrotrons (ESRF, SOLEIL, etc.)
4. **Features**: Leverage ongoing silx development
5. **Standards**: Align with community best practices

### Technical Improvements
1. **Memory Management**: Automatic resource cleanup
2. **Rendering**: OpenGL backend option for large datasets
3. **Colormaps**: Scientific colormap library included
4. **Profiles**: Built-in profile extraction tools
5. **ROIs**: Robust ROI serialization for saving/loading

---

## 9. Implementation Timeline

| Week | Focus | Deliverable |
|------|-------|------------|
| 1 | Setup & Basic Silx | SilxPlotWidget stub, 2D plotting working |
| 2 | RXES Visualization | Both view modes (incident/transfer) working |
| 3 | ROI System | Line ROIs, bandwidth, dragging functional |
| 4 | Profile Extraction | All extraction modes working |
| 5 | Advanced Features | Contour analysis, colormap tools |
| 6 | Integration | Feature flag, parallel implementation |
| 7 | Testing | Bug fixes, performance tuning |
| 8 | Deployment | Documentation, release |

**Total Estimated Time**: 6-8 weeks for full implementation and testing

---

## 10. Implementation Checklist

### Preparation
- [ ] Install silx: `pip install silx[full]`
- [ ] Review silx examples
- [ ] Set up test environment with I20 data
- [ ] Create feature branch: `git checkout -b feature/silx-integration`

### Week 1: Foundation
- [ ] Create `i20_xes/widgets/silx_plot_widget.py`
- [ ] Implement basic SilxPlotWidget class
- [ ] Add 2D image plotting with ImageView
- [ ] Test with simple datasets

### Week 2: RXES Visualization
- [ ] Implement incident energy mode plotting
- [ ] Implement energy transfer mode plotting
- [ ] Handle curvilinear grids (x2d, y2d)
- [ ] Test with real I20 RXES scans

### Week 3: ROI System
- [ ] Implement vertical/horizontal line ROIs
- [ ] Add bandwidth visualization
- [ ] Connect ROI signals to profile updates
- [ ] Test ROI dragging and manipulation

### Week 4: Profile Extraction
- [ ] Implement profile extraction logic
- [ ] Add Plot1D for profile display
- [ ] Support all extraction modes (incident/emission/transfer)
- [ ] Test profile accuracy vs old implementation

### Week 5: Advanced Features
- [ ] Add contour-based ROI creation
- [ ] Implement interactive contour dialog
- [ ] Add colormap adjustment controls
- [ ] Test with various data ranges

### Week 6: Integration & Polish
- [ ] Integrate with MainWindow
- [ ] Update RXESPanel controls
- [ ] Add feature flag for switching implementations
- [ ] Write documentation and examples

### Week 7: Testing & Refinement
- [ ] Run full test suite
- [ ] Beta test with I20 data
- [ ] Fix bugs and edge cases
- [ ] Performance benchmarking

### Week 8: Deployment
- [ ] Remove feature flag
- [ ] Update README and documentation
- [ ] Create migration guide
- [ ] Tag release

---

## 11. Success Criteria

### Must Have
- [ ] All existing RXES viewing modes work
- [ ] ROI line extraction matches old implementation (±1% accuracy)
- [ ] No regressions in XES workflow
- [ ] Documentation updated
- [ ] Test coverage >80%

### Should Have
- [ ] Improved UI responsiveness
- [ ] Contour-based ROI creation working
- [ ] Professional colormap controls
- [ ] Export features functional

### Nice to Have
- [ ] OpenGL rendering for large datasets
- [ ] Advanced ROI shapes (rectangles, polygons)
- [ ] Interactive profile editing
- [ ] ROI save/load to project files

---

## References

- **Silx Documentation**: https://www.silx.org/doc/silx/latest/
- **Silx GitHub**: https://github.com/silx-kit/silx
- **Silx Examples**: https://github.com/silx-kit/silx/tree/main/examples
- **ImageView Tutorial**: https://www.silx.org/doc/silx/latest/modules/gui/plot/imageview.html
- **ROI Documentation**: https://www.silx.org/doc/silx/latest/modules/gui/plot/roi.html

---

**Created**: November 10, 2025  
**Author**: OpenCode Assistant  
**Status**: Planning Complete, Implementation Phase 1 Ready

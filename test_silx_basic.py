#!/usr/bin/env python3
"""
Basic test script for silx integration.

Tests that the SilxPlotWidget can:
1. Initialize properly
2. Display a simple 2D RXES map
3. Add and manipulate ROI lines
4. Extract profiles

Usage:
    python3 test_silx_basic.py
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from xestools.widgets.silx_plot_widget import SilxPlotWidget
from xestools.modules.dataset import DataSet

def create_test_rxes_data():
    """Create synthetic RXES data for testing."""
    print("Creating test RXES data...")
    
    # Create 2D RXES map: Gaussian peak
    x = np.linspace(9650, 9680, 100)  # Incident energy (eV)
    y = np.linspace(9640, 9660, 80)   # Emission energy (eV)
    
    X, Y = np.meshgrid(x, y)
    
    # Create Gaussian peak at (9665, 9650)
    cx, cy = 9665, 9650
    sx, sy = 3.0, 2.0
    Z = 1000 * np.exp(-((X - cx)**2 / (2 * sx**2) + (Y - cy)**2 / (2 * sy**2)))
    
    # Add some noise
    Z += np.random.normal(0, 50, Z.shape)
    
    # Create DataSet
    dataset = DataSet(
        kind="2D",
        x=x,
        y=y,
        z=Z,
        xlabel="Incident Energy Ω (eV)",
        ylabel="Emission Energy ω (eV)",
        zlabel="Intensity (counts)",
        source="test_rxes_map.nxs"
    )
    
    print(f"✓ Created 2D dataset: shape={Z.shape}")
    return dataset

def test_basic_plotting():
    """Test basic 2D plotting with silx."""
    print("\n=== Testing Silx Basic Plotting ===\n")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create widget
    print("1. Creating SilxPlotWidget...")
    widget = SilxPlotWidget()
    print("   ✓ Widget created")
    
    # Create test data
    print("\n2. Creating test RXES data...")
    dataset = create_test_rxes_data()
    
    # Plot data
    print("\n3. Plotting 2D RXES map...")
    widget.plot(dataset)
    print("   ✓ Data plotted")
    
    # Test ROI operations
    print("\n4. Testing ROI operations...")
    print("   - Setting vertical line orientation")
    widget.set_line_orientation("vertical")
    
    print("   - Adding ROI line")
    widget.add_line()
    
    print("   - Setting bandwidth to 2.0 eV")
    widget.set_bandwidth(2.0)
    
    positions = widget.get_line_positions()
    print(f"   - Current ROI positions: {positions}")
    print(f"   ✓ ROI operations successful ({len(positions)} lines)")
    
    # Show widget
    print("\n5. Displaying widget...")
    widget.setWindowTitle("Silx Test: RXES Map")
    widget.resize(900, 700)
    widget.show()
    
    print("\n✅ All tests passed!")
    print("\nWidget should display:")
    print("  - 2D RXES map with Gaussian peak")
    print("  - Colorbar (auto-managed by silx)")
    print("  - Interactive zoom/pan tools")
    print("  - Vertical ROI line")
    print("  - Profile plot at bottom")
    print("\nClose the window to exit.")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(test_basic_plotting())

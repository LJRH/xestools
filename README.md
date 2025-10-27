How the transformation from 2D incident energy to energy transfer is performed  

    Load the 2D meshes from the .nxs file:  
        Incident energy mesh Ω comes from /entry1/I1/bragg1WithOffset  

. 
Emission energy mesh ω comes from /entry1/I1/XESEnergyUpper or XESEnergyLower (depending on detector)  
. 
The intensity Z comes from FFI1_medipix1 (Upper) or FFI1_medipix2 (Lower), with ROI-total as fallback  

    . 
     

Reduce the 2D meshes to 1D axes (reduce_axes_for):  

    Compare variability across rows vs columns of the ω mesh to detect its orientation. 
    Build y_omega (emission axis) by taking the median along the orthogonal direction. 
    Build x_Omega (incident axis) likewise from the Ω mesh. 
    Return a transposed flag to indicate if Z must be transposed to align rows with ω and columns with Ω. 
     

Align the intensity matrix:  

    If transposed is True, transpose Z so that Z.shape == (len(y_omega), len(x_Omega)). 
     

Construct the energy-transfer coordinates for plotting:  

    Keep the X axis as incident energy: X2D = broadcast(x_Omega)[None, :]. 
    Compute the energy transfer Δ row-wise: Y2D = X2D − broadcast(y_omega)[:, None]. 
    This yields a grid where each 
     
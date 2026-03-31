"""Debug 3D RMS to verify it produces meaningful contrast."""
import sys
import os
import importlib.util

_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "diffsim", "data"
)

def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

fg = _load("diffsim.data.flumy_generator", os.path.join(_data_dir, "flumy_generator.py"))
sm = _load("diffsim.data.seismic", os.path.join(_data_dir, "seismic.py"))

import numpy as np

# More realistic: lateral facies variation across z-constant plan view
# This tests what happens with a real Flumy output: each (x,y) has
# DIFFERENT facies across depth, so vertical reflectivity exists
block = np.zeros((20, 20, 30), dtype=np.int8)
block[:, :, :] = fg.FACIES_MUD

# Channel body: sand in center laterally, cuts through vertically
block[5:15, 5:15, 8:22] = fg.FACIES_SAND
block[4:16, 4:16, 7:8] = fg.FACIES_BANK
block[4:16, 4:16, 22:23] = fg.FACIES_BANK

# Check reflectivity
refl = sm.compute_reflectivity_vertical(sm.facies_to_ai(block))
print("Reflectivity at channel center (10,10), z=5..25:")
print("  ", refl[10, 10, 5:25])
print("Reflectivity at mud background (0,0), z=5..25:")
print("  ", refl[0, 0, 5:25])

# RMS at z=15 (middle of sand body) - within sand, no vertical contrasts
rms_mid = sm.generate_rms_from_facies_3d(
    block, z_target=15, rms_window_half=5,
    noise_level=0.0, smooth_sigma=0.0
)
print("\nRMS at z=15 (mid-sand), no noise/smooth:")
print("  Shape:", rms_mid.shape)
print("  Channel center (10,10):", rms_mid[10, 10])
print("  Mud background (0,0):", rms_mid[0, 0])
print("  Range:", rms_mid.min(), "to", rms_mid.max())

# RMS at z=8 (top of sand) - across sand/mud boundary
rms_top = sm.generate_rms_from_facies_3d(
    block, z_target=8, rms_window_half=5,
    noise_level=0.0, smooth_sigma=0.0
)
print("\nRMS at z=8 (top boundary), no noise/smooth:")
print("  Channel center (10,10):", rms_top[10, 10])
print("  Mud background (0,0):", rms_top[0, 0])
print("  Range:", rms_top.min(), "to", rms_top.max())

# RMS with larger window that spans the full sand body
rms_wide = sm.generate_rms_from_facies_3d(
    block, z_target=15, rms_window_half=10,
    noise_level=0.0, smooth_sigma=0.5
)
print("\nRMS at z=15 wide window (half=10), with smoothing:")
print("  Channel center (10,10):", rms_wide[10, 10])
print("  Channel edge (5,10):", rms_wide[5, 10])
print("  Mud background (0,0):", rms_wide[0, 0])
print("  Range:", rms_wide.min(), "to", rms_wide.max())

# 2D fallback for comparison
facies_2d = block[:, :, 15]
print("\nFacies at z=15 unique values:", np.unique(facies_2d))
rms_2d = sm.generate_rms_from_facies_2d(facies_2d, smooth_sigma=2.0, noise_level=0.0)
print("2D RMS at z=15:")
print("  Channel center (10,10):", rms_2d[10, 10])
print("  Mud background (0,0):", rms_2d[0, 0])
print("  Range:", rms_2d.min(), "to", rms_2d.max())

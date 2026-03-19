import numpy as np
from PIL import Image
import math
import os

# === Step 1: Set your input/output paths ===
input_path = "C:/Users/ADMIN/TensilProject/cnn_convolver_output.hex"
output_path_rgb = "C:/Users/ADMIN/TensilProject/fpga_output_image_rgb.png"
output_path_gray = "C:/Users/ADMIN/TensilProject/fpga_output_image_gray.png"

# === Step 2: Load all valid hex values ===
with open(input_path, "r") as f:
    lines = [line.strip().split()[0]
             for line in f if line.strip() and not line.startswith("#")]

if len(lines) == 0:
    raise ValueError("❌ No valid hex values found in the file!")

data_int = np.array([int(x, 16) for x in lines], dtype=np.uint32)
print(f"✅ Loaded {len(data_int)} values from {input_path}")

# === Step 3: Try interpreting as RGB (0xRRGGBBxx or 0xRRGGBB) ===
# Extract RGB components
R = (data_int >> 24) & 0xFF
G = (data_int >> 16) & 0xFF
B = (data_int >> 8) & 0xFF

# Check if RGB interpretation makes sense
# if all equal, likely grayscale/feature data
is_rgb = not (np.all((R == G) & (G == B)))

if is_rgb:
    print("🎨 Detected color-like data — creating RGB image.")
    pixels = np.stack([R, G, B], axis=1).astype(np.uint8)
    H, W = 32, 32   # Expected CNN image size
    n = H * W
    if len(pixels) < n:
        print(f"⚠️ Only {len(pixels)} pixels found, padding to {n}.")
        pad = np.zeros((n - len(pixels), 3), dtype=np.uint8)
        pixels = np.vstack([pixels, pad])
    elif len(pixels) > n:
        print(f"⚠️ Found {len(pixels)} pixels, truncating to {n}.")
        pixels = pixels[:n]
    img = pixels.reshape((H, W, 3))
    Image.fromarray(img).save(output_path_rgb)
    print(f"✅ Saved RGB image as {output_path_rgb}")

else:
    print("📊 Detected feature map / grayscale data — creating intensity image.")
    # === Step 4: Normalize for grayscale visualization ===
    arr = data_int.astype(float)
    arr_norm = (255 * (arr - arr.min()) /
                (np.ptp(arr) + 1e-9)).astype(np.uint8)

    side = math.ceil(math.sqrt(len(arr_norm)))
    new_size = side * side
    if new_size != len(arr_norm):
        print(
            f"⚠️ Output has {len(arr_norm)} values (not a perfect square). Padding to {new_size}.")
        pad = np.zeros(new_size, dtype=np.uint8)
        pad[:len(arr_norm)] = arr_norm
        arr_norm = pad

    img = arr_norm.reshape((side, side))
    Image.fromarray(img).save(output_path_gray)
    print(f"✅ Saved grayscale image as {output_path_gray}")

print("✨ Visualization complete!")

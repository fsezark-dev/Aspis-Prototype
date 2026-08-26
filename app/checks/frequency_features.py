import numpy as np
from PIL import Image


def radial_high_frequency_ratio(image_path: str) -> float:
    image = Image.open(image_path).convert("L")
    array = np.array(image, dtype=np.float64)

    fft = np.fft.fft2(array)
    shifted_fft = np.fft.fftshift(fft)
    magnitude = np.abs(shifted_fft)

    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2

    y, x = np.ogrid[:height, :width]
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = min(center_y, center_x)

    total_energy = magnitude.sum()

    if total_energy == 0:
        return 0.0

    high_frequency_mask = radius > (0.6 * max_radius)

    return float(
        magnitude[high_frequency_mask].sum() / total_energy
    )
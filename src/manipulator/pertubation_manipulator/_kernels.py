import numpy as np

# IMPORTANT: this implementation is based on PerturbationDrive https://github.com/ast-fortiss-tum/perturbation-drive


def create_disk_kernel(radius: int) -> np.ndarray:
    """Create a normalised disk-shaped convolution kernel.

    :param radius: Kernel radius in pixels.
    :returns: 2-D float32 array of shape (2*radius+1, 2*radius+1).
    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel[mask] = 1
    # Normalize the kernel so that the sum of its elements is 1.
    kernel /= kernel.sum()
    return kernel


def create_motion_blur_kernel(size: int, angle: float) -> np.ndarray:
    """Create a normalised motion-blur convolution kernel.

    :param size: Kernel side length in pixels.
    :param angle: Blur direction in degrees.
    :returns: 2-D float64 array of shape (size, size).
    """
    # Create an empty kernel
    kernel = np.zeros((size, size))
    # Convert angle to radian
    angle = np.deg2rad(angle)
    # Calculate the center of the kernel
    center = size // 2
    # Calculate the slope of the line
    slope = np.tan(angle)
    # Fill in the kernel
    for y in range(size):
        x = int(slope * (y - center) + center)
        if 0 <= x < size:
            kernel[y, x] = 1
    # Normalize the kernel
    kernel /= kernel.sum()
    return kernel

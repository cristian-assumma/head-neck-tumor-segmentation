import numpy as np
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu

def apply_rician_denoise(img_norm: np.ndarray, sigma: float) -> np.ndarray:
    """
    Applies custom Non-Local Means denoising with Rician bias correction.
    
    Parameters
    ----------
    img_norm : np.ndarray
        Normalized image array in range [0, 1].
    sigma : float
        Estimated noise standard deviation.
        
    Returns
    -------
    np.ndarray
        Denoised image array.
    """
    def rician_bias_correction(x, s):
        return np.sqrt(np.maximum(x**2 - 2 * (s ** 2), 0))

    img_bias_corrected = rician_bias_correction(img_norm, sigma)

    # Morphological gradient for edge detection
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(img_bias_corrected, kernel)
    eroded = cv2.erode(img_bias_corrected, kernel)
    grad = dilated - eroded

    # Otsu thresholding to separate high-frequency (edges) and low-frequency regions
    th = threshold_otsu(grad)
    high_freq_map = grad > th
    low_freq_map = ~high_freq_map

    def apply_nlm(img, search_d, patch_r, h_val):
        sigma_est = estimate_sigma(img, average_sigmas=True)
        return denoise_nl_means(
            img,
            h=h_val * sigma_est,
            patch_size=2 * patch_r + 1,
            patch_distance=search_d,
            channel_axis=None
        )

    # Aggressive filtering on homogeneous areas, conservative on edges
    img_high = apply_nlm(img_bias_corrected * high_freq_map, search_d=3, patch_r=1, h_val=0.8)
    img_low  = apply_nlm(img_bias_corrected * low_freq_map,  search_d=3, patch_r=2, h_val=1.1)

    return img_high + img_low

def enhance_image_pipeline(image_path: str, ksize: int = 3, sobel_ksize: int = 3, 
                           contrast_factor: float = 2.0, sobel_thresh: int = 50) -> np.ndarray:
    """
    Executes the full preprocessing pipeline: Rician denoising, median filtering, 
    and Sobel-based localized edge contrast enhancement.

    Parameters
    ----------
    image_path : str
        Path to the input grayscale image.
    ksize : int, optional
        Kernel size for the median filter (default is 3).
    sobel_ksize : int, optional
        Kernel size for the Sobel operator (default is 3).
    contrast_factor : float, optional
        Amplification factor for detected edge pixels (default is 2.0).
    sobel_thresh : int, optional
        Intensity threshold for significant edge detection (default is 50).

    Returns
    -------
    np.ndarray
        Processed image as a 2D uint8 array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Normalization and Denoising
    img_norm = img.astype(np.float32) / 255.0
    sigma = estimate_sigma(img_norm, average_sigmas=True)
    denoised = apply_rician_denoise(img_norm, sigma)
    denoised_uint8 = (denoised * 255).astype(np.uint8)

    # Median Filtering
    median_filtered = cv2.medianBlur(denoised_uint8, ksize)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(median_filtered, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(median_filtered, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
    max_val = np.max(sobel_magnitude)

    if max_val == 0:
        edges = np.zeros_like(sobel_magnitude, dtype=np.uint8)
    else:
        sobel_magnitude = (sobel_magnitude / max_val) * 255
        sobel_magnitude = sobel_magnitude.astype(np.uint8)
        _, edges = cv2.threshold(sobel_magnitude, sobel_thresh, 255, cv2.THRESH_BINARY)

    # Localized Contrast Enhancement
    edges_mask = edges > 0
    enhanced = median_filtered.astype(np.float32)
    enhanced[edges_mask] *= contrast_factor
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced

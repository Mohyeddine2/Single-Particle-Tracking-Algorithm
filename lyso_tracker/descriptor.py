import numpy as np
import cv2
from scipy.ndimage import maximum_filter, label, center_of_mass,gaussian_filter
from skimage.measure import moments

def extract_patch(image, x, y, size=11):
    half = size // 2
    x, y = int(x), int(y)
    h, w = image.shape
    if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
        return np.zeros((size, size), dtype=image.dtype)
    return image[y - half:y + half + 1, x - half:x + half + 1]

def compute_descriptor(patch):
    return patch.flatten()

def detect_particles(gray, radius=5, percentile= 99, mask=None):
    """
    Detects particles in a grayscale image using local maxima and intensity thresholding.

    Parameters:
    - gray: Grayscale input image (numpy array).
    - radius: Radius for local maxima detection.
    - percentile: Intensity threshold percentile.
    - mask: Optional binary mask to filter detected centroids.

    Returns:
    - coor: np.array of (x, y) coordinates of detected particles.
    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gaussian_filter(gray, sigma=10)
    threshold = np.percentile(blurred, percentile)
    local_max = (blurred == maximum_filter(blurred, size=3 * radius + 1))
    local_max = local_max & (blurred > threshold)
    labeled, num_features = label(local_max)
    centroids = center_of_mass(blurred, labeled, range(1, num_features + 1))

    if not centroids:
        return np.empty((0, 2))  # aucune détection
    if mask is not None:

        coor = np.array([(c[1], c[0]) for c in centroids if mask[int(round(c[0])), int(round(c[1]))] == 255])
    else:
        coor = np.array([(c[1], c[0]) for c in centroids])

    return coor # (x, y)


def binary_mask(image_mask, percentile=99,Otsu=False):
    """
    Creates a binary mask from the grayscale image.
    Pixels above the specified percentile are set to 255 (white),
    others are set to 0 (black).

    """
    if Otsu:
        _, threshold = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print(f" Otsu's Threshold: {threshold:.2f}")

    else:
        threshold = np.percentile(image_mask, percentile)
        print(f"Percentile Threshold: {threshold:.2f}")
    mask_bool = image_mask >= threshold  # dtype=bool
    binary = mask_bool.astype(np.uint8) * 255  # dtype=uint8
    kernel_size = (5, 5)  # Change the size as needed
    kernel = np.ones(kernel_size, np.uint8)
    # Apply dilation
    dilated_image = cv2.dilate(binary, kernel, iterations=3)
    cv2.imwrite("binary_mask.png", dilated_image)
    vals = np.unique(dilated_image)  # img : votre tableau NumPy avant affichage
    print(vals)
    return dilated_image

def detect_particles_cv(gray, percentile=80, mask=None):
    """
    Detects particles in a grayscale image using contour detection.

    Parameters:
    - gray: Grayscale input image (numpy array).
    - percentile: Intensity threshold percentile for binarization.
    - mask: Optional binary mask to filter detected centroids.

    Returns:
    - coor: np.array of (x, y) coordinates of detected particles.
    """
    #gray = gaussian_filter(image, sigma=12)
    threshold = np.percentile(gray, percentile)
    image_2 = np.where(gray > threshold, 255, 0).astype(np.uint8)  # Create a binary mask
    # Find contours in the binary mask
    contours, _ = cv2.findContours(image_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids_cv = []
    monent = []
    for cnt in contours:
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids_cv.append((cy, cx))# (row, col)
            monent.append(M)
    if mask is not None:

        coor = np.array([(c[1], c[0]) for c in centroids_cv if mask[int(round(c[0])), int(round(c[1]))] == 255])
    else:
        coor = np.array([(c[1], c[0]) for c in centroids_cv])

    return coor # (x, y)
def adaptive_threshold_cv(gray8, block_size=31, C=-6, gaussian=True):
    """
    gray8      : uint8 image (0-255)
    block_size : neighborhood width/height (odd)
    C          : constant subtracted from the local threshold
    gaussian   : True = weighted mean (Gaussian), False = arithmetic mean
    """
    if block_size % 2 == 0 or block_size < 3:
        raise ValueError("block_size doit être impair et ≥ 3")

    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if gaussian else cv2.ADAPTIVE_THRESH_MEAN_C
    binary = cv2.adaptiveThreshold(gray8, 255, method,
                                   cv2.THRESH_BINARY, block_size, C)
    return binary

def detect_particles_adapt(gray, radius_px=10, block_size=31, C=-5):
    # 1. Binaire adaptatif
    binary = adaptive_threshold_cv(gray, block_size, C, gaussian=True)

    # 2. Maxima locaux dans la zone blanche uniquement
    mask_max = (gray == maximum_filter(gray, size=2*radius_px+1)) & (binary > 0)

    labeled, n = label(mask_max)
    centroids = center_of_mass(gray, labeled, range(1, n+1))
    return np.array([(c[1], c[0]) for c in centroids])   # (x, y)


import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter, label, center_of_mass


def detect_particles_sef(gray, radius=3, percentile=90, sigma_log=3.0):
    """
    Particle detection based on SEF (LoG) filter + local maxima + centroids.
    Parameters:
    - gray: input image (grayscale)
    - radius: radius for local detection
    - percentile: global intensity threshold
    - sigma_log: standard deviation of the Laplacian-Gaussian filter
    Returns:
    - np.array of coordinates (x, y) of detected particles
    """
    filtered = -gaussian_laplace(gray.astype(float), sigma=sigma_log)
    threshold = np.percentile(filtered, percentile)
    mask_thresh = filtered > threshold
    local_max = (filtered == maximum_filter(filtered, size=2 * radius + 1))
    mask_local = mask_thresh & local_max
    labeled, num_features = label(mask_local)
    centroids = center_of_mass(filtered, labeled, range(1, num_features + 1))
    if not centroids:
        return np.empty((0, 2))
    # Conversion (y, x) → (x, y)
    return np.array([(c[1], c[0]) for c in centroids])

import numpy as np


def extract_alike_kpts(img):
    """Fallback keypoint extractor used when ALIKE is unavailable."""
    gray = np.mean(img, axis=2) if img.ndim == 3 else img
    gray = gray.astype(np.float32)
    if gray.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    corners = np.column_stack(np.nonzero(gray > gray.mean()))
    if len(corners) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return corners[:, [1, 0]].astype(np.int32)

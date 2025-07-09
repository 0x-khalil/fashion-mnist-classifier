import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(images, radius=1, n_points=8):
    """
    Converts a batch of images into LBP Histogram feature vectors.
    """
    lbp_features = []
    for img in images:
        #generate the LBP texture map
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')

        #create a histogram (8 points has 10 possible bins)
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

        #normalize
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        lbp_features.append(hist)

    return np.array(lbp_features)

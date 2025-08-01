import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.decomposition import PCA
def extract_lbp_features(images, radius=2, n_points=16):
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

def extract_hog_features(images):
    """
    Converts a batch of images into HOG feature vectors.
    """
    hog_features = []
    for img in images:

        fd = hog(img, orientations=12, pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2), visualize=False)
        hog_features.append(fd)

def extract_combined_features(images):
    """
    Extracts both LBP and HOG and stacks them into one large feature vector.
    """
    print("Extracting LBP...")
    lbp_feats = extract_lbp_features(images)
    print("Extracting HOG...")
    hog_feats = extract_hog_features(images)

    combined = np.hstack([lbp_feats, hog_feats])
    return combined

    return np.array(hog_features)

def apply_pca(train_features, test_features, variance_threshold=0.95):
    """
    Reduces the dimensionality of features while retaining a percentage of variance.
    """
    print(f"Applying PCA (retaining {variance_threshold*100}% variance)...")

    pca = PCA(n_components=variance_threshold, svd_solver='full')
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)

    print(f"Reduced feature count from {train_features.shape[1]} to {train_pca.shape[1]}")
    return train_pca, test_pca

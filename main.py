import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from src.features import extract_lbp_features
from src.features import extract_lbp_features, extract_hog_features
from src.model import get_xgb_model
from src.utils import plot_confusion_matrix # <--- New Import
from sklearn.metrics import accuracy_score, classification_report

def run_experiment(feature_mode="hog"): # <--- Added mode switch
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    if feature_mode == "lbp":
        print("--- Extracting LBP Features ---")
        X_train = extract_lbp_features(x_train)
        X_test = extract_lbp_features(x_test)
    else:
        print("--- Extracting HOG Features ---")
        X_train = extract_hog_features(x_train)
        X_test = extract_hog_features(x_test)

    print(f"--- Training XGBoost ({feature_mode.upper()}) ---")
    model = get_xgb_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n{feature_mode.upper()} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    plot_confusion_matrix(y_test, y_pred, class_names)

if __name__ == "__main__":
    run_experiment(feature_mode="hog")

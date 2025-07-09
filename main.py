import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from src.features import extract_lbp_features
from src.model import get_xgb_model
from src.utils import plot_confusion_matrix # <--- New Import
from sklearn.metrics import accuracy_score, classification_report

def run_experiment():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Fashion-MNIST Class Names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 2. Feature Extraction
    print("--- Extracting LBP Features ---")
    X_train_lbp = extract_lbp_features(x_train)
    X_test_lbp = extract_lbp_features(x_test)

    # 3. Train
    print("--- Training XGBoost ---")
    model = get_xgb_model()
    model.fit(X_train_lbp, y_train)

    # 4. Results & Visualization
    y_pred = model.predict(X_test_lbp)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Launch the plot
    plot_confusion_matrix(y_test, y_pred, class_names)

if __name__ == "__main__":
    run_experiment()

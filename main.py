import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from src.features import extract_lbp_features
from src.model import get_xgb_model
from sklearn.metrics import accuracy_score, classification_report

def run_experiment():
    print("--- Loading Fashion-MNIST ---")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #feature Extraction
    print("--- Extracting LBP Texture Features ---")
    X_train_lbp = extract_lbp_features(x_train)
    X_test_lbp = extract_lbp_features(x_test)
    # tain
    print("--- Training XGBoost Classifier ---")
    model = get_xgb_model()
    model.fit(X_train_lbp, y_train)
    # results
    y_pred = model.predict(X_test_lbp)
    print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_experiment()

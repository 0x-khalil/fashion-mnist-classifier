import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from src.features import extract_lbp_features, extract_hog_features
from src.model import get_xgb_model
from src.utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

def run_full_comparison():
    # 1. Load Data
    print("--- Loading Fashion-MNIST ---")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    experiments = ["lbp", "hog"]
    results = {}

    for mode in experiments:
        print(f"\n" + "="*30)
        print(f"RUNNING EXPERIMENT: {mode.upper()}")
        print("="*30)

        # 2. Extract Features
        if mode == "lbp":
            x_train_feat = extract_lbp_features(x_train)
            x_test_feat = extract_lbp_features(x_test)
        else:
            x_train_feat = extract_hog_features(x_train)
            x_test_feat = extract_hog_features(x_test)

        # 3. Train
        print(f"Training XGBoost on {x_train_feat.shape[1]} features...")
        model = get_xgb_model()
        model.fit(x_train_feat, y_train)

        # 4. Evaluate
        y_pred = model.predict(x_test_feat)
        acc = accuracy_score(y_test, y_pred)
        results[mode] = acc

        print(f"\n{mode.upper()} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # 5. Visualize
        plot_confusion_matrix(y_test, y_pred, class_names)

    # Final Summary
    print("\n" + "="*30)
    print("FINAL COMPARISON")
    print("="*30)
    for mode, score in results.items():
        print(f"{mode.upper()}: {score:.4f}")

if __name__ == "__main__":
    run_full_comparison()

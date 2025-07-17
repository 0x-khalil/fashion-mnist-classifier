import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from src.features import extract_lbp_features, extract_hog_features, extract_combined_features
from src.model import get_xgb_model
from src.utils import plot_confusion_matrix
from sklearn.metrics import accuracy_score

def run_full_comparison():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    experiments = ["lbp", "hog", "combined"]
    results = {}

    for mode in experiments:
        print(f"\n" + "="*40)
        print(f"STARTING EXPERIMENT: {mode.upper()}")
        print("="*40)

        if mode == "lbp":
            x_train_feat = extract_lbp_features(x_train)
            x_test_feat = extract_lbp_features(x_test)
        elif mode == "hog":
            x_train_feat = extract_hog_features(x_train)
            x_test_feat = extract_hog_features(x_test)
        else:
            x_train_feat = extract_combined_features(x_train)
            x_test_feat = extract_combined_features(x_test)

        # train
        print(f"Training XGBoost on {x_train_feat.shape[1]} features...")
        model = get_xgb_model()
        model.fit(x_train_feat, y_train)

        # eval
        y_pred = model.predict(x_test_feat)
        acc = accuracy_score(y_test, y_pred)
        results[mode] = acc

        print(f"Accuracy for {mode.upper()}: {acc:.4f}")
        plot_confusion_matrix(y_test, y_pred, class_names)

    # Leaderboard
    print("\n" + "!"*40)
    print("FINAL RESULTS LEADERBOARD")
    print("!"*40)
    for mode, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{mode.upper()}: {score:.4f}")

if __name__ == "__main__":
    run_full_comparison()

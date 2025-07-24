from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
def get_xgb_model():
    return XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            objective='multi:softmax',
            num_class=10,
            tree_method='hist',
            random_state=42,
            device="cuda"
        )
    def evaluate_with_kfold(X, y, k=5):
        """
        Performs Stratified K-Fold cross-validation and returns the scores.
        """
        model = get_xgb_model()
        # Stratified ensures each fold has 10% of T-shirts, 10% of Boots, etc.
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        print(f"Starting {k}-Fold Cross-Validation...")
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        return scores

from xgboost import XGBClassifier

def get_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=10,
        tree_method='hist',
        random_state=42
    )

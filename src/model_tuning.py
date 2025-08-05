from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

def tune_model(model, param_grid, X, y_raw, task_type):
    """
    Safe GridSearchCV with label encoding for classification and input cleaning
    """
    # Prepare target column
    if task_type == 'classification':
        # Clean string labels (strip & lowercase)
        y_raw = pd.Series(y_raw).astype(str).str.strip().str.lower()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # Save label map for decoding
        label_map = dict(zip(range(len(le.classes_)), le.classes_))
        joblib.dump(label_map, "assets/label_map.pkl")
    else:
        y = pd.to_numeric(y_raw, errors='coerce')
        y = y.dropna()
        X = X.loc[y.index]

    # Run GridSearchCV safely
    scoring = 'accuracy' if task_type == 'classification' else 'r2'
    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_score_, grid.best_params_
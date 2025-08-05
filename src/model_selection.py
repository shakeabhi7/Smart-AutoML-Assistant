import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

def auto_train_model(df, target_col):
    try:
        # Split X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features in X
        X = pd.get_dummies(X)

        # Task detection based on original target before encoding
        if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
            task_type = "classification"
        else:
            task_type = "regression"

        # Encode y only if classification
        if task_type == "classification":
            y=y.astype(str).str.strip().str.lower()
            encoded_y, label_map = pd.factorize(y.astype(str))
            y = encoded_y
        else:
            label_map=None # No need for mapping in regression

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model training
        if task_type == "regression":
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = {
                "Task": "Regression",
                "R² Score": round(r2_score(y_test, preds), 4),
                "MAE": round(mean_absolute_error(y_test, preds), 4)
            }

        else:
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Decode numeric predictions to original class labels
            decoded_preds = [label_map[p] for p in preds]
            metrics = {
                "Task": "Classification",
                "Accuracy": round(accuracy_score(y_test, preds), 4),
                "Classes Predicted": sorted(set(decoded_preds))
            }

        # Save model and metadata for prediction module
        joblib.dump(model,"assets/model.pkl")
        cat_cols = df.drop(columns=[target_col]).select_dtypes(include='object').columns.tolist()
        joblib.dump(cat_cols, "assets/cat_columns.pkl")
        joblib.dump(task_type, "assets/task_type.pkl")
        joblib.dump(X.columns.tolist(),"assets/train_columns.pkl")
        if label_map is not None:
            joblib.dump(label_map,"assets/label_map.pkl")
        return metrics

    except Exception as e:
        return {"Error": str(e)}
    
def auto_train_multiple_models(df, target_col):
    models = {}
    metrics_all = {}

    # 📂 Prepare Save Directory
    os.makedirs("assets/models", exist_ok=True)

    # 🧠 Split X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical features
    X = pd.get_dummies(X)

    # Task detection
    task_type = "classification" if y.dtype == 'object' or y.nunique() <= 10 else "regression"

    # Encode target if needed
    if task_type == "classification":
        y=y.astype(str).str.strip().str.lower()
        y_encoded, label_map = pd.factorize(y.astype(str))
        y = y_encoded
    else:
        label_map = None

    # Save label map + task type once
    joblib.dump(label_map, "assets/label_map.pkl")
    joblib.dump(task_type, "assets/task_type.pkl")
    joblib.dump(X.columns.tolist(), "assets/train_columns.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🔧 Define models
    if task_type == "classification":
        model_list = {
            "Random Forest Classifier": RandomForestClassifier(),
            "SVC": SVC(probability=True),
            "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }
    else:
        model_list = {
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR(),
            "XGBoost Regressor": XGBRegressor()
        }

    # 🔁 Train & save each model
    for name, model in model_list.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task_type == "classification":
            metric = {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, preds), 4)
            }
        else:
            metric = {
                "Model": name,
                "R² Score": round(r2_score(y_test, preds), 4),
                "MAE": round(mean_absolute_error(y_test, preds), 4)
            }

        model_key = name.replace(" ", "_").lower()  # eg. "random_forest"

        # 💾 Save individual model safely
        joblib.dump(model, f"assets/models/{model_key}_model.pkl")
        metrics_all[name] = metric

    return metrics_all


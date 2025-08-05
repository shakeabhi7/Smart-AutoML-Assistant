import pandas as pd
import joblib
import numpy as np

def run_prediction(input_df,original_columns,model,label_map=None,task_type=None):
    # Encode new data to match training structure
    df = input_df.copy()

    # Load original categorical column names
    try:
        cat_cols = joblib.load("assets/cat_columns.pkl")
    except:
        cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Ensure all categorical columns are string type
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = pd.get_dummies(df)
    df = df.reindex(columns=original_columns, fill_value=0)

    preds = model.predict(df)

    if task_type == "classification" and label_map is not None:

        decoded_preds = [label_map[p] for p in preds]
        return decoded_preds
    else:
        return preds

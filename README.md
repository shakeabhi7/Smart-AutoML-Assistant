# SmartAutoML Assistant 🚀

A Streamlit-powered AutoML tool for automated preprocessing, model selection, hyperparameter tuning, prediction, and visualization. Built to make machine learning workflows easy, scalable, and insightful.

## 📦 Features

- 🧠 Auto detection of classification/regression task
- 📊 Categorical encoding & preprocessing
- ⚙️ Hyperparameter tuning with GridSearchCV
- ✅ Supports RandomForest, SVM, XGBoost
- 🔄 Model persistence for re-use
- 🧯 Decoded predictions (e.g., 'fire', 'not fire')
- 📈 Visualization options (charts, metrics)
- 📁 Export predictions and model metadata

## install Dependencies
 - pip install -r requirements.txt

## Launch the App
 - streamlit run app.py

## 🧪 Usage Instructions

🔹 Tab 1: Upload CSV
- Upload your dataset
- Select target column

🔹 Tab 2: Explore Data
- View column types
- Check nulls or imbalance

🔹 Tab 3: Model Training
- Auto trains basic model for regression/classification

🔹 Tab 4: Prediction
- Upload new data for prediction
- Decoded output (e.g. 'fire', 'not fire') shown if applicable

🔹 Tab 5: Hyperparameter Tuning
- Choose model
- Run GridSearchCV
- Saves best model to assets/model.pkl

## 📎 Notes
- 🔒 Tuning requires target labels cleaned and encoded internally
- 🧠 label_map.pkl auto saved for decoding predictions
- ✅ Predictions use training-time columns and encoders for consistency


## developer - Abhishek (shakeabhi7)

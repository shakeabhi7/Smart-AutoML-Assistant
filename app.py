import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from src.predictor import run_prediction
from src.model_selection import auto_train_model
from src.model_tuning import tune_model
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
import xgboost as xgb
import time
# Page Config
st.set_page_config(
    page_title="Smart AutoML Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug Toggle

# debug_mode = st.sidebar.checkbox("Enable Debug Mode")

# Sidebar

# with st.sidebar:
#     st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Streamlit_logo.svg/512px-Streamlit_logo.svg.png", width=150)
#     st.markdown("## 📁 Upload Your Dataset")

    
#     uploaded_file = st.file_uploader("Choose a csv file",type=["csv"])

#     st.markdown("---")
#     if st.button("Force Refresh View"):
#         st.session_state.target_col = None
#         st.rerun()
#     st.markdown("## 💡Tip")
#     st.caption("Explore your dataset in EDA tab before training model")

# page Header
st.title("Smart AutoMl Pipeline  for Tabular Data")

top_col1, top_col2 = st.columns([5,1])

with top_col1:
    # st.image(
    #     "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%2Fid%2FOIP.8KTdQaQNILYK7LUW6KdzhAHaHa%3Fpid%3DApi&f=1&ipt=98322798999872e6040cbecda1a5e707a84611d8d169b5b1c590635fef8da77c&ipo=images",
    #     width=150
    # )
    st.markdown("## 📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    st.caption("💡 Tip: Explore your dataset in EDA tab before training model")

with top_col2:
    if st.button("🔄 Force Refresh View"):
        st.session_state.target_col = None
        st.rerun()



#Tabs Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(['EDA Assistant','ML Assistant','Prediction Panel','Model Comparison','Model Tuning Panel'])

#File Upload

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Succesfully!")

    # EDA tab
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10),use_container_width=True)

        st.info("You can scroll to access the Preview")

        if st.button("Analyze", key="eda_analyze"):
            st.subheader("📊 EDA Summary")
            st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.markdown("**Columns:**")
            st.write(list(df.columns))
            st.markdown("**Missing Values (per column):**")
            st.table(df.isnull().sum())


    # ML Tab
    with tab2:
        st.subheader("Model Setup Panel")

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [
            col for col in df.columns
            if df[col].dtype == 'object' or df[col].nunique()<=10
        ]
            
            
        # Target column selection using session_state
        target_column = st.selectbox(
            "🎯 Select Target Column",
            options=df.columns,
            key="target_col"
            )

        
        # Trigger model training only if target is selected
        if st.button("Run Analysis", key="ml_btn") and target_column:
            st.markdown("**Numerical Columns:**")
            st.write(numerical_cols)
            st.markdown("**Categorical Columns:**")
            st.write(categorical_cols)

            metrics = auto_train_model(df, target_column)

            if metrics:
                st.subheader("📊 Training Summary")
                for k, v in metrics.items():
                    st.markdown(f"**{k}:** {v}")
            else:
                st.error("🚨 Model training failed or no metrics returned.")
    with tab3:
        st.subheader("📂 Upload New Data for Prediction")
        pred_file = st.file_uploader("Upload CSV file for Prediction",type=["csv"],key="pred_upload")

        if pred_file:
            new_data = pd.read_csv(pred_file)
            st.success("New Data Uploaded")
            st.dataframe(new_data.head(),use_container_width=True)

            try:
                model = joblib.load("assets/model.pkl")
                task_type = joblib.load("assets/task_type.pkl")
                train_cols = joblib.load("assets/train_columns.pkl")
                try:
                    label_map = joblib.load("assets/label_map.pkl")
                except:
                    label_map = None
                
                if st.button("Run Prediction"):
                    results = run_prediction(new_data,train_cols,model,label_map,task_type)
                    
                    if len(results) == len(new_data):
                        new_data["Prediction"] = results
                        csv_buffer = io.StringIO()
                        new_data.to_csv(csv_buffer,index=False)
                        csv_bytes = csv_buffer.getvalue().encode()

                        st.download_button(
                            label = "📥 Download Predictions",
                            data = csv_bytes,
                            file_name = "smart_predictions.csv",
                            mime="text/csv"
                        )
                        st.subheader("🧠 Prediction Results")
                        st.dataframe(new_data,use_container_width=True)
                    else:
                        st.error("❌ Prediction failed: Output length mismatch")  
                    
                    # Prediction Visualization
                    st.subheader("📊 Prediction Visualization")

                    # chart_type = st.radio("Choose Chart Type:",["Bar Chart (Classification)", "Histogram (Regression)"])
                    # if chart_type == "Bar Chart (Classification)" and label_map is not None:
                    value_counts = pd.Series(results).value_counts()
                    fig, ax = plt.subplots()
                    sns.barplot(x=value_counts.index,y=value_counts.values,ax=ax)
                    ax.set_xlabel("Class")
                    ax.set_ylabel("Count")
                    ax.set_title("Predicted Class Distribution")
                    st.pyplot(fig)

                    # elif chart_type == "Histogram (Regression)" and label_map is None:
                    #     fig, ax = plt.subplots()
                    #     sns.histplot(results,kde=True,ax=ax)
                    #     ax.set_xlabel("Predicted Value")
                    #     ax.set_ylabel("Frequency")
                    #     ax.set_title("Prediction Value Histogram")
                    #     st.pyplot(fig)
                

            except Exception as e:
                st.error(f"Failed to load model or metadata: {e}")

    with tab4:
        st.subheader("📊 Compare Multiple Models")

        if uploaded_file:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("🚨 File appears empty or incorrectly formatted.")
                st.stop()
            except Exception as e:
                st.error(f"🚨 Failed to read uploaded file: {e}")
                st.stop()

            target_col = st.session_state.get("target_col")

            if target_col:
                from src.model_selection import auto_train_multiple_models
                metrics_all = auto_train_multiple_models(df, target_col)

                # Display comparison table
                st.markdown("### 🧠 Model Performance Summary")
                st.table(pd.DataFrame(metrics_all).T)

                # Let user choose model to activate for prediction
                selected_model = st.selectbox("✅ Select Model to Use", list(metrics_all.keys()))
                model_key = selected_model.replace(" ", "_").lower()

                if st.button("🔐 Use Selected Model"):
                    try:
                        import shutil

                        # Paths
                        model_path = f"assets/models/{model_key}_model.pkl"

                        # 🔁 Copy selected model to primary slot
                        shutil.copy(model_path, "assets/model.pkl")

                        st.success(f"✅ '{selected_model}' is now set for prediction panel!")

                    except Exception as e:
                        st.error(f"🚨 Failed to apply selected model: {e}")

            else:
                st.warning("🎯 Please select a target column first in the ML Assistant tab.")
        else:
            st.warning("📁 Upload a CSV file to begin model comparison.")

    with tab5:
        st.subheader("Hyperparameter Tuning")

        model_choice = st.selectbox("Select Model to tune",['RandomForest','SVM','XGBoost'])
        target_col = st.session_state.get("target_col")

        if uploaded_file and target_col and st.button("Run GridSearchCV"):
            from sklearn.preprocessing import LabelEncoder
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            X_raw = df.drop(columns=[target_col])
            X = pd.get_dummies(X_raw)
            
            #Auto task detection
            y_raw = df[target_col]
            task_type = "classification" if y_raw.dtype == "object" or y_raw.nunique() <= 10 else "regression"

            # safe encoding
            if task_type == "regression":
                y = pd.to_numeric(df[target_col],errors='coerce')
                y=y.dropna()
                X=X.loc[y.index]
            else:
                y_raw = y_raw.astype(str).str.strip().str.lower()
                le = LabelEncoder()
                y = le.fit_transform(y_raw)
                label_map = dict(zip(range(len(le.classes_)), le.classes_))
                joblib.dump(label_map, "assets/label_map.pkl")  # Save for prediction decoding
 


            if model_choice == "RandomForest":
                model = RandomForestClassifier() if task_type == "classification" else RandomForestRegressor()
                param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
            
            elif model_choice == "SVM":
                model = SVC() if task_type == "classification" else SVR()
                param_grid = {'C' : [0.5,1.0], 'kernel':['linear','rbf']}
            else:
                model = xgb.XGBClassifier() if task_type == "classification" else xgb.XGBRegressor()
                param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}

            #Tuning Start
            start = time.time()
            best_model, best_score, best_params = tune_model(model,param_grid,X,y,task_type)
            end = time.time()

            st.success(f"Best Score : {round(best_score,4)}")
            st.info(f"🕒 Training Time: {round(end - start, 2)}s")
            st.write("📋 Best Parameters:")
            st.json(best_params)

            # Save tuned model
            joblib.dump(best_model, "assets/model.pkl")
            joblib.dump(X.columns.tolist(), "assets/train_columns.pkl")
            joblib.dump(task_type, "assets/task_type.pkl")

            # if task_type == "classification":
            #     label_map = {i: label for i, label in enumerate(y.unique())}
            #     joblib.dump(label_map, "assets/label_map.pkl")

            # st.success("🎯 Tuned model saved and ready for prediction!")



            
        


else:
    with tab1:
        st.info("Upload a Dataset to begin EDA")
    with tab2:
        st.info("Upload a Dataset to enable ML Pipeline")
    with tab3:
        st.info("Upload the Dataset for Prediction")
    # with tab4:
    #     st.info("Uplaod The Dataset for Model Comparison")
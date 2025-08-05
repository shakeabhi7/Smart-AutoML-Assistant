import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda():
    st.subheader("Exploratory Data Analysis")
    
    uploaded_file = st.file_uploader("Choose a file",types=["csv","xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview",df.head())

        st.write("Shape",df.shape)
        st.write("Null Value",df.isnull().sum())
        st.write("Data Types",df.dtypes)

        if st.checkbox("Show the Summary Stats"):
            st.write(df.describe())
        if st.checkbox("Correlation Heatmap"):
            st.write("Generating heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True,cmap='coolwarm',ax=ax)
            st.pyplot(fig)

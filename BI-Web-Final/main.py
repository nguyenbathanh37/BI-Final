import streamlit as st
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from joblib import load
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import plotly.express as px

st.markdown("# Predict Credit Status")
st.sidebar.markdown("# Predict Credit Status")


uploaded_file = st.file_uploader("Upload file (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Upload Success:")
    st.write(df)

    # handle data upload
    # encode binary features
    df["CODE_GENDER"] =  df["CODE_GENDER"].replace(['F','M'],[0,1])
    df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].replace(["Y","N"],[1,0])
    df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].replace(["Y","N"],[1,0])

    # df.drop('ID', axis=1, inplace=True)

    names = df.keys()
    types = df.dtypes
    print('222')
    model1 = tree.DecisionTreeClassifier()
    model1 = joblib.load('../dt.joblib')
    print('111')

    le = model1['label_encoder']
    # le = LabelEncoder()
    for i in range(len(types)):
        if types[i] == 'object':
            # Chuyển đổi cột thành chuỗi nếu nó không phải kiểu chuỗi
            df[names[i]] = df[names[i]].astype(str)
            
            le.fit(df[names[i]])
            df[names[i]] = le.transform(df[names[i]])

    data = np.array(df.values)
    X_test = data

    # scaler = MinMaxScaler()
    scaler = model1['StandardScaler']
    scaler.fit(X_test)
    X_scaled = scaler.transform(X_test)

    st.sidebar.header("Select Machine Learning Model")
    model_option = st.sidebar.selectbox("Select Machine Learning Model:", ["", "Decision Tree", "Logistic Regression", "KNN", "SVM"])
    if model_option == "Decision Tree":

        model = tree.DecisionTreeClassifier()
        model = load('../dt.joblib')


        y_pred = model['model'].predict(X_scaled)

        st.markdown("Data after prediction")


        df['STATUS'] = y_pred
        df["STATUS"] =  df["STATUS"].replace([0,1,2,3,4,5,6,7],['X','C', '0', '1', '2', '3', '4', '5'])

        st.write(df)

        count_data = df['STATUS'].value_counts().reset_index()
        count_data.columns = ['Status', 'Count']

        st.write(count_data)

        # Tạo biểu đồ cột từ DataFrame mới
        fig = px.bar(count_data, x='Status', y='Count', title='Bar Chart of Status')

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

    if model_option == "Logistic Regression":

        # Tạo mô hình
        model = LogisticRegression()
        model = load('../logistic.joblib')
        # model.set_params(max_depth=max_depth)


        y_pred = model['model'].predict(X_scaled)

        st.markdown("Data after prediction")


        df['STATUS'] = y_pred
        df["STATUS"] =  df["STATUS"].replace([0,1,2,3,4,5,6,7],['X','C', '0', '1', '2', '3', '4', '5'])

        st.write(df)

        count_data = df['STATUS'].value_counts().reset_index()
        count_data.columns = ['Status', 'Count']

        st.write(count_data)

        # Tạo biểu đồ cột từ DataFrame mới
        fig = px.bar(count_data, x='Status', y='Count', title='Bar Chart of Status')

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

    if model_option == "KNN":

        model = joblib.load('../knn_model.joblib')


        y_pred = model['model'].predict(X_scaled)

        st.markdown("Data after prediction")


        df['STATUS'] = y_pred
        df["STATUS"] =  df["STATUS"].replace([0,1,2,3,4,5,6,7],['X','C', '0', '1', '2', '3', '4', '5'])

        st.write(df)

        count_data = df['STATUS'].value_counts().reset_index()
        count_data.columns = ['Status', 'Count']

        st.write(count_data)

        # Tạo biểu đồ cột từ DataFrame mới
        fig = px.bar(count_data, x='Status', y='Count', title='Bar Chart of Status')

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

    if model_option == "SVM":

        model = joblib.load('../svm_model.joblib')


        y_pred = model['model'].predict(X_scaled)

        st.markdown("Data after prediction")


        df['STATUS'] = y_pred
        df["STATUS"] =  df["STATUS"].replace([0,1,2,3,4,5,6,7],['X','C', '0', '1', '2', '3', '4', '5'])

        st.write(df)

        count_data = df['STATUS'].value_counts().reset_index()
        count_data.columns = ['Status', 'Count']

        st.write(count_data)

        # Tạo biểu đồ cột từ DataFrame mới
        fig = px.bar(count_data, x='Status', y='Count', title='Bar Chart of Status')

        # Hiển thị biểu đồ trong Streamlit
        st.plotly_chart(fig)

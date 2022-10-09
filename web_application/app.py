#importing required packages

import streamlit as st

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from PIL import Image


#reading and preprocessing data
train_df = pd.read_csv('data/train.csv', index_col='Unnamed: 0')
test_df = pd.read_csv('data/test.csv', index_col='Unnamed: 0')

train_data = train_df.copy()
test_data = test_df.copy()

#dropping NA values
train_data = train_data.dropna()
test_data = test_data.dropna()

#Encoding categorical data into numeric
numerics = ['int64', 'float64']
train_con_col = train_data.select_dtypes(include = numerics).columns
train_cat_col = train_data.select_dtypes(include = "object").columns
test_con_col = test_data.select_dtypes(include = numerics).columns
test_cat_col = test_data.select_dtypes(include = "object").columns

for cat in train_cat_col:
    le = LabelEncoder()
    train_data[cat] = le.fit_transform(train_data[cat])
    test_data[cat] = le.fit_transform(test_data[cat])


#web part
st.title("Airline passenger satisfaction :airplane:")

header_image = Image.open('./web_application/images/HeaderImage-1.png')
st.image(header_image, caption = "source: https://www.airlines.org/")

st.markdown("### Intro :pencil:")

st.markdown("Airlines are one of the most famous types of transportation. It is fast, luxuries and at some kind romantic. "
        "However, not each flight might be satisfying for passengers. Therefore, airline companies gather feedback of the customers "
        "in order to improve their service, increase their income and decrease the amount of complains from the clients.")

st.markdown("In this web application you will see a research based on the public dataset in which I will try to investigate "
            "which factors impact customers satisfaction. ")

st.markdown("As a bonus, you will have a chance to simulate a response from the "
            "client and see a prediction of the trained model.")

st.markdown("### Dataset")
st.markdown("Let's get closer to the dataset. It was obtained from the Kaggle competition "
            "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction. "
            "In a nutshell, it contains different categories of services provided by airline company as well as some data"
            "about passenger such as age, for example.")

st.markdown("You can get detailed information about variables below")
with st.expander("Detailed info"):
    with open("data/variables.txt") as variables_txt:
        #contents = variables_txt.readlines()
        for line in variables_txt:
            st.markdown(line)

    st.markdown("#### Example of dataset entries:")
    st.write(train_data.head())

st.markdown("Our goal is to predict Satisfaction level. There are two levels: 'neutral or dissatisfied' and 'satisfied'.")

st.markdown("### Model")
st.markdown("In this project it was used an XGBoost algorithm. After feature selection process model managed to get 95% "
            "of accuracy.")


st.markdown("### Bonus part")
st.markdown("Here you are proposed to generate answers of imaginary client and to see which feedback he or she will leave.")

customer_type = st.selectbox(label="Select customer type",
                             options=pd.unique(train_df['Customer Type']))

type_of_travel = st.selectbox(label="Select type of travel",
                             options=pd.unique(train_df['Type of Travel']))

class_ = st.selectbox(label="Select class",
                             options=pd.unique(train_df['Class']))


inflight_wifi_service = st.selectbox(label="Choose level of wifi service",
                             options=range(6))

online_boarding = st.selectbox(label="Choose level of online boarding",
                             options=range(6))

inflight_entertainment = st.selectbox(label="Choose level of inflight entertainment",
                             options=range(6))

baggage_handling = st.selectbox(label="Choose level of baggage handling",
                             options=range(6))


new_X = np.array([customer_type, type_of_travel, class_, inflight_wifi_service,
                 online_boarding, inflight_entertainment, baggage_handling])

with st.sidebar:
    sidebar_image = Image.open('./web_application/images/photo_linkedin.png')
    st.image(sidebar_image)
    st.markdown("# Hi!")
    st.markdown("This is a Data science pet project of Ihor Hetman. I hope that you will enjoy it!")
    st.markdown("If you want to contact me, you can do that via several sources:")
    st.markdown("https://www.linkedin.com/in/ihor-hetman/")
    st.markdown("ihorhetman99@gmail.com")
    st.caption("You can close the sidebar in order to stay focused :)")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Airline satisfation",
                   page_icon=":airplane:",
                   layout="wide")



@st.cache(allow_output_mutation=True)
def read_data():
    train_df = pd.read_csv('data/train.csv', index_col='Unnamed: 0')
    test_df = pd.read_csv('data/test.csv', index_col='Unnamed: 0')
    return train_df, test_df


@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('model/model.joblib')
    return model


# reading and preprocessing data
train_df, test_df = read_data()

train_df["Customer Type"] = train_df["Customer Type"].replace("disloyal Customer", "Disloyal Customer")
test_df["Customer Type"] = test_df["Customer Type"].replace("disloyal Customer", "Disloyal Customer")

train_df["Type of Travel"] = train_df["Type of Travel"].replace("Personal Travel", "Personal travel")
test_df["Type of Travel"] = test_df["Type of Travel"].replace("Personal Travel", "Personal travel")


# web part


st.title("Airline passenger satisfaction :airplane:")

header_image = Image.open('./web_application/images/HeaderImage-1.png')
st.image(header_image, caption="source: https://www.airlines.org/")

st.markdown("### Intro :pencil:")

st.markdown(
    "Airlines are one of the most famous types of transportation. It is fast, luxuries and at some kind romantic. "
    "However, not each flight might be satisfying for passengers. Therefore, airline companies gather feedback of the customers "
    "in order to improve their service, increase their income and decrease the amount of complains from the clients.")

st.markdown(
    "In this web application you can see the research results based on the public dataset. In this research I "
    "investigated which factors impact customers satisfaction.")

st.markdown("You will have a chance to try yourself as an owner of an airline company and simulate a response from the "
            "client to see a prediction of the trained model.")

st.markdown("### Dataset :open_book:")
st.markdown("Let's get closer to the dataset. It was obtained from the Kaggle competition "
            "https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction. "
            "In a nutshell, it contains different services categories provided by airline company and some "
            "data about passenger (e.g., age).")

st.markdown("You can get detailed information about variables below")
with st.expander("Detailed info"):
    with open("data/variables.txt") as variables_txt:
        for line in variables_txt:
            st.markdown(line)

    st.markdown("#### Example of dataset entries:")
    st.write(train_df.head())

st.markdown("Our goal is to predict **Satisfaction level**. There are two levels: 'Neutral or dissatisfied' and "
            "'Satisfied'.")

st.markdown("### Model :desktop_computer:")
st.markdown("In this project it was used a Random forest algorithm. After a backward feature selection process,"
            "model managed to get next values of metrics performance:")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", 0.903)
col2.metric("Precision", 0.889)
col3.metric("Recall", 0.889)

with st.expander("Feature selection process info"):
    st.markdown("Backward feature selection process was done based on exploratory data analysis + using feature "
                "importance of the Random forest model.")
    initial_set_features = Image.open('./web_application/images/initial_set_features.png')
    used_features_model = Image.open('./web_application/images/used_features_model.png')
    col1, col2, col3 = st.columns(3)
    col1.image(initial_set_features)
    col2.image(Image.open('./web_application/images/right-arrow.png'), width=200)
    col3.image(used_features_model)


st.markdown("### Predicting part :zap:")
st.markdown("Here you are proposed to generate answers of imaginary client and to see which feedback he or she will "
            "leave.")

# loading model
model = load_model()

st.markdown("##### Instructions:")
st.markdown("1. Choose values from the form below.")
st.markdown("2. Press 'Submit new input' button.")
st.markdown("3. Press 'Predict satisfaction level' button.")

# creating a new input

score_options = [1, 2, 3, 4, 5]
with st.form(key='input_form'):
    type_of_travel = st.selectbox(label="Select type of travel",
                                  options=pd.unique(train_df['Type of Travel']))
    class_ = st.selectbox(label="Select class",
                          options=pd.unique(train_df['Class']))
    flight_distance = st.number_input(label="Choose flight distance (you can write it down)",
                                      min_value=30, max_value=5000)
    inflight_wifi_service = st.selectbox(label="Choose level of wifi service",
                                         options=score_options)
    online_boarding = st.selectbox(label="Choose level of online boarding",
                                   options=score_options)
    inflight_entertainment = st.selectbox(label="Choose level of inflight entertainment",
                                          options=score_options)

    new_X = np.array([type_of_travel, class_, flight_distance, inflight_wifi_service,
                      online_boarding, inflight_entertainment, ])

    # encoding to numeric type
    encoding = {'Eco Plus': 2,
                'Business': 0,
                'Eco': 1,
                'Personal travel': 1,
                'Business travel': 0}

    for key in encoding.keys():
        if key in new_X:
            new_X[new_X == key] = encoding[key]

    new_X = np.array([new_X], dtype=object)

    submitted = st.form_submit_button('Submit new input', help='To create a new input, press this button')
    if submitted:
        st.success("Input is created")

if st.button('Predict satisfaction level', help='To make a prediction, press this button.'):
    y_pred = model.predict(new_X)
    y_pred_proba = model.predict_proba(new_X)
    if y_pred == 0:
        txt_pred = 'Neutral or dissatisfied'
    else:
        txt_pred = 'Satisfied'

    st.info('###### Prediction: {}'.format(str(txt_pred)))

st.write("")

features_rf = ['Type of Travel',
               'Class',
               'Flight Distance',
               'Inflight wifi service',
               'Online boarding',
               'Inflight entertainment',
               ]

st.markdown("To understand how much impact does a feature in the model, you can take a look at their importance.")

feat_imp = pd.DataFrame({'feature name ': features_rf, 'feature importance': model['rf'].feature_importances_})
feat_imp_plot = plt.figure(figsize=(12, 2.5))
plt.bar(feat_imp.iloc[:, 0], feat_imp.iloc[:, 1])
plt.xticks(rotation=45)
plt.title("Feature importance of variables, measured in %")
st.pyplot(feat_imp_plot)

st.success("### That is it. I hope that you enjoyed this little project. :smile:")

# sidebar
with st.sidebar:
    sidebar_image = Image.open('./web_application/images/photo_linkedin.png')
    st.image(sidebar_image)
    st.markdown("# Hi! :wave:")
    st.markdown("This is a Data science pet project made by Ihor Hetman. Enjoy using my app!")
    st.markdown("Please contact me by following links:")
    st.markdown("https://www.linkedin.com/in/ihor-hetman/")
    st.markdown("ihorhetman99@gmail.com")
    st.caption("You can close the sidebar in order to stay focused :)")

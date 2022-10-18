from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib


def read_data():
    train_df = pd.read_csv('data/train.csv', index_col='Unnamed: 0')
    test_df = pd.read_csv('data/test.csv', index_col='Unnamed: 0')
    return train_df, test_df


def get_metrics_scores(y_true, y_pred):
    print('accuracy: ', accuracy_score(y_true, y_pred))
    print('precision: ', precision_score(y_true, y_pred))
    print('recall: ', recall_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


# reading and preprocessing data
train_df, test_df = read_data()

train_data = train_df.copy()
test_data = test_df.copy()

# dropping NA values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Encoding categorical data into numeric
numerics = ['int64', 'float64']
train_con_col = train_data.select_dtypes(include=numerics).columns
train_cat_col = train_data.select_dtypes(include="object").columns
test_con_col = test_data.select_dtypes(include=numerics).columns
test_cat_col = test_data.select_dtypes(include="object").columns

for cat in train_cat_col:
    le = LabelEncoder()
    train_data[cat] = le.fit_transform(train_data[cat])
    test_data[cat] = le.fit_transform(test_data[cat])

# creating train and test sets
features_rf = ['Type of Travel',
               'Class',
               'Flight Distance',
               'Inflight wifi service',
               'Online boarding',
               'Inflight entertainment',
               ]

X_train = train_data[features_rf]
y_train = train_data.iloc[:, -1].to_numpy()

X_test = test_data[features_rf]
y_test = test_data.iloc[:, -1].to_numpy()


model = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestClassifier())])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
get_metrics_scores(y_pred, y_test)

#saving the model
joblib.dump(model, filename='model\model.joblib', compress = 3)

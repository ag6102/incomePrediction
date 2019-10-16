import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Training data
dataset = pd.read_csv('training_data.csv',low_memory=False)

# Data for which prediction needs to be done
dataset1 = pd.read_csv('test_data.csv',low_memory=False)
dataset.isnull().any()
dataset1.isnull().any()

#back fill
dataset = dataset.fillna(method='bfill')
#front fill
dataset = dataset.fillna(method='ffill')

#back fill
dataset1 = dataset1.fillna(method='bfill')
#front fill
dataset1 = dataset1.fillna(method='ffill')

# Label encoding
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])

for column in dataset1.columns:
    if dataset1[column].dtype == type(object):
        le = LabelEncoder()
        dataset1[column] = le.fit_transform(dataset1[column])

# To remove outliers - Increasing RMSE in Kaggle but in local RMSE is lowered to 93252.08
# z = np.abs(stats.zscore(dataset))
# dataset = dataset[(z < 3).all(axis=1)]
# print(dataset.shape);
# X_feature = ['Year of Record','Gender','Age','Country','Size of City','Profession','University Degree','Wears Glasses','Hair Color','Body Height [cm]']
# y_feature = ['Income in EUR']

# For fetching values from dataset to X and y
# X = dataset[X_feature].values     
# y = dataset[y_feature].values

# Implementing Random Forest Classifier
a,b = make_classification(n_samples=1000, n_features=12,
n_informative=2, n_redundant=0,
random_state=0, shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(a, b)

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(a.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(a.shape[1]), indices)
plt.xlim([-1, a.shape[1]])
plt.show()

# Fetching values of X_train, y_traina nd X_test
X_train = dataset[X_feature].values    
y_train = dataset[y_feature].values
X_test = dataset1[X_feature].values    

# Spliting test data in training and validation data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting values for test data
y_pred = regressor.predict(X_test)

# To import predictions to file
output = {'Instance': dataset1['Instance'].values,
        'Income': y_pred
        }
df = pd.DataFrame(output, columns= ['Instance', 'Income'])
export_csv = df.to_csv ('/Users/tushargarg/Desktop/aashimacollege/machinelearning/Kaggle/income_output.csv',index = None)

# Metrics when predicting with validation data
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
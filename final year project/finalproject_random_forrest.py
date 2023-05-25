

# Importing libraries

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/rajib/Downloads/new/final year project/DATASET FOR NEW .csv')


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = '/home/rajib/Downloads/new/final year project/RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()




import pickle
import numpy as np

def predict_with_random_forest(model_path):
    # Load the trained Random Forest model
    with open('/home/rajib/Downloads/new/final year project/RandomForest.pkl', 'rb') as file:
        model = pickle.load(file)
    # Define the feature names
    feature_names = ['N', 'P', 'K', 'Temperature','Humidity','pH','Rainfall']
    # Get input values from the user
    input_features = []
    for feature_name in feature_names:
        feature = float(input(f"Enter value for {feature_name}: "))
        input_features.append(feature)

    # Convert the input features to a numpy array
    input_features = np.array(input_features).reshape(1, -1)

    # Make predictions using the input features
    predictions = model.predict(input_features)

    return predictions

model_path = '/home/rajib/Downloads/new/final year project/RandomForest.pkl'  # Replace with the path to your trained model file

predictions = predict_with_random_forest(model_path)

print("The recommended crop is :",predictions)


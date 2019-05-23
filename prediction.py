# ---------------------------------------------------- #
# Import classes and functions
# ---------------------------------------------------- #
import numpy as np
import pandas 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ---------------------------------------------------- #
# load and preprocessing dataset
# ---------------------------------------------------- #
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
import os
dataframe = pandas.read_csv("test.csv", header=None)
dataset = dataframe.values
X = dataset.astype(float)

exists = os.path.isfile('scaler.pkl')
if exists:
	scaler = joblib.load('scaler.pkl')
else:
	sys.exit('Error: scaler not found.')

standardized_X = scaler.transform(X)

# ---------------------------------------------------- #
# load and print model
# ---------------------------------------------------- #
from keras.models import load_model
import sys

exists = os.path.isfile('model.h5')
if exists:
	model = load_model('model.h5')
else:
	sys.exit('Error: model not found.')

print(model.summary())

# ---------------------------------------------------- #
# perform prediction
# ---------------------------------------------------- #

predictions = model.predict_classes(standardized_X)
probability = model.predict_proba(standardized_X)

#print(predictions)
#print(pandas.DataFrame(probability).max(axis=1))

d=pandas.concat([pandas.DataFrame(predictions), pandas.DataFrame(probability).max(axis=1)], axis=1, ignore_index=True)
print(d)

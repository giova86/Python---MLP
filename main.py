# ---------------------------------------------------- #
# Import classes and functions
# ---------------------------------------------------- #
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ---------------------------------------------------- #
# fix random seed for reproducibility
# ---------------------------------------------------- #
seed = 7
np.random.seed(seed)

# ---------------------------------------------------- #
# load dataset
# ---------------------------------------------------- #
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 
import os
dataframe = pandas.read_csv("dataset/iris.csv", header=None)
dataset = dataframe.values

from sklearn.model_selection import train_test_split
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)

exists = os.path.isfile('scaler.pkl')
if exists:
	scaler = joblib.load('scaler.pkl')
else:
	scaler = StandardScaler().fit(X_train)
	joblib.dump(scaler, 'model/scaler.pkl') 


standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)

# ---------------------------------------------------- #
# encode the output variable
# ---------------------------------------------------- #
encoder = LabelEncoder()						# encode class values as integer
encoder.fit(Y_train)
encoder_Y = encoder.transform(Y_train)
dummy_y = np_utils.to_categorical(encoder_Y)	# convert integers to dummy variables

encoder.fit(Y_train)
encoder_Y = encoder.transform(Y_test)
dummy_y_test = np_utils.to_categorical(encoder_Y)	# convert integers to dummy variables

# ---------------------------------------------------- #
# define neural network
# ---------------------------------------------------- #
# number of features
input_dimension = X_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=input_dimension, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	
	# Compile model
	model.compile(
		loss='categorical_crossentropy', 
		optimizer='adam', 
		metrics=['accuracy']
		)
	return model

# ---------------------------------------------------- #
# define estimator and plot network structure
# ---------------------------------------------------- #

model = baseline_model()

print()
print(model.summary())
print()

history = model.fit(
	x=standardized_X,
	y=dummy_y,
	epochs=430,
	verbose=1,
	batch_size=64,
	validation_data=(standardized_X_test, dummy_y_test)		
	)

# ---------------------------------------------------- #
# define estimator and plot network structure
# ---------------------------------------------------- #
score = model.evaluate(standardized_X_test, dummy_y_test)
print()
print('-----------------------------------')
print('------------ RESULTS --------------')
print('-----------------------------------')
print('Loss: \t\t',score[0])
print('Accuracy: \t',score[1])


# ---------------------------------------------------- #
# plot accuracy and loss history
# ---------------------------------------------------- #
# summarize history for accuracy
f1 = plt.figure(1, figsize=(8, 9))

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy', fontsize=20)
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='lower right')
#plt.plot()

# summarize history for loss
#f2 = plt.figure(2)
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', fontsize=20)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.legend(['Train', 'Test'], loc='upper right')

plt.subplots_adjust(
	#top=0,
	#bottom=0.1,
	hspace=0.3,
	top=0.95,
	bottom=0.08,
	right=0.95,
	left=0.1)

plt.savefig('plots/acc_loss.pdf', format= 'pdf')
plt.savefig('plots/acc_loss.png', format= 'png')
#plt.show()

# ---------------------------------------------------- #
# save/load models
# ---------------------------------------------------- #
from keras.models import load_model
model.save('model/model.h5')


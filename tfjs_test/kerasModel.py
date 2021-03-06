from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from keras.layers.convolutional import Conv2D
import numpy as np
import pandas as pd
import tensorflowjs as tfjs

train_data=pd.read_csv('train_afterchange.csv')
test_data=pd.read_csv('test_afterchange.csv')

X_train=train_data.iloc[:1058,1:-1]
X_valid=train_data.iloc[1058:,1:-1]
Y_train=train_data.iloc[:1058,-1]
Y_valid=train_data.iloc[1058:,-1]
X_test=test_data.iloc[:,1:]

X_=X_train.as_matrix()
Y_=np.array([Y_train])
X_valid=X_valid.as_matrix()
Y_valid=np.array([Y_valid])
X_test_=X_test.as_matrix()
# print(X_)
# print(Y_)

sx = np.column_stack([X_,  np.zeros((1058,1))])
sx_valid=np.column_stack([X_valid,  np.zeros((400,1))])
sx_test=np.column_stack([X_test_,  np.zeros((1459,1))])
Y_=Y_.T
Y_valid=Y_valid.T

model=Sequential(name='house_price_prediction')
model.add(BatchNormalization(input_shape=(400,)))
model.add(Reshape((400,1,1)))
model.add(Conv2D(filters=32, strides=1, padding='same', kernel_size=1, activation='relu'))
model.add(Conv2D(filters=64, strides=1, padding='same', kernel_size=1, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.summary()

history=model.fit(sx,Y_,batch_size=8,epochs=100,verbose=1,validation_data=(sx_valid,Y_valid))
# print(history.history)

tfjs.converters.save_keras_model(model, './keras_model')

# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# import numpy as np
# import tensorflowjs as tfjs

# X = np.array([[i] for i in range(1000)])
# y = 2*X

# model = Sequential()
# model.add(Dense(units=1, activation='linear', input_shape=[1]))
# model.add(Dense(units=512, activation='linear'))
# model.add(Dropout(0.5))
# model.add(Dense(units=128, activation='linear'))
# model.add(Dense(units=128, activation='linear'))
# model.add(Dense(units=1, activation='linear'))

# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# model.summary()

# model.fit(X, y, epochs=45)

# tfjs.converters.save_keras_model(model, './keras_model')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
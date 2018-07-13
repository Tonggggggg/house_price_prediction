from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from keras.layers.convolutional import Conv2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
history=model.fit(sx,Y_,batch_size=8,epochs=400,verbose=1,validation_data=(sx_valid,Y_valid))
print(history.history)
print(history.history['loss'])
print(history.history['val_loss'])

fig = plt.figure(figsize=(20, 3))
axes = fig.add_subplot(1, 1, 1)
line1, = axes.plot(range(len(history.history['loss'])),history.history['loss'], 'r', label=u'train_loss', linewidth=2)
line2, = axes.plot(range(len(history.history['val_loss'])), history.history['val_loss'], 'g', label=u'valid_loss')
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line2])
plt.title('loss')
plt.show()
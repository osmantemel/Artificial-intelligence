import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten #dense katmanı yoğunluk katmanıdır,
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() #X bağımsız y bağımlı yaptık
print(X_train.shape)
print(X_test.shape)

temp = []
for i in range(len(Y_train)):
    temp.append(to_categorical(Y_train[i], num_classes = 10))
Y_train = np.array(temp)

temp = []    
for i in range(len(Y_test)):
    temp.append(to_categorical(Y_test[i], num_classes = 10))
Y_test = np.array(temp)

model = Sequential()
model.add(Flatten(input_shape= (28,28)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(X_train, Y_train, epochs = 15, validation_data = (X_test, Y_test))
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis = 1)
fig, axes = plt.subplots(ncols=10, sharex=False, sharey=True, figsize=(20, 4))

for i in range(10):
    axes[i].set_title(predictions[i])
    axes[i].imshow(X_test[i], cmap = 'gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()

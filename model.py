import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator	#To apply data augmentation

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print(len(x_train)) 
##print(x_train.shape)
#print(x_train[0])
#print(x_test.shape[0])

batch_size = 128
#num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape = input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

#hist = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

#trying with fit_generator and applying data augmentation..

#datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.10, fill_mode = "nearest", shear_range = 0.10, horizontal_flip = False, width_shift_range = 0.1, height_shift_range = 0.1)
datagen = ImageDataGenerator(rotation_range = 0.10, zoom_range = 0.0, fill_mode = "constant", cval = 3, shear_range = 0.0, vertical_flip = False, horizontal_flip = False, brightness_range = (0.5,1.5))
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), validation_data = (x_test, y_test), steps_per_epoch = len(x_train)//batch_size, epochs = epochs)

print("Model trained")

model.save('mnist1.h5')

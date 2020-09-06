import numpy as np
from skimage import color, exposure, transform
from skimage import io
from keras.preprocessing.image import *
import os
import glob
import cv2
import sys
import random





NUM_CLASSES = 3 #number of signs

IMG_SIZE1 = 240
IMG_SIZE2 = 320

#size after cropping. Remember, also need to go specify how much in "image.crop"
CRP_SIZE1=63
CRP_SIZE2=224

BATCH_SIZE = 32


"""
class IDs in main.py
0: "No sign",
1: "Fork right",
2: "Fork left",


Pass in the root directory of your log file as an argument, or use ./ with no arg passed
Make your folder structure like this and then stick the images for each class in the correct folder
\root_dir\0\
\root_dir\1\
\root_dir\2\
\root_dir\3\
...

"""

def preprocess_img(img, lbl):
    image = load_img(img, target_size=(IMG_SIZE1 ,IMG_SIZE2))
    
    #examine top for sign and bottom for obsticle
    image = image.crop((0,0,320,90))
    image = img_to_array(image)
    image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)
    
    r, g, b = cv2.split(image)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 85) & (g < 70) & (b < 220)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    black_filter =  ((r < 40) & (g < 40) & (b < 40))
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    gray_filter = ((np.absolute(r-g) < 40) & (np.absolute(g-b) < 40) & (np.absolute(b-r) < 40)) & ((r>32) & 
        (g>32) & (b> 32)) & ((r<239) & (g<239) & (b<239))  



    b[gray_filter], b[np.invert(b_filter | gray_filter)] = 0, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 0, 0
    
    
    masked_sign_image = cv2.merge((r, g, b))
    
    masked_sign_image = (masked_sign_image / 255. -.5).astype(np.float32)

    masked_sign_image = np.rollaxis(masked_sign_image, -1)
    
    return masked_sign_image
    
    
    
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x        
    
def get_class(img_path):
    img_path = os.path.normpath(img_path)
    class_id = img_path.split(os.sep)[-2]
    return int(class_id)



    
argv_count=len(sys.argv)
if argv_count>1:
    root_dir = sys.argv[1]
else:
    root_dir="./"
    

imgs = []
labels = []

all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    label = get_class(img_path)
    img = preprocess_img(img_path, label)
    
    
    #remove all signs that got clipped out of existence
    if np.count_nonzero(img) == 0: label = 0
    
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
# Make one hot targets
Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
K.set_image_data_format('channels_first')
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Input, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping



def cnn_model():
    #shape = (3,CRP_SIZE1,CRP_SIZE2)
    ###nvidia model without dropouts
    #img_input = Input(shape)
    #x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(img_input)
    #x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    #y = Flatten()(x)
    #y = Dense(100, activation='elu', kernel_initializer='he_normal')(y)
    #y = Dense(50, activation='elu', kernel_initializer='he_normal')(y)
    #y = Dense(10, activation='elu', kernel_initializer='he_normal')(y)
    #y = Dense(NUM_CLASSES, activation='softmax', kernel_initializer='he_normal')(y)
    #model = Model(inputs=img_input, outputs=y)


    #model from old steering 
    conv_layers, dense_layers = [32, 32, 64, 128], [512]
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=(3,CRP_SIZE1,CRP_SIZE2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    for cl in conv_layers:
        model.add(Conv2D(cl, (3, 3), padding='same', activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    #.001
    lr = 0.001
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])
    
    return model
    
    
    
    
model = cnn_model()


epochs = 50


X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.2, random_state=42)

datagen = ImageDataGenerator(featurewise_center=False,
                             featurewise_std_normalization=False,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10.)
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                    steps_per_epoch=(X_train.shape[0]/BATCH_SIZE),
                    epochs=epochs,
                    validation_data=(X_val, Y_val),
                    callbacks=[
                    ModelCheckpoint('sign.h5', save_best_only=True),
                    EarlyStopping(monitor='val_acc', patience=5)]
                    )


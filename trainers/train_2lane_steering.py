#based on https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a

import csv, random, numpy as np
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, ELU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import cv2
import os
import glob
import sys


SIZE1=240
SIZE2= 320

#size after cropping. Remember, also need to go specify how much in "image.crop"
CRP_SIZE1=64 #180
CRP_SIZE2=64 #320


"""
pass in the root directory of your log file as an argument, or use ./ with no arg passed
in the root directory place mutliple "Log" files renamed to whatever.
Put the image files in the IMG subdirectory of where the csv file is
Example:
\root_dir\craigs_logs\craigs_logs2.csv
\root_dir\craigs_logs\craigs_logs2.csv
\root_dir\craigs_logs\IMG
\root_dir\tims_logs\tims_logs1\tims_logs.csv
\root_dir\tims_logs\tims_logs2\tims_logs.csv
\root_dir\tims_logs\tims_logs1\IMG
root_dir\tims_logs\tims_logs2\IMG
"""
data_path = ''
BATCH_SIZE = 64 #keep lowering this if you get out of memory errors 256 -> 128 -> 64 -> 32
LEARNING_RATE = .001
SAMPLE_MULT = 1 #We augment samples, so we can train on more than we have. Each epoch will train (TOTAL_SAMPLES*SAMPLE_MULT)/BATCH_SIZE



def model(load, shape, checkpoint=None):
    """Return a model from file or to train on."""
    if load and checkpoint: return load_model(checkpoint)

    #conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]
    #
    #model = Sequential()
    #model.add(Conv2D(32, (3, 3), activation='elu', input_shape=shape))
    #model.add(MaxPooling2D())
    #for cl in conv_layers:
    #    model.add(Conv2D(cl, (3, 3), activation='elu'))
    #    model.add(MaxPooling2D())
    #model.add(Flatten())
    #for dl in dense_layers:
    #    model.add(Dense(dl, activation='elu'))
    #    model.add(Dropout(0.5))
    #model.add(Dense(1, activation='linear'))
    #model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    
    
    ##nvidia model with dropouts
    #img_input = Input(shape)
    #x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(img_input)
    #x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    #x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    #y = Flatten()(x)
    #y = Dense(1024, activation='elu', kernel_initializer='he_normal')(y)
    #y = Dropout(0.5)(y)
    #y = Dense(512, activation='elu', kernel_initializer='he_normal')(y)
    #y = Dropout(0.5)(y)
    #y = Dense(1, activation='linear', kernel_initializer='he_normal')(y)
    #model = Model(inputs=img_input, outputs=y)
    #model.compile(optimizer=Adam(lr=LEARNING_RATE), loss = 'mse')
    
    
    ##nvidia model without dropouts
    img_input = Input(shape)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(img_input)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
    x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    x = Conv2D(64, (3, 3), activation='elu', strides=(1, 1), padding='valid', kernel_initializer='he_normal')(x)
    y = Flatten()(x)
    y = Dense(100, activation='elu', kernel_initializer='he_normal')(y)
    y = Dense(50, activation='elu', kernel_initializer='he_normal')(y)
    y = Dense(10, activation='elu', kernel_initializer='he_normal')(y)
    y = Dense(1, activation='linear', kernel_initializer='he_normal')(y)
    model = Model(inputs=img_input, outputs=y)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss = 'mse',  metrics=['accuracy'])
    
    
    ##comma.ai model 
    #model = Sequential()
    #model.add(Conv2D(16, (8, 8), padding='same', strides=(4,4), activation='elu',  input_shape=shape, name='Conv1'))
    #model.add(Conv2D(36, (5, 5), padding='same', strides=(2,2), activation='elu', name='Conv2'))
    #model.add(Conv2D(64, (5, 5), padding='same', strides=(2,2), activation='elu', name='Conv3'))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(ELU())
    #model.add(Dense(512, activation='elu', name='FC1'))
    #model.add(Dropout(0.5))
    #model.add(ELU())
    #model.add(Dense(1, name='output'))
    #model.compile(optimizer=Adam(lr=LEARNING_RATE), loss = 'mse')
    
    return model

    
def get_X_y(root_dir):
    """Read the log file and turn it into X/y pairs. Add an offset to left images, remove from right images."""
    X, y = [], []
    #all_csv_paths = glob.glob(os.path.join(root_dir, '*/*.csv'))
    for csv_file in glob.iglob(os.path.join(root_dir, '**/*.csv'), recursive=True):
        with open(csv_file) as fin:
            print("Loading " + str(csv_file))
            for ctr_img,  steering_angle, _, _, speed, _, _ in csv.reader(fin):

                if float(speed) < .1: continue  # throw away low-speed samples
                #if abs(float(steering_angle)) < .01: continue  # throw away low-angle samples
                
                #Check if the file exists
                original_path = ctr_img
                ctr_img = os.path.normpath(ctr_img)
                file_name = ctr_img.split(os.sep)[-1]

                ctr_img = glob.glob(os.path.join(os.path.dirname(csv_file), 'IMG', file_name), recursive=False)

                if len(ctr_img) > 1:
                    #print("----ERROR---- Duplicate file names:  " + str(original_path))
                    continue
                elif len(ctr_img) < 1:
                    #print("----ERROR---- File was not found in subdirectory: " +str(original_path))
                    continue
                    
                #add the file and steering angle to the samples
                #ctr_img[0] = ctr_img[0].replace('\\', '/')
                X += [ctr_img[0].strip()]
                y += [float(steering_angle)]
    return X, y


def process_image(path, steering_angle, augment, shape=(SIZE1,SIZE2)):
    """Process and augment an image."""
    path = os.path.normpath(path)
    file_name = path.split(os.sep)[-1]
   

    image = load_img(path, target_size=shape)
    image = image.crop((0,40,320,220))
    
    #if augment and random.random() < 0.5: #we don't have shadows like udacity, but it'll be good when we use the real cars
     #   image = random_darken(image)  # before numpy'd


    image = img_to_array(image)
    #image = cv2.resize(image, (0,0), fx=0.35, fy=0.35)
    image = cv2.resize(image, (64, 64))
    #image = cv2.resize(image, (160, 120))  
    
        #ranslate_image uses pixels, so be careful if it's before or after resize
        #image = random_shift(image, 0, 0.2, 0, 1, 2)
    image, steering_angle = translate_image(image, steering_angle, -3, 3, -5, 5,  0.00025)
    if random.random() < 0.5:
        image = flip_axis(image, 1)
        steering_angle = -steering_angle
    
    #r, g, b = cv2.split(image)
    #r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    #g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    #b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    #black_filter =  ((r < 40) & (g < 40) & (b < 40))
    #y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    #white_filter = ((r > 200) & (g > 200) & (b > 200))
    #
    #
    #b[white_filter ], b[np.invert(white_filter )] = 255, 0
    #r[ r_filter], r[np.invert( r_filter)] = 255, 0
    #g[black_filter], g[np.invert(black_filter)] = 255, 0
    #
    #masked_image = cv2.merge((r, g, b))
    masked_image = image

    masked_image /= 255.
    masked_image -= 0.5


    #masked_image = (masked_image / 255).astype(np.float32)
    return masked_image, steering_angle

    
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x 

#https://towardsdatascience.com/teaching-cars-to-drive-using-deep-learning-steering-angle-prediction-5773154608f2
def translate_image(img, st_angle, low_x_range, high_x_range, low_y_range, high_y_range, delta_st_angle_per_px):
    """
    Shifts the image right, left, up or down. 
    When performing a lateral shift, a delta proportional to the pixel shifts is added to the current steering angle 
    """
    rows, cols = (img.shape[0], img.shape[1])
    translation_x = np.random.randint(low_x_range, high_x_range) 
    translation_y = np.random.randint(low_y_range, high_y_range) 
    
    st_angle += translation_x * delta_st_angle_per_px
    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
    return img, st_angle

    
def random_darken(image):
    """Given an image (from Image.open), randomly darken a part of it."""
    w, h = image.size

    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    # Loop through every pixel of our box (*GASP*) and darken.
    for i in range(x1, x2):
        for j in range(y1, y2):
            new_value = tuple([int(x * 0.5) for x in image.getpixel((i, j))])
            image.putpixel((i, j), new_value)
    return image

    
def _generator(batch_size, X, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)
            sa = y[sample_index] 
            image, sa = process_image(X[sample_index], sa, augment=True)
            batch_X.append(image)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)

        
def train():
    """Load our network and our data, fit the model, save it."""
    net = model(load=False, shape=(CRP_SIZE1, CRP_SIZE2, 3))
    X, y = get_X_y(data_path)
    y = [x/40 for x in y] #normalize it to between -1 and 1
    X_train, X_val, Y_train, Y_val = train_test_split(X, y,test_size=0.1, random_state=42)
    print(str(len(X)) + " records loaded")
    #print(Y_val)

    net.fit_generator(_generator(BATCH_SIZE, X_train, Y_train),
            validation_data=(_generator(BATCH_SIZE, X_val, Y_val)),
            validation_steps = len(X_val) / BATCH_SIZE,
            steps_per_epoch = ((len(X_train)*SAMPLE_MULT)/BATCH_SIZE), epochs=50,
            callbacks=[
                    ModelCheckpoint('steering_two.h5', save_best_only=True),
                    EarlyStopping(monitor='val_loss', patience=3)]
                    )

        
if __name__ == '__main__':
    argv_count=len(sys.argv)
    if argv_count>1:
        data_path = sys.argv[1]
    else:
        data_path="./"
    train()

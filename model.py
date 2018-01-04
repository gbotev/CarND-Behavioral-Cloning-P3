import csv
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

BATCH_SIZE = 256
EPOCHS = 14
VAL = 0.2
N_train = None

def read_image(row, prefix_path='', flip=False,  off='None',  angle_correction=0.1):
    image = None
    measurement = None
    source_path = ''
    if off == 'None':
        source_path = join(prefix_path,  row[0].strip())
        measurement = float(row[3])
    elif off == 'Left':
        source_path = join(prefix_path,  row[1].strip())
        measurement = float(row[3]) + angle_correction
    elif off == 'Right':
        source_path = join(prefix_path,  row[2].strip())
        measurement = float(row[3]) - angle_correction      
    image = cv2.imread(source_path)
    if flip:
        image = np.fliplr(image)
        measurement = -measurement
    #print(type(image),  source_path,   off,  flip)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB),  measurement

def read_data(csv_file_location,  nb_train):
    lines = []

    with open(csv_file_location + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        
    images = []
    measurements = []
    prefix_path = ''
    start_line = 0
    try:
        #check if we are working with supplied data or own data
        measurement = float(lines[0][3])
    except Exception as e:
        print("OK exception",  e)
        prefix_path = csv_file_location
        start_line = 1
    for line in lines[start_line:nb_train]:
        for flip in [False,  True]:
            for off in ['None', 'Left', 'Right']:
                image,  measurement = read_image(line, prefix_path=prefix_path, flip=flip,  off=off)
                images.append(image)
                measurements.append(measurement)
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    print("total images loaded:",  len(images),  "of size",  images[0].shape,  
        "and measurements are:",  len(measurements))
    return X_train,  y_train

X_train,  y_train = read_data('../SelfDrivingBehaviourCloning/my_data/' ,  N_train)
X_new,  y_new = read_data('../SelfDrivingBehaviourCloning/data/',  N_train) 
X_train = np.append(X_train,  X_new,  axis=0)
y_train = np.append(y_train,  y_new,  axis=0)




#Model parameters:
from keras.models import Sequential
from keras.layers import Flatten,  Dense,  Lambda,   MaxPooling2D,  Conv2D,  Cropping2D
#from keras.utils import plot_model

model = Sequential()
#160, 320, 3
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
print("after cropping:", model.layers[-1].output_shape)
# 90, 320, 3
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Conv2D(24, (5,  5),  activation='relu'))
print("after Conv1:",  model.layers[-1].output_shape)
# 86, 316, 24
model.add(MaxPooling2D(pool_size=(3, 3)))
print("after MP1:", model.layers[-1].output_shape)
# 28, 105, 24
model.add(Conv2D(36, (5,  5),  activation='relu'))
print("after Conv2", model.layers[-1].output_shape)
# 24, 101, 36
model.add(MaxPooling2D())
print("after MP2",model.layers[-1].output_shape)
# 12, 50, 36
model.add(Conv2D(48, (3, 3),  activation='relu'))
print("after Conv3",model.layers[-1].output_shape)
# 10, 48, 48
model.add(MaxPooling2D(pool_size=(2, 2)))
print("after MP3",model.layers[-1].output_shape)
# 5, 24, 48
model.add(Conv2D(64, (3, 3),  activation='relu'))
print("after Conv4",model.layers[-1].output_shape)
# 3,  22, 64
model.add(MaxPooling2D(pool_size=(2, 2)))
print("after MP4",model.layers[-1].output_shape)
# 1, 11, 64
model.add(Flatten())
print("after Flatten",model.layers[-1].output_shape)
model.add(Dense(704))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',  optimizer='adam')

#plot_model(model,  to_file='model.png',  show_layer_names=False,  show_shapes=True)

history_object = model.fit(X_train,  y_train,  
        validation_split=VAL,  shuffle=True,  epochs=EPOCHS,  batch_size = BATCH_SIZE)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


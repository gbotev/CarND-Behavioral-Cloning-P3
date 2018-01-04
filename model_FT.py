import csv
import cv2
from os.path import join
import numpy as np

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

    with open(csv_file_location + 'FT_log.csv') as csvfile:
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





N_train = None

BATCH_SIZE = 256
EPOCHS = 10
VAL = 0.05


X_train,  y_train = read_data('../SelfDrivingBehaviourCloning/data/' ,  N_train)

from keras.models import load_model
import matplotlib.pyplot as plt

loaded_model = load_model('model.h5')

for l in loaded_model.layers[:-1]:
    l.trainable  = False
    
history_object = loaded_model.fit(X_train,  y_train,  
        validation_split=VAL,  shuffle=True,  epochs=EPOCHS,  batch_size = BATCH_SIZE)
        
loaded_model.save('model_ft.h5')

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

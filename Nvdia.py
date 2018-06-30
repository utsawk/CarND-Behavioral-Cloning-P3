import csv
import cv2
import numpy as np 
import matplotlib.image as mpimg


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# keras setup
from keras.models import Model 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout, Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


# image processing
# image processing
def brightness_augment_image(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def translate_image(image, angle, translate_range_hor, translate_range_ver):
    dx = np.random.randint(-translate_range_hor//2, translate_range_hor//2)
    dy = np.random.randint(-translate_range_ver//2, translate_range_ver//2)
    transform_matrix = np.float32([[1,0,dx],[0,1,dy]])
    image = cv2.warpAffine(image,transform_matrix,(image.shape[1], image.shape[0]))
    angle = angle + dx * 0.0025
    return image,angle

def perturb_image_helper(image, angle):
    image = brightness_augment_image(image)
    image, angle = translate_image(image, angle, translate_range_hor, translate_range_ver)
    return image,angle


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

correction = [0, 0.25, -0.25]
translate_range_hor = 100
translate_range_ver = 20

train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

# add flipped image and the measurement for data augmentation with 1/2 probability
def generator(samples, batch_size, training):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples:
                for index in range(3):
                    source_path = sample[index]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = mpimg.imread(current_path)
                    measurement = float(sample[3]) + correction[index]
                    # copy for processing later
                    image_copy = np.copy(image)
                    measurement_copy = measurement
                    if training:
                        image, measurement = perturb_image_helper(image, measurement)
                    images.append(image)
                    measurements.append(measurement)
                    # add flipped image and the measurement for data augmentation
                    image = np.fliplr(image_copy)
                    measurement = -measurement_copy
                    if training:
                        image, measurement = perturb_image_helper(image, measurement)
                    images.append(image)
                    measurements.append(measurement)
       
            X = np.array(images)
            y = np.array(measurements)
            yield shuffle(X, y)

# dropout probabilities
dropout_prob_cl = 0.25
dropout_prob_fc = 0.5
# batch size
batch_size = 16

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size, True)
validation_generator = generator(validation_samples, batch_size, False)

inputs = Input(shape = (160, 320, 3))
preprocess = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

crop = Cropping2D(cropping=((50,20), (0,0)))(preprocess)

conv1 = Conv2D(24, (5,5) , subsample=(2,2), border_mode='valid',  activation="relu")(crop)
conv1 = BatchNormalization()(conv1)

conv2 = Conv2D(36, (5,5), subsample=(2,2), activation="relu")(conv1)
conv2 = BatchNormalization()(conv2)

conv3 = Conv2D(48, (5,5), subsample=(2,2), activation="relu")(conv2)
conv3 = BatchNormalization()(conv3)

conv4 = Conv2D(64, (3,3), activation="relu")(conv3)
conv4 = BatchNormalization()(conv4)

conv5 = Conv2D(64, (3,3), activation="relu")(conv4)
conv5 = BatchNormalization()(conv5)


flat_layer = Flatten()(conv5)

fc1 = Dense(1164, activation='relu')(flat_layer)
fc1 = Dropout(dropout_prob_fc)(fc1)
fc1 = BatchNormalization()(fc1)

fc2 = Dense(100, activation='relu')(fc1)
fc2 = Dropout(dropout_prob_fc)(fc2)
fc2 = BatchNormalization()(fc2)

fc3 = Dense(50, activation='relu')(fc2)
fc3 = Dropout(dropout_prob_fc)(fc3)
fc3 = BatchNormalization()(fc3)

fc4 = Dense(10, activation='relu')(fc3)

prediction = Dense(1)(fc4)
model = Model(inputs = inputs, outputs = prediction)

model.compile(loss = 'mse', optimizer = 'adam')

checkpoint = ModelCheckpoint('checkpoints/model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size, \
    validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, nb_epoch=10)
model.save('model_track2.h5')
model.summary()

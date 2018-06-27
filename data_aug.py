import csv
import cv2
import numpy as np 
import matplotlib.image as mpimg


# keras setup
from keras.models import Model 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout, Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


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

def perturb_image(image, steer, perturb_range_hor, perturb_range_ver):
	dx = perturb_range_hor*np.random.uniform()-perturb_range_hor/2
	steer_ang = steer + dx * 0.004
	dy = perturb_range_ver*np.random.uniform()-perturb_range_ver/2
	transform_matrix = np.float32([[1,0,dx],[0,1,dy]])
	image = cv2.warpAffine(image,transform_matrix,(image.shape[1], image.shape[0]))
	return image,steer_ang


lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = [0, 0.25, -0.25]
perturb_range_hor = 100
perturb_range_ver = 20

for line in lines[1:]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'data/IMG/' + filename
		image = mpimg.imread(current_path)
		images.append(image)
		measurement = float(line[3]) + correction[i]
		measurements.append(measurement)
		# add flipped image and the measurement for data augmentation
		images.append(np.fliplr(image))
		measurements.append(-measurement)

X = np.array(images)
y = np.array(measurements)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

def generator_train(X, y, batch_size):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
    	X, y = shuffle(X,y)
        for offset in range(0, num_samples, batch_size):
            X_samples = X[offset:offset+batch_size]
            y_samples = y[offset:offset+batch_size]
            images = []
            measurements = []
            for i in range(len(X_samples)):
                image = brightness_augment_image(X_samples[i])
                image, measurement = perturb_image(image, y_samples[i], perturb_range_hor, perturb_range_ver)
                images.append(image)
                measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

# dropout probabilities
dropout_prob_cl = 0.25
dropout_prob_fc = 0.5
# batch size
batch_size = 128

# compile and train the model using the generator function
train_generator = generator_train(X_train, y_train, batch_size)

inputs = Input(shape = (160, 320, 3))
preprocess = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

crop = Cropping2D(cropping=((70,25), (0,0)))(preprocess)

conv1 = Conv2D(12, (5, 5))(crop)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Dropout(dropout_prob_cl)(conv1)
conv1 = MaxPooling2D((2, 2))(conv1)

conv2 = Conv2D(32, (5, 5))(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Dropout(dropout_prob_cl)(conv2)
conv2 = MaxPooling2D((2, 2))(conv2)

maxpool_layer1 = MaxPooling2D((2, 2))(conv1)
flat_layer1 = Flatten()(maxpool_layer1)
flat_layer2 = Flatten()(conv2)

concat = concatenate([flat_layer1, flat_layer2])
concat = Dropout(dropout_prob_cl)(concat)

fc1 = Dense(100)(concat)
fc1 = BatchNormalization()(fc1)
fc1 = Activation('relu')(fc1)
fc1 = Dropout(dropout_prob_fc)(fc1)

prediction = Dense(1)(fc1)
model = Model(inputs = inputs, outputs = prediction)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, steps_per_epoch= len(X_train)//batch_size, validation_data=(X_validation, y_validation),  validation_steps=len(X_validation)//batch_size, nb_epoch=10)
model.save('model.h5')
model.summary()



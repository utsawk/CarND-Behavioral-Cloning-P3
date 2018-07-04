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
# from keras.callbacks import ModelCheckpoint


# correction for center, left and right images respectively
correction = [0, 0.25, -0.25]

# image translation ranges
translate_range_hor = 100
translate_range_ver = 20


def brightness_augment_image(image):
    """
    Scales the brightness uniformly randomly 
    :param image: RGB image
    :return: RGB image randomly scaled brightness
    """

    # convert to HSV and scale last dimension randomly
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    random_brightness = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_brightness
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    # convert back to RGB
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def translate_image(image, angle, translate_range_hor, translate_range_ver):

    """
    Translates the image horizontally and vertically
    :param image: RGB image
    :param angle: steering angle associated with input image
    :param translate_range_hor: max translation in horizontal direction
    :param translate_range_ver: max translation in vertical direction
    :return: RGB image translated horizontally and vertically
    """
    dx = np.random.randint(-translate_range_hor//2, translate_range_hor//2)
    dy = np.random.randint(-translate_range_ver//2, translate_range_ver//2)
    transform_matrix = np.float32([[1,0,dx],[0,1,dy]])
    image = cv2.warpAffine(image,transform_matrix,(image.shape[1], image.shape[0]))
    angle = angle + dx * 0.0025
    return image,angle

def random_shadow(image):
    """
    Adds random shadow to image
    :param image: RGB image
    :return: RGB image with random shadow
    """
    x1, y1 =  np.random.randint(0, image.shape[1]), 0
    x2, y2 = np.random.randint(0, image.shape[1]), image.shape[0] 
    
    xn, yn = np.mgrid[0:image.shape[0] , 0:image.shape[1] ]
    
    mask = np.zeros_like(image[:, :, 1])
    mask[(yn - y1) * (x2 - x1) - (y2 - y1) * (xn - x1) > 0] = 1

    cond = mask == np.random.randint(0, 2)
    random_brightness = np.random.uniform(low=0.2, high=0.5)

    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_hls[:, :, 1][cond] = image_hls[:, :, 1][cond] * random_brightness
    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def perturb_image_helper(image, angle):
    """
    Processes the image with brightness augmentation, random shadow and image translation
    :param image: RGB image
    :return: RGB image with random shadow
    """
    image = brightness_augment_image(image)
    image = random_shadow(image)
    image, angle = translate_image(image, angle, translate_range_hor, translate_range_ver)
    return image,angle

# add flipped image and the measurement for data augmentation with 1/2 probability
def generator(samples, batch_size, training):
    """
    Generator function to read and perturb images randomly on the fly. Image perturbation is
    only applied during training
    :param samples: lines read from the input image file representing training/validation data
    :param batch_size: batch size for training/validation
    :training: bool variable representing training (True) or validation (False)
    :return: RGB image with random shadow
    """
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
                    if training and np.random.rand() > 0.5:
                        image, measurement = perturb_image_helper(image, measurement)
                    images.append(image)
                    measurements.append(measurement)
                    # add flipped image and the measurement for data augmentation
                    image = np.fliplr(image_copy)
                    measurement = -measurement_copy
                    if training and np.random.rand() > 0.5:
                        image, measurement = perturb_image_helper(image, measurement)
                    images.append(image)
                    measurements.append(measurement)
       
            X = np.array(images)
            y = np.array(measurements)
            yield shuffle(X, y)


def SermaNet():
    """
    Creates and returns the SermaNet model
    """
    # dropout probabilities
    dropout_prob_cl = 0.25 # for convolutional layer (only used for SermaNet architecture)
    dropout_prob_fc = 0.5 # fully connected layers

    inputs = Input(shape = (160, 320, 3))
    preprocess = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

    crop = Cropping2D(cropping=((50,20), (0,0)))(preprocess)

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
    return model


def Nvdia():
    """
    Creates and returns the Nvdia model
    """
    # dropout probabilities
    dropout_prob_fc = 0.5 # fully connected layers

    inputs = Input(shape = (160, 320, 3))
    preprocess = Lambda(lambda x: x / 255.0 - 0.5)(inputs)

    crop = Cropping2D(cropping=((50,20), (0,0)))(preprocess)

    conv1 = Conv2D(24, (5,5) , strides=(2,2), activation="relu")(crop)
    conv1 = BatchNormalization()(conv1)

    conv2 = Conv2D(36, (5,5), strides=(2,2), activation="relu")(conv1)
    conv2 = BatchNormalization()(conv2)

    conv3 = Conv2D(48, (5,5), strides=(2,2), activation="relu")(conv2)
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
    return model


if __name__ == "__main__":
    # read from file
    lines = []
    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    run_model = 1 # choose 0 for SermaNet, 1 for Nvdia

    # dividing available data into training and validation sets
    train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

    # batch size
    batch_size = 16

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size, True)
    validation_generator = generator(validation_samples, batch_size, False)

    if run_model == 0:
        model = SermaNet()
    else:
        model = Nvdia()


    model.compile(loss = 'mse', optimizer = 'adam')

    # checkpoint = ModelCheckpoint('checkpoints/model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

    model.fit_generator(train_generator, steps_per_epoch= len(train_samples)//batch_size, \
        validation_data=validation_generator, validation_steps=len(validation_samples)//batch_size, nb_epoch=15)
    model.save('model.h5')
    model.summary()



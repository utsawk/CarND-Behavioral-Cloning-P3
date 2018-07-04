# **Behavioral Cloning** 

---

### Goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/hist_given.png "histogram of steering angles"
[image2]: ./writeup_images/hist_all_images.png "histogram of all"
[image3]: ./writeup_images/hist_aug.png "histogram of augmented dataset"
[image4]: ./writeup_images/Nvdia_arch.jpg "Nvdia arch"
[image5]: ./writeup_images/block_diagram.jpg "block diagram"
[image6]: ./writeup_images/image_aug.png "image augmentation"

### Files

My project includes the following files:
* exploratory_data_analysis.ipynb (Jupyter notebook) containing analysis of data given by Udacity & how enhancements on it can help with training
* model.py containing the script to create and train a neural network based on SermaNet[1] and Nvdia paper [2] (default runs Nvdia, change run_model to 0 (line# 238) for SermaNet)
* drive.py for driving the car in autonomous mode (speed increased to 12mph for faster driving)
* video.py for creating video of pics taken during autonomous mode
* model.h5 containing a trained convolution neural network (I could not get git lfs to work, so I have uploaded the model.h5 on dropbox link [here](https://www.dropbox.com/s/1zyafhi0nd7eq8j/model.h5?dl=0))
* writeup.md/README.md summarizing the results
* track1 containing the video of driving around track 1
* track2 containing the video of driving around track 2
* track3 containing the video of driving around the mountain track in old simulator


### Using the trained model to run the simulator
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Data Analysis Of Given Data

I started by using the Udacity provided data and only center images. I implemented the LeNet architecture, but did not try tuning it too much when it did not work. Next, I used the exact model (SermaNet [1]) I trained for the [Traffic Sign Classifier project](https://github.com/utsawk/CarND-Traffic-Sign-Classifier-Project) and immediately saw improvement. However, it still did not complete the track. Plotting the histogram of the steering angles shows that the given data is biased and ~0 steering angles dominate. A model trained on this dataset would have a tendency of predicting ~0 steering angles. Also, because the track has more left turns, negative angles are better represented than positive angles.

![alt text][image1]

As recommended, I decided to include the left and right images and add 0.25 and -0.25 corrections (in radians) respectively. Additionally, I flipped the images and the corresponding steering angles and added them to the dataset. The histogram of the new dataset is shown below. Using this and the SermaNet architecture was good enough to drive around track 1 without touching any of the lane lines. However, the car was veering from side to side every now and then and not driving as smoothly. 

![alt text][image2]

From the histogram, it can be seen that the model may tend to learn three steering angles - -0.25, 0 and 0.25 and is probably the reason the car veers from side to side sometimes. To fix it, I added horizontal translations to balance the dataset. Note that I also consider other image augmentation techniques (more on this in the next section) for generalization, but these don't affect the steering angles and are thus not considered for plotting the histogram here. The resulting histogram shown below has a nice Gaussian look to it and car seems to drive better as well.

![alt text][image3]

I had achieved the required performance easily on track 1 (within a couple of days) and tried using the same architecture and image augmentation on track 1 images to train the model to drive on track 2. 

## Solution Overview
Training on the dataset consists of multiple steps. Training images are augmented (as described in next section) and fed into the CNN that computes the steering angle. The predicted steering angle is compared to the measured steering angles and the weights of the CNN are adjusted to decrease the mean squared error. The overall block diagram is shown in figure below.

![block_diagram][image5]


### Image augmentation

I considered the following image augmentation techniques:
1. Horizontal translation: As mentioned earlier, this was done to create the effect of different views and alter steering angles accordingly. The translation was drawn uniformly at random from [-50, 50] and applied to the image. I used 0.0025 radians angle per pixel of translation and it seemed to work well. I did not experiment much with this value.
2. Vertical translation: I also added vertical translation to create the effect of slopes - uphill and downhill. The translation was drawn uniformly at random from [-20, 20] and applied to the image. I did not try higher values for this. This does not affect the steering angle.  
3. Brighness augmentation: The brightness is scaled uniformly at random from [0.5, 1.5]. The track 2 for the project is a little darker and changing brightness would help the neural network to learn better. 
4. Random shadow: Almost a week of training was based on the above mentioned image augmentation techniques and SermaNet architecture. However I could not modify it to work on track 2. Then I decided to switch architecture to the Nvdia architecture because it seems suited to the problem. Also, I realized that brightness adjustment uniformly scales the pixels and random shadowing would be a good way to simulate shadows and generalize the model. Inspired by [3], I added random shadow to the images.

Note that all the above image augmentation techniques are applied using the *perturb_image_helper(image, angle)* function (lines# 84-93) in model.py. An example of applying these functions on an image is given below.

![image_aug][image6]


### Neural Network Architecture

I used two architectures for training - SermaNet (implementation in lines# 139-178) and Nvdia (implementation in lines# 181-227). The former along with image augmentation techniques is good enough to drive on track 1. The SermaNet architecture is same as used in [4] with dropout probabilities of 0.25 and 0.5 for convolutional and fully connected layers respectively. However, I could not get it to work on track 2. Then I switched to the Nvdia architecture. Only the Nvdia architecture is presented below for brevity.


#### Nvdia architecture

I used the Nvdia architecture for the final submission, that has 5 convolution layers and 5 fully connected layers, with the last layer representing the vehicle steering angle as output. The input to the model is a (160, 320, 3) image. The following are salient features of the model:
1. I use a lambda layer for normalizing the data. I used the Udacity recommended normalization (divide by 255.0 and subtract 0.5).
2. The top 50 pixels and bottom 20 pixels are cropped off to get rid of the surroundings (like trees, sky, etc.) and hood of the car respectively. 
3. I have observed faster convergence when using batch normalization [5], thus I use that in this project as well. Additionally, batch normalization allows the use of higher learning rates and also acts like a regularizer [5]. 
4. I used the Adam optimizer and did not need to tune the learning rate throughout the project. 
5. To prevent overfitting in the Nvdia model, I used dropout with 0.5 (keep/dropout) probability for the fully connected layers.


| Layer (type)        		|     Output Shape	        					| Param # |
|:---------------------:|:------------------------------------:|:--------: 
| input_1 (InputLayer)   |      (None, 160, 320, 3)   |    0         |
| lambda_1 (Lambda)       |     (None, 160, 320, 3)   |    0         |
| cropping2d_1 (Cropping2D)  |  (None, 90, 320, 3)    |    0         |
| conv2d_1 (Conv2D)        |    (None, 43, 158, 24)   |    1824      |
| batch_normalization_1    |    (None, 43, 158, 24)   |    96        |
| conv2d_2 (Conv2D)        |    (None, 20, 77, 36)    |    21636     |
| batch_normalization_2    |    (None, 20, 77, 36)    |    144       |
| conv2d_3 (Conv2D)        |    (None, 8, 37, 48)     |    43248     |
| batch_normalization_3    |    (None, 8, 37, 48)     |    192       |
| conv2d_4 (Conv2D)        |    (None, 6, 35, 64)     |    27712     |
| batch_normalization_4    |    (None, 6, 35, 64)     |    256       |
| conv2d_5 (Conv2D)        |    (None, 4, 33, 64)     |    36928     |
| batch_normalization_5    |    (None, 4, 33, 64)     |    256       |
| flatten_1 (Flatten)      |    (None, 8448)          |    0         |
| dense_1 (Dense)          |    (None, 1164)          |    9834636   |
| dropout_1 (Dropout)      |    (None, 1164)          |    0         |
| batch_normalization_6    |    (None, 1164)          |    4656      |
| dense_2 (Dense)          |    (None, 100)           |    116500    |
| dropout_2 (Dropout)      |    (None, 100)           |    0         |
| batch_normalization_7    |    (None, 100)           |    400       |
| dense_3 (Dense)          |    (None, 50)            |    5050      |
| dropout_3 (Dropout)      |    (None, 50)            |    0         |
| batch_normalization_8    |    (None, 50)            |    200       |
| dense_4 (Dense)          |    (None, 10)            |    510       |
| dense_5 (Dense)          |    (None, 1)             |    11        |


Total params: 10,094,255

Trainable params: 10,091,155

Non-trainable params: 3,100


The overall achitecture is presented in the figure below.
![architecture][image4]

### Training data

As mentioned above, Udacity provided data (all center, right and left images) was enough to train the models to drive on track 1. With both the architectures and image augmentations mentioned above, I could not get the car to drive on track 2. I really wanted to make it work without collecting additional data, but I gave up eventually. Then I collected data by driving 2 laps on track 2. Driving on track 2 was challenging for me as well and I could not drive in one lane. Even after this, both the architectures were failing at a sharp left turn (almost a U-turn). I decided to collect data driving and putting the car in the exact same situation. Adding this to the training data helped the car navigate the tricky spot and it was able to complete the track autonomously (only the Nvdia architecture). 

For every run, the data is shuffled and randomly split into two data sets - 80% is used for training and 20% is used for validation (line# 241).

The data used for the project can be downloaded from [here](https://www.dropbox.com/s/6nh03a8mm142zgz/data.zip). It contains the training data provided by Udacity on track 1 and a couple of runs from track 2 (driving manually on track 2 while exhibiting good driving behaviour is a challenge too, but the data is good enough to keep the car on the road).

### Conclusion

In this project, I learnt the lesson that there is no substitute for good labelled data. Image augmentation techniques are helpful to generalize the model, but it may not be possible to generalize it to the extent that it can provide a solution for all possible scenarios. Whenever possible, if collecting data is cheap (in this case it was), it is always recommended to do so.

[1] http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

[2] http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

[3] https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f

[4] https://github.com/utsawk/CarND-Traffic-Sign-Classifier-Project

[5] https://arxiv.org/abs/1502.03167

# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

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
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Files

My project includes the following files:
* exploratory_data_analysis.ipynb (Jupyter notebook) containing of data given by Udacity & how enhancements on it can help with training
* model.py containing the script to create and train a neural network based on Nvdia paper [1]
* SermaNet.py containing the script to create and train a neural network based on SermaNet [2]
* drive.py for driving the car in autonomous mode (speed increased to 12mph for faster driving)
* video.py for creating video of pics taken during autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* track1 containing the video of driving around track 1
* track2 containing the video of driving around track 2
* track3 containing the video of driving around the mountain track in old simulator

### Data

The data can be downloaded from [here](https://www.dropbox.com/s/6nh03a8mm142zgz/data.zip). It contains the training data provided by Udacity on track 1 and a couple of runs from track 2 (driving manually on track 2 while exhibiting good driving behaviour is a challenge too, but the data is good enough to keep the car on the road).


#### Using the trained model to run the simulator
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Data Analysis

I started by using the Udacity provided data and only center images. I implemented the LeNet architecture, but did not try tuning it too much when it did not work. Next, I used the exact model I trained for the [Traffic Sign Classifier project](https://github.com/utsawk/CarND-Traffic-Sign-Classifier-Project) [2] and immediately saw improvement. However, it still did not complete the track. Plotting the histogram of the steering angles shows that the given data is biased and and ~0 steering angles dominate. A model trained on this dataset would have a tendency of predicting ~0 steering angles. Also, because the track has more left turns, negative angles are better represented than positive angles.

![alt text][image1]

As recommended, I decided to include the left and right images and add 0.25 and -0.25 corrections (in radians) respectively. Additionally, I flipped the images and the corresponding steering angles and added them to the dataset. The histogram of the new dataset is shown below. Using this and the SemraNet architecture was good enough to drive around track 1 without touching any of the lane lines. However, the car was veering from side to side every now and then and not driving as smoothly. 

![alt text][image2]

From the histogram, it can be seen that the model tends to learn the three steering angles - -0.25, 0 and 0.25 and is probably the reason the car veers from side to side sometimes. To fix it, I added horizontal translations to balance the dataset. Note that I also consider other image augmentation techniques (more on this in the next section) for generalization, but these don't affect the steering angles and are thus not considered for plotting the histogram here. The resulting histogram shown below has a nice Gaussian look to it and car seems to drive better as well.

![alt text][image3]

I had achieved the required performance easily on track 1 (within a couple of days) and tried using the same architecture and image augmentation on track 1 images to train the model to drive on track 2. 

### Image augmentation
I considered the following image augmentation techniques:
1. Horizontal translation: As mentioned earlier, this was done to create the effect of different views and alter steering angles accordingly. The translation was drawn uniformly at random from [-50, 50] and applied to the image. I used 0.0025 radians angle per pixel of translation and it seemed to work well. I did not experiment much with this value.
2. Vertical translation: I also added vertical translation to create the effect of slopes - uphill and downhill. The translation was drawn uniformly at random from [-20, 20] and applied to the image. I did not try higher values for this. This does not affect the steering angle.  
3. Brighness augmentation: The brightness is scaled uniformly at random from [50, 150]. The track 2 for the project is a little darker and changing brightness would help the neural network to learn better. 
4. Random shadow: 

### Neural Network Architecture




#### An appropriate model architecture has been employed
 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

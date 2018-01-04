# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/recovery_start.jpg "Start of recovery"
[image3]: ./examples/recovery_mid.jpg "Mid recovery"
[image4]: ./examples/recovery_end.jpg "After recovery"
[image5]: ./examples/recovery_end_fl.jpg "Flipped"
[image6]: ./examples/train_val.png "Model training"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for importing data, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In the beginning there is a functionality to read and import the training data. I haven't used generators as the model was trained on a server with enough ram , and not reading every batch from disk speeds up the training.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network similar to the one used by nvidia for self driving cars. It consists of 2 5x5 convolution layers each followed by max pooling, then 2 more convolution layers with 3x3 kernel size and finally there are 5 fully connected layers. The main difference is that because of the higher resolution of images compared to nvidia, after the first layer convolutio layer I am using 3x3 max pooling instead of 2x2. The other difference is that I am using max pooling layers instead of strided convolutions.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 81). 

#### 2. Attempts to reduce overfitting in the model

No dropout layers were used to reduce overfitting. Instead I have tried to generate more training data so the model generalizes better. Also early stopping was used when the training error drops below the validation  error and the validation error stop decreasing.

The model was tested by running it through both tracks of the simulator and ensuring that the vehicle could stay on the track. I also tried  to  steer the car in wrong directions and even manually get the car out of the track to check that the model can deal with such 'unseen' situations.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving on both tracks in both directions.

The data was then augmented by using the flipped images and also the images from the left and right cameras as well as the center one.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try to test the simplest one first and then add more layers and additional training the data to increase the performance while keeping the model from overfitting. 

My first step was to use the simplest model of one fully connected layer to see ifthe model trains fine and make use of the data.The car was trying to stay in the center of the road, but steered too much.

Then I used a model similar to LeNet5 with a normalization layer for the input pixels. This time the car steering was smooth, but it was pulling to the left.

Next step was to generate  additional data by flipping the steering angle and images and also by driving in the opposite direction of track 1. Here I also cropped the upper 50 and lower 20 pixels of the images. This time the car was not moving to left or right intentionally, but it couldn't recover when it went off the center of the road.

In order to fix this I used the left and right camera images and tried a few correction angles (0.05, 0.07, 0.1, 0.13, 0.15, 0.2) - the 0.1 seemed to work best.. Now the car was able to  recover from going off the center and was trying to keep at the center, but had difficulties with hard corners. I decided it is time to move to a more advanced model which can capture the differences between a sharp and moderate turn.

The model I used was similar to the one nvidia uses in their self-driving cars with a few minor tweaks. I decided to use maxpooling layers instead of strided convolutions and in the first pooling layer I used 3x3 max pooling so  the size of my next layer would be similar to the second convolution layer of nvidia's model.

This model worked flawlessly on track 1 but started overfitting (the train error dropped much below validation error) after the first epoch. Here I decided that it would be best to use more data and more challeging data in order to try the potential of this model.

I added  5 laps on the second track in both drections with also a few specific recovery sections. After training the model for 10 epochs it could handle the first track in both directions and also almost all of the second track without a specific section with a U turn with almost no visibility.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-112) consisted of a convolution neural network with the following layers and layer sizes:
* 2D Convolution with 5x5 kernel and 1,1 stride
* Max Pooling 3x3 with stride 1,1
* 2D Convolution with 5x5 kernel and 1,1 stride
* Max Pooling 2x2 with stride 1,1
* 2D Convolution with 3x3 kernel and 1,1 stride
* Max Pooling 2x2 with stride 1,1
* 2D Convolution with 3x3 kernel and 1,1 stride
* Max Pooling 2x2 with stride 1,1
* Flatten and fully connected layer with 704 units
* Fully connected 100
* Fully connected 50
* Fully connected 10
* Fully connected 1 (output layer)
 
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the road. These images show what a recovery looks like starting from right to left:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points. I also drived the car in reverse direction of the track to add more unobserved data to my samples.

To augment the data set, I also flipped images and angles thinking that this would help generalization by adding new data and avoid bias to left or right by having a symmetrical data. For example, here is an image that has then been flipped:

![alt_text][image5]

I also used the images from left and right side cameras with correction angle coefficient of 0.1 in order to help with recovery. 

This was done on both supplied training data, and the laps generated by me.

After the collection process, I had 148,662  number of data points. I then preprocessed this data by normalizing using the formula x/255.0 - 0.5 to obtain pixel intensity values within [-0.5, 0.5]

I finally randomly shuffled the data set and put 20% of the data into a validation set. During modeling and testing the top 50 and bottom 20 rows are removed.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 14 as evidenced by the graph below. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image6]

In addition there was a tricky section with two U turns on the second track which the car crashed. This was why I decided to fine tune the model on 4 samples of passing that section in both directions.

First I tried to retrain the whole network for one epoch just on the new data, but this worsened the generalization ability of the net. 

After that I freezed all the layers up to the last layer of the network (fully connected 1) and fine tuned just the last layer using the new samples for 2 epochs. As a result the car passes the second track without error at 20 mph, and leaves the track only in 5 places while driving at 30 mph. The only negative effect I saw was that on the straight parts of track 1 the car tended to wiggle a bit more.

You can find the fine tuning part at model_FT.py, the fine tuned model is model_ft.h5 and in the videos folder you may find track1 run (run1.mp4) with the first model and track1  (run1_FT.mp4) and track2 (run2_FT.mp4) runs with the second, fine tuned, model.

Further test might be to record a sample with "sportier"  driving on track2 and try to fine_tune the last 2 or 3 layers so the car can pass the whole track2 at 30mph by learning to choose better trajectories.

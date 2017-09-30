#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./dataDistribution.png "Model Distribution"
[image2]: ./cnn-architecture-624x890.png "NVIDIA architecture"
[image3]: ./training_curve.png "Error Plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Trainingday_generator.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* vid.mp4 successful video of the car making a lap across the track


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses the NIVIDIA archetecture (TrainingDay_generator.py). It uses 'elu' activations instead of 'relu' for better performance. The data is normalized in the model using a Keras lambda layer. The data is also cropped to remove irrelevant information. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I tried adding dropout layers in my neural network, but didn't see any significant improvement, so I took it out.



####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

I first tried to gather my own data and then train my model from that. Unfortunately, I'm a poor driver, so my model kept telling the car to drive into the water. It wasn't until I added Udacity's sample data did the car manage to drive around the track with no issues.

I also plotted the distribution of angles in the data set and saw that a large majority of data points have a 0 angle curve. This is bad because we want to have as close to uniform distribution as possible to prevent bias. To remove this problem, in my "Preprocess" function, I removed all of the lines with the steering at 0 angles. There was still a huge distribution of steering angles close to 0 and very few instances where the steering angle was close to 1. To smooth this out, I put the data through my "Equalize" function, that randomly removed data points until each bin was at most the average data points. If you look at the chart below, the blue lines represent the distribution before removing the data points and the orange lines represent the distribution after randomly removing data points.

![alt text][image1]

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used NIVIDIA's architecture for my model. In the FAQ for this project, one of the most prevalent comments was that this architecture was good enough for this project. Manipulating and getting good test data was more important than the architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The Test error and the validation error were pretty similar. This tells me that I am not overfitting my data.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I realized that 'cv2.imread' opens the image in BGR mode, whereas the 'video.py' file opens the image in RGB. To prevent any errors, I converted our image to RGB right after opening the image. I also flipped the image horizontally if their steering angle was above a certain threshold. This helped me get more data points without creating more data points near the center of my distribution. I also used the left and right camera pictures and added a correction factor to the steering angle. This gave me even more data points and helped my model know what to do if it got closer to the edge.

Inside the model, I normalized the values by dividing each of the pixels by 255 an subtracting by 0.5 to make the values zero meaned with a spread of 1. Finally, I cropped the top of each image by 70 pixels and the bottom by 25 to remove the sky (which is static) and the dashboard (which is also static). This reduced the amount of information my model needed to process and made training go by faster.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

Input plane 3@320x160
Normalization
Cropping 3@320x65
2D Convolution 5x5 kernel 24@316x61
Subsample 2x2 kernel 24@158x31
2D Convolution 5x5 kernel 36@154x27
Subsample 2x2 kernel 36@77x14
2D Convolution 5x5 kernel 48@73x10
Subsample 2x2 kernel 48@37x5
2D Convolution 3x3 kernel 64@35x3
2D Convolution 3x3 kernel 64@33x1
Flatten 2212 neurons
Dense 100 neurons
Dense 50 neurons
Dense 10 neurons
Dense 1 neuron -> output

Here is a visualization of the architecture

![alt text][image2]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I recorded the vehicle driving the other direction for one lap.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to recover if it ended up on the edge of a line.

To augment the data sat, I also flipped images and angles thinking that this would remove the car's bias to turn left only if the steering angle was above a certain threshold.

I also used both the left camera and right camera angles and added a correction factor of 0.2 and -0.2 to the angles, respectively. This correction factor was chosen experimentally and seems to do a good job.

I then preprocessed this data by converting the image from BGR to RGB. This is because cv.imread() creates the image in BGR format, whereas the drive.py file reads it in RGB. This mismatch will cause the car to drive over the dirt path. Converting it beforehand will keep the car on the track. In the model, I also normalized the data and cropped the image. This is beacuse normalizing the data will lead to a faster convergence and cropping the data will remove uninteresting data such as the sky and the car's dashboard.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the error output. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image3]

####4. Conclusion

This project was relatively difficult because there wasn't much that I could change on the architecture to create a better result. The way I was able to make my model drive properly was to 1) Convert the images to RGB from BGR, 2) Remove any instances when the steering angle was 0, and 3) collect more training data. Since I was a terrible driver in the simulator, I started with Udacity's sample data and worked from there. This current model performs poorly on the challenge set, I believe it's because the lanes of the challenge set are too close to each other. This is causing my correction factor to over compensate and is causing my car to go off the road. If I go back to this project, I will lower the correction factor and train on the challenge course extensively. 

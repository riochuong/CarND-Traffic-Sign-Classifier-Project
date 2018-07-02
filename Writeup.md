# **Traffic Sign Recognition** 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to explore the characteristics of the data:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is: **12630**
* The shape of a traffic sign image is: **32x32x3**
* The number of unique classes/labels in the data set is **42**

#### 2. Include an exploratory visualization of the dataset.

Below is the historams shows the distribution of the data for each type of traffic signs in training, validation and test sets. Each unique traffic sign has label id from 0 to 42.



![alt text](./train_data.png)
![alt text](./test_data.png)
![alt text](./valid_data.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, convert all data from RGB to grayscale in order to greatly reduce number of unneccessary features so the training time can be faster. The network can then focus on learning the correct pattern from the single feature map.

![alt text](before.png)
![alt text](after.png)

Second, normalize the pixel value and scale them in between -1 and 1 in order to avoid exploding in gradients and help backpropagation works much better.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 30x30x64 	|
| BatchNormalization     	|  	|
| RELU					|												|
| Dropout					|			keep probaility: 0.7				|
| Max pooling	      	| 2x2 stride,  outputs 15x15x64 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 13x13x128					|
| BatchNormalization     	|  	|
| RELU					|												|
| Dropout					|			keep probaility: 0.7				|
| Max pooling	      	| 2x2 stride,  outputs 6x6x128 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 4x4x256					|
| BatchNormalization     	|  	|
| RELU					|												|
| Dropout					|			keep probaility: 0.7				|
| Max pooling	      	| 2x2 stride,  outputs 2x2x64 				|
| Fully connected		| ouputs 500        									|
| RELU					|												|
| Dropout					|			keep probaility: 0.4				|
| Fully connected		| ouputs 300        									|
| RELU					|												|
| Dropout					|			keep probaility: 0.4				|
| Fully connected		| ouputs 42        									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the **reduce mean** on **softmax crossentropy** as loss function. 
I chose **Adam Optimizer** with learning rate **0.0005**.
I use **batch size** of 32
I use **number of epochs** of 100

I started with learning rate 0.001 however the network did not reach the required accuracy for validation set, therefore I decided to reduce learning rate to 0.0005 in order to help the gradient descent find better optimal points. Later on after adding more layers to the network and the validation accuracy reached the required 93% value. I decided to stick with this learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **100%**
* validation set accuracy of **95%**
* test set accuracy of **92.2%**

I have been using the iterative process to come up with the final network architecture. Using the LeNet architecture as a starting point, I start increasing number of filters per convolution layers as well as adding more convolution layers to help the network learn more high level features and improve the validation accuracy. In order to help the network converge faster and avoid overfitting, I apply some optimization techniques like batch normalization and dropout between each convolution layers as well as fully connected layers. The keep probability for convolution is at 70% while the keep proablity for fully connected layer is at 60%. These are set back to 100% during inference. The network is trained for 100 epochs and reach 95% accuracy on validation set at the 90th epoch. However, the network reach 94% accuracy on validation set around the 41st epoch so we do not have much improvement from that on. I decided to stop at 100th epoch and use the result of the 90th epoch for the final evaluation. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](sign_examples/31.jpg)
![alt text](sign_examples/27.jpg)
![alt text](sign_examples/23.jpg)
![alt text](sign_examples/25.jpg)
![alt text](sign_examples/12.jpg)
The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Wild animal crossings	     		| Road Work  									| 
| Pedestrians     			| Ahead only 										|
| Slippery Road				| Slippery Road											|
| Road work	      		| Right of way at next extension					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This accuracy is a little low in compares with the test set. It might due to the position of the sign and size of the sign that might cause the model some confusion.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the **31.jpg** image which is "Wild and animal crossings" sign, the model predict

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| Road Work   									| 
| .086     				| Road narrows on the right|
| .017					| Right-of-way at the next intersection|
| .015	      			|General caution|
| .010				    | Traffic signals|

For the **27.jpg** image which is "Pedestrians" sign, the model predict

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .122         			| Ahead only   									| 
| .121     				| Road work|
| .096					| Speed limit (60km/h)|
| .083	      			|General caution|
| .070				    |Right-of-way at the next intersection|

For the **23.jpg** image which is "Slippery road" sign, the model predict

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .828        			| Slippery road   									| 
| .100     				| Roundabout mandatory|
| .019					| No entry|
| .010	      			|Vehicles over 3.5 metric tons prohibited|
| .008				    | Beware of ice/snow|

For the **25.jpg** image which is "Road work" sign, the model predict

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .513         			| Right-of-way at the next intersection| 
| .182     				| Double curve|
| .146					| Road work|
| .038	      			|Children crossings|
| .026				    | priority road|

For the **12.jpg** image which is "Priority road" sign, the model predict

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Priority road   									| 

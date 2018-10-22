# SimpleImgClassifier
Use TensorFlow to recognize between several classes of objects.
## Classify Target: Monkey Species Dataset
I choose to train a classifier to recognize between several different monkey species by using TensorFlow.  
The dataset contains 10 different monkey species. I directly downloaded the dataset from kagge:https://www.kaggle.com/slothkong/10-monkey-species  
Here is an overview of the dataset:  
![Alt text](images_for_readme/image01.PNG?raw=true "Title")  
Also, we can take a look at one of the images in the dataset:  
![Alt text](images_for_readme/image02.PNG?raw=true "Title")
## How do I decide the train/test/validation dataset?
Since the dataset already provides images for training and validation. I decide to choose 70% of the training dataset for training and 30% for validation. Also, I will use the validation images provided as a test dataset to test the accuracy of my trained model.  
## My classification model 1: a normal CNN
Firstly, I designed a CNN model to classify 10 different monkey species  
The structure of my model looks like this:  
![Alt text](images_for_readme/image05.PNG?raw=true "Title")  
I haved trained like 200 epochs and here's the result:  
![Alt text](images_for_readme/image03.PNG?raw=true "Title")  
Seems that my model is overfitting, even though I have used the dropout. Also, the training speed is very low  
## My classification model 2: use Inception V3 Model
Inception-v3 is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012. This is a standard task in computer vision, where models try to classify entire images into 1000 classes, like "Zebra", "Dalmatian", and "Dishwasher".  
The first phase analyzes all the images on disk and calculates and caches the bottleneck values for each of them. 'Bottleneck' is an informal term we often use for the layer just before the final output layer that actually does the classification. (TensorFlow Hub calls this an "image feature vector".) This penultimate layer has been trained to output a set of values that's good enough for the classifier to use to distinguish between all the classes it's been asked to recognize. That means it has to be a meaningful and compact summary of the images, since it has to contain enough information for the classifier to make a good choice in a very small set of values.  
Then, we will use the bottlenecks as the inputs to our own last layer.  
Our own one layer looks like this:  
![Alt text](images_for_readme/image07.PNG?raw=true "Title")  
Here's the training result:
![Alt text](images_for_readme/image04.PNG?raw=true "Title")  
Obviously, it's better and faster.  
For more code detail, you can take a look at the demo.ipynb  

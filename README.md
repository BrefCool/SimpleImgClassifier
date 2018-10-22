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

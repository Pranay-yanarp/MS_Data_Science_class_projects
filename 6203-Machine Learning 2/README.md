# Cotton Plant Disease Prediction

## Summary
This repository contains final project for George Washington University's DATS 6202:Machine Learning II course.
Our project objective was to predict disease in a Cotton plant/leaf.

## Folders
* Code contains all of our code used in the project.
* Group-Proposal contains the proposal for the project.
* Final-Group-Presentation contains a PowerPoint presentation of our project.
* Final-Group-Project-Report contains a report of our findings from this project.

## Code Execution
1. Download the dataset from https://www.kaggle.com/janmejaybhoi/cotton-disease-dataset
2. The 'train_pretrained.py' and 'mannual_cnn_code.py' should be in the same directory as the 'train', 'val' and 'test' folders after you download the data.
3. (FOR train_pretrained.py) The 'train_pretrained.py' script contains training for 3 pretrained models. At the beginning of the code, variable 'model_name' should be set in order to choose which pretrained model to use. (options are 'resnet50', 'vgg16' and 'densenet121'). The end of the script also consists a test section where the path of an unseen image can be given to predict its class.
4. (FOR 'mannual_cnn_code.py') This file is used to create the mannual CNN model along with the performance metrics and graphs.
5. (FOR GUI) The 'Code' folder also contains a folder called 'GUI Code'. In order to run the GUI just run the 'app.py' file and maintain the file and folder structure as it is and once the app.py files runs, you will get an IP, which you will have to paste on the browser.

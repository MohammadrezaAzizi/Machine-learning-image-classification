# This notebook includes
  #Image_classification_ML #Binary_image_classification #ML_image_classification #XGBoost #Random_forest #Decision_Tree #KNN 
  #K-fold_cross_validation #Cross_validation #SVM #support_vector_machine

## Purpose of this notebook is
A binary classification, to test some of machine learning models to classify cataract disease from healthy eyes

# Steps


## Step1 
First renamed pictures (two classes were moved to separate folders before renaming) to the following format : "class name" + "_" + "name of the picture"

## Step2
After that, labeling the images wasn't that hard of a task.

## Step3
In order to fit images to machine learning models that are not designed for working with images, I made features of each image in order to get the best of ML. the features were color histograms of images. feeding images as tensors wasn't feasible as I tried to do so but didn't work. (always bumped to a same error)

## Step4
After making lists of features and labels, our dataset was whole and ready to be used, so I sampled the images via train_test_split

## Step5
Then I fit the data to the model and assessed the predictions of each model using ROC_AUC curves, Confusion matrix, Recall, Precision, Classification_report.

## Step6
There has been a revise on the code. after validation results using cross validation, I preferred to enhance the results using finetuning.
At some point I used randomized search, because Grid search took too long to respond on google colab GPU & TPU.
The sole criterion of the whole finetuning phase was F1_Score. So if there are some perplexing results after Gridsearch that raise the question 
   
   ### (the question) "if No improvement yielded on the accuracy, why bother using gridsearch?" , my answer would be:
   ### (My answer) "Imrovements have been made on F1_score :) "
   
   

Any suggestions or questions are welcome! Good Luck!

## Contact me via:
  ### Linkedin: https://www.linkedin.com/in/mohammadreza-azizi-lnkdin/
  ### e-mail: mo.reza.azizi.1997@gmail.com
  
  

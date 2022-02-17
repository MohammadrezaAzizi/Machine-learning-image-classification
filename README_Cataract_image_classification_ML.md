# Chashmyar
Is a startup based on artificial intelligence healthcare services for eye.

#Image_classification_ML #Binary_image_classification #ML_image_classification #XGBoost #Random_forest #Decision_Tree #KNN #K-fold_cross_validation #Cross_validation #SVM #support_vector_machine

*** A binary classification, Testing machine learning models that are designed to classify cataract disease from healthy eyes.***

1- First renamed pictures (two classes were moved to separate folders before renaming) to the following format : "class name" + "_" + "name of the picture"

2- After that, labeling the images wasn't that hard of a task.

3- In order to fit images to machine learning models that are not designed for working with images, I made features of each image in order to get the best of ML.
  the features were color histograms of images. feeding images as tensors wasn't feasible as I tried to do so but didn't work. (always bumped to a same error)
  
4- After making lists of features and labels, our dataset was whole and ready to be used, so I sampled the images via train_test_split

5- Then I fit the data to the model and assessed the predictions of each model using ROC_AUC curves, Confusion matrix, Recall, Precision, Classification_report. 

  Any suggestions or questions are welcome! 
              Good Luck!

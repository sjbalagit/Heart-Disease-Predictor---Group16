# Heart-Disease-Predictor---Group16

Authors: Shrabanti Bala Joya, Sarisha Das, Omowunmi Obadero, Mantram Sharma

## About

Here we are attempting to build a strong classification model for predicting whether an individual will have heart disease or not. The dataset contain 1026 unique rows, each containing information such as cholestrol, blood pressure, fasting blood sugar, etc for some individual. Our target column contains binary encoding where 1 translates to 'yes, heart disease' and 0, 'no heart disease'. In our exploratory data analysis (EDA), we initiated preprocessing, primarily StandardScaler and then built three different models; Decision Tree, Support Vector Machine and the Dummy Classifier. We were able to use the baseline Dummy Classifier as a reference and then compared the cross-validation scores achieved from the other two models. In the end, we decided to use the Support Vector Machine model for this binary classification problem, as the results we got in terms of test set scores is the highest, 0.9704. It is integral that we make accurate diagnosis on whether a new individual has a heart disease or not based on the features presented, so we believe the X model will provide us with the best diagnosis accuracy/(precision/recall/F1 score).

The data we used has been taken from the Kaggle website ([Download data here](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/tree/main/data/raw/Cardiovascular_Disease_Dataset)). A detailed explanation of all the important features are provided on the Kaggle and GitHub page and you can get a better overview of the summary statistics for all numerical columns in the Datacard section. In our data folder, you can find the raw data in the Cardiovascular_Disease_Dataset folder and the processed data in the processed folder. Our X_train and X_test are represented in X_train.csv and X_test.csv respectively.

## Usage (Attributed from Breast-Cancer-Predictor Project README)
If this is your first time running this project, then run the following from the root of this repository:

conda-lock install --name Heart-Disease-Predictor---Group16 conda-lock.yml
To run the analysis, run the following from the root of this repository:

jupyter lab 
Open notebooks/heart_disease_analysis.ipynb in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:Heart-Disease-Predictor---Group16]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Report

The final report can be found in our [analysis](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/heart_disease_analysis.ipynb).

## Dependencies

- conda (version 23.9.0 or higher)
- conda-lock (version 2.5.7 or higher)
- jupyterlab (version 4.0.0 or higher)
- nb_conda_kernels (version 2.3.1 or higher)
- Python and packages listed in environment.yml

## References

# Heart-Disease-Predictor---Group16

Authors: Shrabanti Bala Joya, Sarisha Das, Omowunmi Obadero, Mantram Sharma

## About

Here we attempt to build a classification model to predict whether an individual is at risk of a heart disease. The dataset contains 1000 unique examples and 14 features containing information on the individuals cholesterol, blood pressure, fasting blood sugar, etc. Our target column contains binary encoding where 1 translates to 'heart disease' and 0 to 'no heart disease'. 

We performed exploratory data analysis (EDA) and applied SciKit Learn's preprocessing tools such as StandardScaler, OneHotEncoder and Ordinal encoder to preprocess the data based on the EDA. We built four different models - Decision Tree, Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel, Logistic Regression and a Dummy Classifier. We used the Dummy Classifier as the baseline and compared cross-validation scores achieved from the other three models. The Support Vector Machine (Classifier) performed reasonably well than the other models with 0.98 test accuracy with recall = 0.98 and precision = 0.98.

It is imperative to ensure accurate diagnosis of heart disease based on a individuals clinical features. Among the evaluated models, we believe that the Support Vector Machine with RBF Kernel will yield the most reliable results as reflected in it's overall performance.

The [dataset](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/tree/main/data/raw/Cardiovascular_Disease_Dataset) used in this project has been obtained from [`Mendeley Data`](https://data.mendeley.com/datasets/dzz48mvjht/1). A detailed explanation of all the important features are provided in our analysis. You can find the raw and processed datasets in the data directory of this repository. Our train and test dataset are represented in [train_heart.csv](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/data/processed/train_heart.csv) and [test_heart.csv](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/data/processed/test_heart.csv) respectively.

## Dependencies
- [Docker](https://www.docker.com/)

## Report

The final report can be found [here](https://sjbalagit.github.io/Heart-Disease-Predictor---Group16/analysis/heart_disease_analysis.html).

## Usage (Attributed from Breast-Cancer-Predictor Project README)

### Setup

> If you are using Windows or Mac, make sure Docker Desktop is running.

1. Clone this GitHub repository.

### Running the analysis

1. Navigate to the root of this project on your computer using the command line and enter the following command:

```
docker compose up
```

2. In the terminal, look for a URL that starts with `http://127.0.0.1:8888/lab?token=` (for an example, see the highlighted text in the terminal below). Copy and paste that URL into your browser.

![Host URL Sample](./images/host_url_sample.png)

3. Open a terminal and execute the following commands to run the analysis:

```
# Step 1: Download and extract data
python scripts/import_data.py \
    --url https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dzz48mvjht-1.zip \
    --write-to data/raw \
    --zip-name dataset.zip
    
# Step 2: Validate data
python scripts/validate_data.py \
    --raw-data data/raw/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv \
    --data-to data/validated

# Step 3: Split + preprocess
python scripts/preprocessing.py \
    --raw-data data/validated/heart_validated.csv \
    --data-to data/processed \
    --preprocessor-to results/preprocessor \
    --seed 123 \
    --split 0.3

# Step 4: Perform EDA 
python scripts/eda.py \
    --data data/processed/train_heart.csv \
    --output-dir results/eda_results \
    --target-col target \
    --num-cols age,resting_bp,serum_cholesterol,max_heart_rate,old_peak \
    --cat-cols gender,chest_pain,fasting_blood_sugar,resting_electro,exercise_angina,slope,num_major_vessels \
    --axis-titles "gender:Gender,chest_pain:Chest Pain Type,fasting_blood_sugar:Fasting Blood Sugar,resting_electro:Resting ECG,exercise_angina:Exercise-Induced Angina,slope:Slope of ST Segment,num_major_vessels:Number of Major Vessels"

# Step 5: Run models
python scripts/evaluate_default_models.py \
    --train-data data/processed/train_heart.csv \
    --target-col target \
    --preprocessor-path results/preprocessor/heart_preprocessor.pickle \
    --pos-label "Heart Disease" \
    --beta 2.0 \
    --random-state 123 \
    --results results/cv_default_models/cv_scores_default_parameters.csv
        
# Step 6: Hyperparameter tuning
python scripts/hyperparameter_tuning.py --train-data data/processed/train_heart.csv \
	--target-col target \
	--preprocessor-path results/preprocessor/heart_preprocessor.pickle \
	--pos-label "Heart Disease" \
	--beta 2.0 \
	--seed 123 \
	--results-to results/final_model_results

# Step 7: Evaluate final model
python scripts/evaluate_scores.py \
    --test-data data/processed/test_heart.csv \
    --target-col target \
    --final-model-path results/final_model_results/final_model.pickle \
    --pos-label "Heart Disease" \
    --beta 2.0 \
    --results-to results/final_model_results   

quarto render analysis/heart_disease_analysis.qmd --to html
```

### Clean up

To shut down the container and clean up the resources, type `Ctrl + C` in the terminal where you launched the container, and then type

```
docker compose rm
```

## Developer notes
### Developer dependencies
- `conda (version 23.9.0 or higher)`
- `conda-lock (version 2.5.7 or higher)`

### Adding a new dependency
1. Add the dependency to the `environment.yml` file on a new branch.

2. To update the conda-linux-64.lock file run the following. 
```
conda-lock -k explicit --file environment.yml -p linux-64
``` 

3. Re-build the Docker image locally to ensure it builds and runs properly.

4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5. Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).

6. Send a pull request to merge the changes into the main branch.


## License
The Heart Disease Predictor report contained in this repository is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please refer to the [license file](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/LICENSE) for full details. If you reuse or adapt any part of this report, kindly provide proper attribution and include a link to this webpage.

The software code included in this repository is licensed under the MIT License. See the [license file](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/LICENSE) for further information.

## References

Ttimbers. (n.d.). TTIMBERS/breast-cancer-predictor. GitHub. <https://github.com/ttimbers/breast-cancer-predictor/tree/main?tab=readme-ov-file>

Doppala, B. P., & Bhattacharyya, D. (2021, April 16). Cardiovascular Disease Dataset (Version 1) [Data Set]. Mendeley Data. <https://doi.org/10.17632/dzz48mvjht.1>

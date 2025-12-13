# Heart-Disease-Predictor---Group16

Authors: Shrabanti Bala Joya, Sarisha Das, Omowunmi Obadero, Mantram Sharma

## About

Here we attempt to build a classification model to predict whether an individual is at risk of a heart disease. The dataset contains 1000 unique examples and 14 features containing information on the individuals' cholesterol, blood pressure, fasting blood sugar, etc. Our target column contains binary encoding where 1 translates to 'heart disease' and 0 to 'no heart disease'. 

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

3. Open a terminal and execute the following commands from the `root` to run the analysis:
3a. Reset the project to a clean state (remove all generated files from the analysis)
```
make clean
```
3b. Run the analysis in its entirity, including generation of new HTML report, run the following:
```
make all 
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

### Running the analysis on local

1. Open a terminal in the root of the folder. Please make sure conda and conda-lock is installed and the base environment is activated. To ensure the base is active run:
```
conda activate base
```

2. Run the following to create a new environment for the analysis (Replace `<env_name>` with a relevent name for your new environment)
```
conda-lock install --name <env_name> conda-lock.yml
```
Please wait a while for the packages to download.

3. Now activate the new environment using
```
conda activate <env_name>
```
4. Now in the terminal execute the following commands from the `root` to run the analysis:

Reset the project to a clean state (remove all generated files from the analysis)
```
make clean
```
Run the analysis in its entirity, including generation of new HTML report, run the following:

```
make all 
```

### Adding a new dependency
1. Add the dependency to the `environment.yml` file on a new branch.

2. To update the conda-linux-64.lock file run the following. 
```
conda-lock -k explicit --file environment.yml -p linux-64
``` 
_Note: This may create additional lockfiles for multiple OS types, please ignore/delete the irrelevent lock-files._

3. Re-build the Docker image locally to ensure it builds and runs properly. Replace `<your_tag>` with a tag of your choice.

```
docker build --tag <your_tag> .
```

4. Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

5. Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).

6. Send a pull request to merge the changes into the main branch.


## License
The Heart Disease Predictor report contained in this repository is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please refer to the [license file](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/LICENSE) for full details. If you reuse or adapt any part of this report, kindly provide proper attribution and include a link to this webpage.

The software code included in this repository is licensed under the [MIT License](https://mit-license.org/)  See the [license file](https://github.com/sjbalagit/Heart-Disease-Predictor---Group16/blob/main/LICENSE) for further information.

## References

Ttimbers. (n.d.). TTIMBERS/breast-cancer-predictor. GitHub. <https://github.com/ttimbers/breast-cancer-predictor/tree/main?tab=readme-ov-file>

Doppala, B. P., & Bhattacharyya, D. (2021, April 16). Cardiovascular Disease Dataset (Version 1) [Data Set]. Mendeley Data. <https://doi.org/10.17632/dzz48mvjht.1>

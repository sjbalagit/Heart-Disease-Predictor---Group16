# =========================================================
# 1. Download and extract data
# =========================================================
data/raw/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv : scripts/import_data.py
	python scripts/import_data.py \
		--url https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/dzz48mvjht-1.zip \
		--write-to data/raw \
		--zip-name dataset.zip

# =========================================================
# 2. Validate data
# =========================================================
data/validated/heart_validated.csv : scripts/validate_data.py data/raw/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv
	python scripts/validate_data.py \
		--raw-data data/raw/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv \
		--data-to data/validated

# =========================================================
# 3. Split + preprocess
# =========================================================
PREPROC_OUTPUTS = \
	data/processed/train_heart.csv \
	data/processed/test_heart.csv \
	data/processed/heart_train_preprocessed.csv \
	data/processed/heart_test_preprocessed.csv \
	results/preprocessor/heart_preprocessor.pickle

$(PREPROC_OUTPUTS) : scripts/preprocessing.py data/validated/heart_validated.csv
	python scripts/preprocessing.py \
		--raw-data data/validated/heart_validated.csv \
		--data-to data/processed \
		--preprocessor-to results/preprocessor \
		--seed 123 \
		--split 0.3

# =========================================================
# 4. Perform EDA
# =========================================================
EDA_OUTPUTS = \
	results/eda_results/boxplots_vs_target.png \
	results/eda_results/categorical_vs_target.png \
	results/eda_results/correlation_heatmap.png \
	results/eda_results/numerical_feature_distributions.png \
	results/eda_results/summary_statistics.csv \
	results/eda_results/target_distribution.png

$(EDA_OUTPUTS) : scripts/eda.py data/processed/train_heart.csv
	python scripts/eda.py \
		--data data/processed/train_heart.csv \
		--output-dir results/eda_results \
		--target-col target \
		--num-cols age,resting_bp,serum_cholesterol,max_heart_rate,old_peak \
		--cat-cols gender,chest_pain,fasting_blood_sugar,resting_electro,exercise_angina,slope,num_major_vessels \
		--axis-titles "gender:Gender,chest_pain:Chest Pain Type,fasting_blood_sugar:Fasting Blood Sugar,resting_electro:Resting ECG,exercise_angina:Exercise-Induced Angina,slope:Slope of ST Segment,num_major_vessels:Number of Major Vessels"

# =========================================================
# 5. Run models
# =========================================================
results/CV_scores_default_parameters.csv : scripts/evaluate_default_models.py data/processed/train_heart.csv results/preprocessor/heart_preprocessor.pickle
	python scripts/evaluate_default_models.py \
		--train-data data/processed/train_heart.csv \
		--target-col target \
		--preprocessor-path results/preprocessor/heart_preprocessor.pickle \
		--pos-label "Heart Disease" \
		--beta 2.0 \
		--random-state 123 \
		--results results/CV_scores_default_parameters.csv

# =========================================================
# 6. Hyperparameter tuning
# =========================================================
HPT_OUTPUTS = \
    results/final_model_results/final_model.pickle \
    results/final_model_results/hyperparameter_model_results.csv

$(HPT_OUTPUTS): scripts/hyperparameter_tuning.py $(PREPROC_OUTPUTS)
	python scripts/hyperparameter_tuning.py --train-data data/processed/train_heart.csv \
	--target-col target \
	--preprocessor-path results/preprocessor/heart_preprocessor.pickle \
	--pos-label "Heart Disease" \
	--beta 2.0 \
	--seed 123 \
	--results-to results/final_model_results

# =========================================================
# 7. Evaluate final model
# =========================================================
EVAL_OUTPUTS = \
    results/final_model_results/evaluate_model_results.csv \
    results/final_model_results/confusion_matrix.png

$(EVAL_OUTPUTS): scripts/evaluate_scores.py data/processed/test_heart.csv results/final_model_results/final_model.pickle
	python scripts/evaluate_scores.py \
		--test-data data/processed/test_heart.csv \
		--target-col target \
		--final-model-path results/final_model_results/final_model.pickle \
		--pos-label "Heart Disease" \
		--beta 2.0 \
		--results-to results/final_model_results

# =========================================================
# Default target
# =========================================================
all: $(PREPROC_OUTPUTS) $(EDA_OUTPUTS) results/CV_scores_default_parameters.csv $(HPT_OUTPUTS) $(EVAL_OUTPUTS)
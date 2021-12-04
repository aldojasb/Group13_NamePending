all: doc/Quality_white_wine_predictor.html doc/Quality_white_wine_predictor.md

# Download the data
data/raw/winequality/winequality-white.csv:
	python src/download_data.py --url=http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --path=data/raw/

# Split into train and test sets
data/processed/X_test.csv data/processed/X_train.csv data/processed/y_test.csv data/processed/y_train.csv: data/raw/winequality/winequality-white.csv
	python src/split.py data/raw/winequality/winequality-white.csv data/processed

# Train models
results/raw_results: data/processed/X_test.csv data/processed/X_train.csv data/processed/y_test.csv data/processed/y_train.csv
	python src/ml_models.py data/processed results/raw_results

# Perform EDA
relationship_between_individual_features_and_the_quality_3.png Distribution_of_white_wine_quality.png relationship_between_individual_features_and_the_quality_1.png relationship_between_individual_features_and_the_quality_2.png: data/processed/X_test.csv data/processed/X_train.csv data/processed/y_test.csv data/processed/y_train.csv
	python src/EDA.py data/processed/X_train.csv data/processed/y_train.csv results

# Evaluate the models
results/best_model.csv: results/raw_results
	python src/analyze.py --r_path=results

# Generate report
doc/Quality_white_wine_predictor.html doc/Quality_white_wine_predictor.md: results/best_model.csv relationship_between_individual_features_and_the_quality_3.png Distribution_of_white_wine_quality.png relationship_between_individual_features_and_the_quality_1.png relationship_between_individual_features_and_the_quality_2.png
	Rscript -e "rmarkdown::render('doc/Quality_white_wine_predictor.Rmd')"

clean:
	rm -rf data/*
	rm -rf results/*
	rm -f doc/Quality_white_wine_predictor.html
	rm -f doc/Quality_white_wine_predictor.md

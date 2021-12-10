# Predicting the quality of white wine

Authors: Aldo Saltao Barros, Nikita Shymberg, Yair Guterman, Son Chau

## Goal

This project aims to determine a good model to predict wine quality given measurable wine features.

## Introduction

According to experts, wine is differentiated according to its smell, flavour, and colour, but most people are not wine experts to say that wine is good or bad. The quality of the wine is determined by many variables including, but not limited to, the ones mentioned previously. The quality of a wine is important for the consumers as well as the wine industry. For instance, industry players are using product quality certifications to promote their products. However, this is a time-consuming process and requires the assessment given by human experts, which makes this process very expensive. Nowadays, machine learning models are important tools to replace human tasks and, in this case, a good wine quality prediction can be very useful in the certification phase. For example, an automatic predictive system can be integrated into a decision support system, helping the speed and quality of the performance.

## Dataset description

The wine quality dataset is publicly available on the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The dataset has two files, red wine and white wine variants of the Portuguese “Vinho Verde” wine. It contains a large collection of datasets that have been used for the machine learning community. The red wine dataset contains 1599 instances and the white wine dataset contains 4898 instances. Both files contain 11 input features and 1 output feature. Input features are based on the physicochemical tests and output variable based on sensory data is scaled in 11 quality classes from 0 to 10 (0-very bad to 10-very good).

Input variables:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol

Output variable:

12. quality (score between 0 and 10)

## How to analyze the data

Our task here is to focus on what white wine features are important to get the most promising result. For the purpose of classification model and evaluation of the relevant features, we are using algorithms such as 1) Decision Tree, 2) SVC, 3) K-NN, 4) Navie Bayes, and 5) Logistic Regression. We will test them and compare what is the best model and the best parameters to use in this prediction.

## EDA first conclusions

According to our first EDA, we do not have a balanced database, our wines are concentrated around quality 5 and 7.5 (around 80% of data points). Besides, we have a couple of signs about some variables. For instance, it appears that the higher the alcohol level, the better the wine quality. Additionally, the smaller the chlorides and total sulphur dioxide the better the wine quality. Some variables seem do not influence wine quality on their own. When combining these variables, they might indeed influence wine quality.

## Usage

Running with Docker:

Make sure to install docker then run the following commands:

```bash
docker build --tag v0.1.0 /$(pwd)
docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}":/home/jovyan/work v0.1.0 make -C /home/jovyan/work all
```

Download the data:

```bash
python src/download_data.py --url=http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip --path=data/raw/
```

Split into train and test sets:

```bash
python src/split.py data/raw/winequality/winequality-white.csv data/processed
```

Train models:

```bash
python src/ml_models.py data/processed results/raw_results
```

Perform EDA:

```bash
python src/EDA.py data/processed/X_train.csv data/processed/y_train.csv results
```

Evaluate the models:

```bash
python src/analyze.py --r_path=results
```

## License

The Quality white wine predicto materials here are licensed under MIT License, Copyright (c) 2021 Master of Data Science at the University of British Columbia. If re-using/re-mixing please provide attribution and link to this webpage.

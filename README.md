# Quality white wine predictor
Authors: Aldo Saltao Barros, Nikita Shimberg, Yair Guterman, Son Chou

## About 

**Goal**
This project aims to determine which features are the best quality white wine indicators and generate insights into each of these factors to our model’s wine quality.

**Introduction**
According to experts, wine is differentiated according to its smell, flavour, and colour, but most people are not wine experts to say that wine is good or bad. Those variables and others not mentioned so far compose what we define as the quality of the wine. The quality of a wine is important for the consumers as well as the wine industry. For instance, industry players are using product quality certifications to promote their products. However, this is a time-consuming process and requires the assessment given by human experts, which makes this process very expensive. Nowadays, machine learning models are important tools to replace human tasks and, in this case, a good wine quality prediction can be very useful in the certification phase. For example, an automatic predictive system can be integrated into a decision support system, helping the speed and quality of the performance.

**Source**
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


**How to analyze the data**
Our task here is to focus on what white wine features are important to get the promising result. For the purpose of classification model and evaluation of the relevant features, we are using algorithms such as 1) Decision Tree, 2) SVC, 3) K-NN, 4) Navie Bayes, and 5) Logistic Regression. We will test them and compare what is the best model and the best parameters to use in this prediction.


**EDA first conclusions**

According to our first EDA, we do not have a balanced database, our wines are concentrated around quality 5 and 7.5 (around 80% of data points). Besides, we have a couple of signs about some variables. For instance, it appears that the higher the alcohol level, the better the wine quality. Additionally, the smaller the chlorides and total sulphur dioxide the better the wine quality. Some variables seem do not to influence wine quality, such as free sulphur dioxide, fixed acidity, and sulphates.

## License

The Quality white wine predicto materials here are licensed under MIT License, Copyright (c) 2021 Master of Data Science at the University of British Columbia. If re-using/re-mixing please provide attribution and link to this webpage.

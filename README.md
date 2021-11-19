## Quality white wine predictor
Authors: Aldo Saltao Barros, Nikita Shimberg, Yair Guterman, Son Chou

## About 

\underline{Goal}
This project aims to determine which features are the best quality white wine indicators and generate insights into each of these factors to our model’s wine quality.

\underline{introduction}
According to experts, the wine is differentiated according to its smell, flavor, and color, but most people are not a wine expert to say that wine is good or bad. Those variables and others not mentioned so far compose what we define as the quality of the wine. The quality of a wine is important for the consumers as well as the wine industry. For instance, industry players are using product quality certifications to promote their products. However, this is a time-consuming process and requires the assessment given by human experts, which makes this process very expensive. Nowadays, machine learning models are important tools to replace human tasks and, in this case, a good wine quality prediction can be very useful in the certification phase. For example, an automatic predictive system can be integrated into a decision support system, helping the speed and quality of the performance.

Our task here is focusing on what white wine features are important to get the promising result. For the purpose of classification model and evaluation of the relevant features, we are using algorithms such as 1) Random Forest 2) SVC and 3) Logistic Regression

\underline{Source}
The wine quality dataset is publicly available on the UCI machine learning repository (https://archive.ics.uci.edu/ml/datasets/Wine+Quality). The dataset has two files, red wine and white wine variants of the Portuguese “Vinho Verde” wine. It contains a large collection of datasets that have been used for the machine learning community. The red wine dataset contains 1599 instances and the white wine dataset contains 4898 instances. Both files contain 11 input features and 1 output feature. Input features are based on the physicochemical tests and output variable based on sensory data is scaled in 11 quality classes from 0 to 10 (0-very bad to 10-very good).

Input variables:
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol

Output variable:
quality (score between 0 and 10)

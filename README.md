# statistical-machine-learning
## Do (wo)men talk too much in films? Project in Machine Learning
October 21, 2022
Version 3.0

### Department of Information Technology
### Uppsala University

### Abstract
In this report, we present an analysis of Hollywood movies based on a given
data set. We use multiple machine-learning methods to predict the gender of the
lead actor based on 13 features using python programming language. Multiple
parameters are tuned in each model to achieve the highest accuracy. The chosen
model, deep neural networks, is then used to predict the gender of the lead actor
from an unknown test data set.

### Introduction
There have been allegations in Hollywood of sexism and racism in movie roles, in which white men
dominate movie roles. A motivated team gathered data from eight thousand (8,000) screenplays
segregated into gender and age, then analyzed it to confirm the allegations [1]. In this project, we
set out to produce machine learning models that can predict the gender of the lead role using data
such as the year the movie was released, the number of female actors, profits made by the film,
the number of words for each gender, among others. The algorithms used were logistic regression,
discriminant analysis, K-nearest neighbors, random forests (tree-based methods), boosting, and deep
neural networks. The highest-performing algorithm will be used to classify an unknown test set.

### Data Pre-processing
#### Feature Selection
A fundamental approach to tackling supervised learning problem is feature engineering and selection.
One way to remove noise from the data is to identify linear dependencies in the data and drop highly
dependent features. To achieve this, we use correlation to determine the linear dependencies and then
prune the features.
From Fig. 4, the features "Total words", "Difference in words lead and co-lead", and "Number of
words lead" have a high correlation. After some testing, we decided to drop the "Total words" feature.
![alt text](https://github.com/Dna072/statistical-machine-learning/blob/master/feature_corr.png)


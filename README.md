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
![alt text](https://github.com/Dna072/statistical-machine-learning/blob/master/feature_corr.png "Figure 4: Grid of correlations between input columns")

#### Feature Scaling
Features may have varying values, affecting how well the model learns from the data. Using
StandardScaler from sklearn, we scale the training data points to help the model understand and learn
from the training data.

### Linear and quadratic discriminant analysis
Linear and quadratic discriminant models are generative models derived from the Gaussian mixture 
model (GMM) [2]. They provide a probabilistic description of how to generate the input and output
data. Since the logarithm of the Gaussian probability density function is a quadratic function in x,
the decision boundary for this classifier is also quadratic, and the method is referred to as quadratic
discriminant analysis (QDA). Making an additional simplifying assumption, we instead obtain the
linear discriminant analysis (LDA).

#### LDA
#### Training the model
Here are outlined the steps taken to train and find the best model:
• Create a train-test-split from the data with our selected features. The test data size is set to 
10%, and the train data size at 90%. We will run 12-fold cross-validation on the training set, so we picked a small size for the final test set.
• Try out multiple parameters using a pipeline and apply grid search cross-validation to run the algorithm for each parameter and return the best estimator model. We experimented with the following:
  – solver = [’lsqr’, ’eigen’] and ’svd’: Describes the behavior of a matrix on a particular set of vectors. The ’svd’ solver does not rely on the covariance matrix and may be preferable when the number of features is extensive. Since it does not support shrinkage, it was done separately. The ‘lsqr’ solver is an efficient algorithm that only works for classification. The ‘eigen’ solver is based on optimizing the between-class scatter to the within-class scatter ratio.
 – shrinkage = [’auto’]: A shrinkage is a form of regularization used to improve the estimation of covariance matrices. It is useful in situations where the number of training samples is small compared to the number of features. Each cross-validation fold uses the same approach so the overall train time is increased.

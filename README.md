# statistical-machine-learning
## Do (wo)men talk too much in films? Project in Machine Learning
October 21, 2022
Version 3.0

### Department of Information Technology
### Uppsala University

### 1. Abstract
In this report, we present an analysis of Hollywood movies based on a given
data set. We use multiple machine-learning methods to predict the gender of the
lead actor based on 13 features using python programming language. Multiple
parameters are tuned in each model to achieve the highest accuracy. The chosen
model, deep neural networks, is then used to predict the gender of the lead actor
from an unknown test data set.

### 2. Introduction
There have been allegations in Hollywood of sexism and racism in movie roles, in which white men
dominate movie roles. A motivated team gathered data from eight thousand (8,000) screenplays
segregated into gender and age, then analyzed it to confirm the allegations [1]. In this project, we
set out to produce machine learning models that can predict the gender of the lead role using data
such as the year the movie was released, the number of female actors, profits made by the film,
the number of words for each gender, among others. The algorithms used were logistic regression,
discriminant analysis, K-nearest neighbors, random forests (tree-based methods), boosting, and deep
neural networks. The highest-performing algorithm will be used to classify an unknown test set.

### 3. Data Pre-processing
#### 3.1 Feature Selection
A fundamental approach to tackling supervised learning problem is feature engineering and selection.
One way to remove noise from the data is to identify linear dependencies in the data and drop highly
dependent features. To achieve this, we use correlation to determine the linear dependencies and then
prune the features.
From Fig. 4, the features "Total words", "Difference in words lead and co-lead", and "Number of
words lead" have a high correlation. After some testing, we decided to drop the "Total words" feature.
![alt text](https://github.com/Dna072/statistical-machine-learning/blob/master/feature_corr.png "Figure 4: Grid of correlations between input columns")

#### 3.2 Feature Scaling
Features may have varying values, affecting how well the model learns from the data. Using
StandardScaler from sklearn, we scale the training data points to help the model understand and learn
from the training data.

### 4. Linear and quadratic discriminant analysis
Linear and quadratic discriminant models are generative models derived from the Gaussian mixture 
model (GMM) [2]. They provide a probabilistic description of how to generate the input and output
data. Since the logarithm of the Gaussian probability density function is a quadratic function in x,
the decision boundary for this classifier is also quadratic, and the method is referred to as quadratic
discriminant analysis (QDA). Making an additional simplifying assumption, we instead obtain the
linear discriminant analysis (LDA).

#### 4.1  LDA
#### Training the model
Here are outlined the steps taken to train and find the best model:
• Create a train-test-split from the data with our selected features. The test data size is set to 
10%, and the train data size at 90%. We will run 12-fold cross-validation on the training set, so we picked a small size for the final test set.<br/>
• Try out multiple parameters using a pipeline and apply grid search cross-validation to run the algorithm for each parameter and return the best estimator model. We experimented with the following:
  * solver = [’lsqr’, ’eigen’] and ’svd’: Describes the behavior of a matrix on a particular set of vectors. The ’svd’ solver does not rely on the covariance matrix and may be preferable when the number of features is extensive. Since it does not support shrinkage, it was done separately. The ‘lsqr’ solver is an efficient algorithm that only works for classification. The ‘eigen’ solver is based on optimizing the between-class scatter to the within-class scatter ratio.
 * shrinkage = [’auto’]: A shrinkage is a form of regularization used to improve the estimation of covariance matrices. It is useful in situations where the number of training samples is small compared to the number of features. Each cross-validation fold uses the same approach so the overall train time is increased.

Each cross-validation fold uses the same approach so the overall train time is increased.

#### 4.2 QDA
#### Training the model
• The train-test-split step is the same as in Section 4.1
• The pipeline approach was implemented here. We experimented with the following:
  * reg_param = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]: Regularization used to improve the estimation.

Each cross-validation fold uses the same approach, increasing the overall train time. The
73 only solver available for QDA is ’svd’.

#### 4.3 Best estimator parameters for LDA and QDA
The parameters for the best estimator of LDA we found are solver = ’lsqr’ and shrinkage = ’auto’,
with an accuracy of 0.875, i.e., 87.5%. For QDA, the best parameter is reg_param = 0.0, with an
accuracy of 0.904, i.e., 90.4%. For being more flexible, the QDA model provided a better result than
(or at least as good as) the LDA model.

### 5. K nearest neighbors
The KNN algorithm tries to find only one k value, the number of nearest neighbors with the least
error on a given training set. Once k is found, given a test input, a majority vote is carried out for
classification problems and the average for regression problems. KNN is known to perform better on
classification problems, in which it is recommended to use an odd value for k since a majority vote
will always predict one class. In case k is even, we can recalculate the distances and predict the class
closest to k or pick one of the classes.

#### 5.1 Training the model
• The train-test-split step is the same as in Section 4.1 <br/>
• The pipeline approach was implemented here. We experimented with various parameters:
 * {k : k ∈ [1, 30]}: Values which k can take. For each value, find the error and select the value of k with the least error.
 * weights = [’uniform’, ’distance’]: For each value of k, apply “uniform” weights to all data points such that all points have the same influence on a test point; and “distance” weight which is closer points have stronger influence to a test point.
 * metrics = [’minkowski’, ’manhattan’, ’euclidean’]: For each value of k and weights, run the metrics used for the distance calculation between point.. In essence, it tries the different distance calculation methods and returns the best one that minimizes the error in the classification.

Each cross-validation fold uses the same approach, increasing the overall train time.

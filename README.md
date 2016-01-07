# Driver-Telematics Analysis 

In this projet, we worked on the [Kaggle challenge](https://www.kaggle.com/c/axa-driver-telematics-analysis) to identify a driver signature using telematic data. The dataset consists of 50,000 anonymized driver trips provided by AXA. 

We devised a telematic fingerprint for each driver and evaluated the different feature selection/dimensionality reduction methods as well as classification algorithms. This telematic fingerprint would then be able to predict if the given trip was undertaken by the designated driver.

# Feature Extraction
We generated the following group of features based on 
- Distance
- Speed, Acceleration and Jerk 
- Angle of travel

For each group, we generated the mean, standard deviation, percentiles and other related features. The total feature set consisted of 92 features for each trip.

# Dimensionality Reduction
We evaluated the use of the following dimensionality reduction techniques -
- Principal Component Analysis 
- Independent Component Analysis 
- Linear Discriminant Analysis

PCA was the best performing dimensionality reduction technique.

# Classification
We also compared the performance of the following four classifiers -
- Support Vector Machines
- Logistic Regression
- Random Forest
- Gradient Boosted Regression Trees

For each of the classifiers, we evaluated the performance on the full feature set as well as on the reduced feature set.

In our evaluation, Random Forest classifier produced the best leaderboard score of **0.9126** on Kaggle.


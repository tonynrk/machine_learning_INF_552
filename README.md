# machine_learning_INF_552
These are Machine Learning INF 552 Work Projects in Summer 2019, USC. The intuition is to understand the concept of Machine learning models and apply then to real-world data.

# Programming Language

Python

# Libraries For Work Projects

Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn

# Summary For Work Projects

1.)	Vertebral Comlumn Data Set. (File: INF_552_Work_Project_01.ipynb) This Biomedical data set was built by Dr. Henrique da Mota during a medical residence period in Lyon, France. Each patient in the data set is represented in the data set by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (in this order): pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius and grade of spondylolisthesis. 

Task: a binary classification task, is a patient gene’s normal or abnormal. 
Preprocessing and Exploratory data analysis: doing scatter plot and boxplot for each of the independent variables. 
Classification: by KNN  with Euclidean metrics, Minkowski Distance, Mahalanobis Distance, find the suitable k from 208,205,…7,4,1 by see train error and test error. learn the curve (best test error rate vs Size of Training Set)

2.)	Combined Cycle Power Plant Data Set. (File: INF_552_Work_Project_02.ipynb) The dataset contains data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant. 
Making scatter plots, what their mean, the median, range, first and third quartiles, and interquartile ranges of each of the variables in the dataset? 

Task: a simple linear regression model to each variable to predict the response. Fit a multiple regression model. Using possible interaction terms or nonlinear associations between the predictors and response, find the significant features to predict the response, and improve model by using only those features. (j)  Compare the results of KNN Regression with the linear regression model that has the smallest test error and provide your analysis. 

3.)	Time Series Classification. (File: INF_552_Work_Project_03.ipynb) An interesting task in machine learning is classification of time series. In this problem, we will classify the activities of humans based on time series obtained by a Wireless Sensor Network.

Task: Feature Extraction: Classification of time series usually needs extracting features from them. In this problem, we focus on time-domain features. 
Binary Classification Using Logistic Regression: Using cross validation to determine the best no. of features, using backward selection algorithm. Learning how to encounter class imbalance. Compare the results with logistic Regression with L1-penalized logistic regression. Also, compare the results between L1- penalized multinomial regression model, and Na ̈ıve Bayes’ classifier (Gaussian and Multinomial priors)

4.)	File: INF_552_Work_Project_04.ipynb
4.a)  Communities and Crime data Communities within the United States. The data combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR

Task: dealing the missing values, Fit a linear regression model, ridge regression with λ chosen by cross-validation model and Lasso model with list of the variables selected by the model. Fit a PCR model on the training set, with M (the number of principal components) chosen by cross-validation, L1-penalized regression gradient boosting tree, Determine α (the regularization term) using cross-validation 

4.b) APS Failure at Scania Trucks Data Set, The datasets' positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS.

Task: Train a random forest to classify the data set. Do and don’t compensate for class imbalance in the data set. Using Weka (Java library) to train Logistic Model Trees for classification. Do and don’t compensate for class imbalance in the data set.

5.)	File: INF_552_Work_Project_05.ipynb
5.a) The Anuran Calls (MFCCs) Data Set Acoustic features extracted from syllables of anuran (frogs) calls, including the family, the genus, and the species labels (multilabel).

Task:  Train a SVM for each of the labels, using Gaussian kernels and one versus all classifiers. Determine the weight of the SVM penalty and the width of the Gaussian Kernel using 10 fold cross validation Train. L1-penalized SVMs, remember to standardize the attributes. Determine the weight of the SVM penalty using 10 fold cross validation, using SMOTE or any other method you know to remedy class imbalance.

5.b) K-Means Clustering on a Multi-Class and Multi-Label the Anuran Calls (MFCCs) Data Set 

Task: Use k-means clustering on the whole data Set Choose k ∈ {1, 2, . . . , 50} automatically based on Silhouettes method. In each cluster, determine which family is the majority by reading the true labels. Repeat for genus and species. Now for each cluster you have a majority label triplet (family, genus, species). Calculate the average Hamming distance, Hamming score, and Hamming loss between the true labels and the labels assigned by clusters.

6.) File: INF_552_Work_Project_06.ipynb	
6.a) the Breast Cancer Wisconsin (Diagnostic) Data Set, classes (Benign=B, Malignant=M), and 30 attributes

Task: Supervised Learning, Semi-Supervised Learning and unsupervised learning, spectral clustering methods. 

6.b) the banknote authentication Data Set

Task: Understand active learning and passive learning algorithm by training with SVM.

# Customer-segmentation

Customer Segmentation is the concept of understanding a company's target audience. This approach
helps a business know it's customers profiles,interests,preferences and choices and helps in the growth 
of the company. Generally, there are many Unsupervised Machine Learning Algorithms which help us implement
customer segmentation. 
In this package,we have implemented a K-means clustering algorithm which groups all 
the data available into sub-groups which are distinct from each other. This algorithm can take in unlabelled
customer data and assign each data point to clusters. Generally, the steps that are included in this concept are 1) Data 
Preprocessing for K-means clustering 2)Building a K-means clustering algorithm 3) The metrics
used for evaluating the performance of a clustering model 4) Visualizing the clusters built. The code in this package
represents all the steps in customer segmentation, in which the parameters can be modified according to the 
user requirements.

# Dataset

We have used two datasets as test cases to our package:
1) Mall Customers Data:This dataset has attributes Age,Spending Score,Annual Income,Gender and a unique Customer ID for every customer.
2) Credit Card Customer Data: This dataset contains attributes Serial no, Customer key, Average Credit limit, Total Credit cards 
   Total visits to bank, Total visits online and Total calls made.


# Walk-through of the code:

The overall purpose of this package is to implement ‘Customer Segmentation using K-means Algorithm’. This package fulfills the purpose of segmenting customers into groups or clusters and this clustering is done based on customers having similar characteristics which include geography, behavioral, customer content etc. There are two modules in this package: 1) Main 2) Clustering.
Main module has a class named Analysis where the user fits in his/her dataset for implementing segmentation. Clustering module has a class named Clustering where all the functions of segmentation have been implemented. In Clustering class, there are functions like kmeans(), clusteringmodel1(), clusteringmodel2(), plots_clusters()  which are quite vital in the program without which the purpose of the package is lost, since kmeans() is used for implementing the K-means algorithm, clusteringmodel1() and clusteringmodel2() functions are used for implementing two different models of creating the clusters and plots_clusters() is a function where both the clustering methods are compared and the plot of the better and optimal clustering method is displayed.
Analysis class, in Main module, consists of mall_customers() function which invokes all the functions from Clustering class in Clustering module. Clustering class, in Clustering module, consists of the following functions:
### __init__() 
 Initializes the clustering class with given dataframe and range of clusters.
### visualizations ()
 This function prints the distribution plot of the attribute passed as a parameter to this function. Ex: Spending Score
### kmeans ()
 This function takes a dataframe as input and calculates the inertia scores for data points and returns a list containing those inertia scores.
### print_clusters()
 This function prints the cluster points and their respective inertia scores.
### optimal_number_of_clusters()
 Calculates the optimal number of clusters by using inertia scores as parameter and returns the optimal number of clusters.
### ClusterModel1()
 This function uses the output of kmeans (), builds a model for segmentation and the silhouette score of this model is calculated.
### ClusterModel2()
 This function uses Principal Component Analysis, builds a model for segmentation and the silhouette score of this model is calculated.
### plots_Clusters()
 This function compares the silhouette score of both the cluster models and prints the clusters of the model with highest silhouette score.
### plots_Analysis()
 This function compares the silhouette score of both the cluster models and prints bar plots for cluster analysis.

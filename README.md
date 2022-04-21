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

#Walk-through of the code:

Two classes have been created here: 1) Analysis 2) Clustering. The Analysis class has all the functions of the Clustering class which
when invoked, performs aall the functionalities of the K-means algorithm. In order to begin the process of customer segmentation, a 
dataset will be given by the user in the Analysis class wich will be converted to a dataframe. In this class, Data Preprocessing step
takes place, where if any null values are present in the dataset, the fillna() function will convert all the null values in the dataset 
to 0 and updates the dataset. Also,categorical data in the dataset are replaced by integer values in the dataset using the replace()
function. The user can select the  number of clusters he/she requires by giving the minimum and maximum range of clusters. To display
the clustering models, we call the functions clusteringmodel1() and clusteringmodel2() functions and the user gives the attriutes of the 
dataset according to which segmentation is performed. There are other functions in the Analysis class which play a vital role in performing
the segmentation. These functions are:
->k-means(): This function accepts the range of clusters given by the user and calculates the inertia score of the points and accordingly
displays the plots of the clusters.
->calculate_wcss(): This function calculates Within-clusters-sum of-squares, which is the sum of the squared distance between the centroid
of the clusters and the points and appends the scores to a list.
->Optimal_number_of_clusters(): This function takes a list containing the within clusters sum-of-squares for each 
number of clusters that we calculated using the calculate_wcss() method, and as a result, it gives back the optimal number of clusters. 
->clusteringmodel1(): This function does the segmentation based on the optimal number of clusters and takes in two columns of the dataset
 given by the user to display the clusters.
->clusteringmodel2():  This function does the segmentation based on PCA (Principle Component Analysis) technique to reduce the dimensions of the dataset.
The reduced dimensions explain the maximum variance in the model.The two components which have the highest variance will be used to generate the clusters 
and be displayed. 
->: Based on the silhoutte score of the models, the model with highest silhoutte score will be used to interpret the different customer segments.
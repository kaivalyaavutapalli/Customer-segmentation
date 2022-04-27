# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score

class Clustering:
    """
    Clustering class takes in customer data and use k-means
    clustering algorithm and assign each data point to clusters.
    """
    silhouette_Model1=0
    silhouette_Model2=0
    
    clusters_Model2= None
    clusters_Model1=None
    
    prediction_Model2=None
    prediction_Model1=None
    
    def __init__(self,dataFrame,min_Range,max_Range):
        
        """
        Initialize the clustering class with given dataframe and cluster range
        
        """
        
        self.dataFrame=dataFrame
        
        # minimum number of clusters
        self.min_Range=min_Range
        
        # maximum number of clusters
        self.max_Range=max_Range
        
    def visualizations(self, component_Name):
        
        """
        visualizations takes an attribute as input and prints the distribution plot of that attribute
        param : attribute of a dataset
        prints the distribution plot
        returns: None
        
        """
        
        sns.distplot(self.dataFrame[component_Name],color= 'blue',bins=20)
        plt.title('distribution plot', fontsize = 15)
        plt.xlabel(component_Name, fontsize = 12)
        plt.ylabel('Count', fontsize = 12)
        plt.show()
        
    def kmeans(self):
        
        """
        kmeans takes a dataframe as input and calculates the inertia scores for data points and returns a list containing inertia scores
        input: dataframe and minimum and maximum number of clusters
        returns: list of inertia scores
        
        """
        minimumRange = self.min_Range
        maximumRange = self.max_Range

        inertiaScore= []
        
        # initializes the range of clusters
        rangeList = range(minimumRange, maximumRange)

        # calculates the inertia score and append to inertiaScore list
        for k in rangeList:
            kmeans = KMeans(n_clusters = k, init='random', random_state= 0)
            kmeans.fit(self.dataFrame) 
            score = kmeans.inertia_
            inertiaScore.append(score)
        return(inertiaScore)
        
    def print_clusters(self):
        
        """
        print_clusters prints the cluster points and inertia scores.
        input: cluster range and inertia scores from kmeans()
        returns: None
    
        """
        minimumRange = self.min_Range
        maximumRange = self.max_Range
        inertiaScore= self.kmeans()
        
        plt.figure(1,figsize = (10 ,8))
        plt.plot(np.arange(minimumRange , maximumRange) , inertiaScore , '-' ,marker='o', alpha = 0.5)
        plt.xlabel('Number of Clusters') , plt.ylabel('Inertia score')
        plt.show()

    def optimal_number_of_clusters(self,inertia_Score):
        
        """
        Calculates the optimal number of clusters.
        param inertia_score: A list of inertia scores obtained by invoking kmeans()
        returns: The optimal number of clusters
        
        """
       
        min_range = self.min_Range
        max_range = self.max_Range
        
        x1, y1 = min_range, inertia_Score[0]
        x2, y2 = max_range, inertia_Score[len(inertia_Score)-1]
        distances_List= []
        
        # calculates the absolute value of sum of squares of data points 
        for i in range(len(inertia_Score)):
            variable1 = i+2
            variable2 = inertia_Score[i]
            numerator = abs((y2-y1)*variable1 - (x2-x1)*variable2 + x2*y1 - y2*x1)
            denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances_List.append(numerator/denominator)
    
        return distances_List.index(max(distances_List)) + 2
    
    def ClusterModel1(self):
        
        """
        clustermodel1 uses the output of kmeans() and builds a model for segmentation.
        imput : data frame
        returns: None
        """

        sum_of_Squares = self.kmeans()
        optimal_clusters = self.optimal_number_of_clusters(sum_of_Squares)
        
        kmeans = KMeans(n_clusters= optimal_clusters, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
        kmeans.fit(self.dataFrame)
        
        clusters = kmeans.fit_predict(self.dataFrame)
        self.clusters_Model1=clusters
        
        self.prediction_Model1 = kmeans.fit_predict(self.dataFrame.iloc[:,1:])
        self.silhouette_Model1 =silhouette_score(self.dataFrame, kmeans.labels_, metric='euclidean')
        
        
    def ClusterModel2(self,components_Count_Value):
        
        """
        clustermodel2 uses Principal Component Analysis and builds a model for segmentation.
        param: count of components in the data set
        returns: None
        
        """
        min_range = self.min_Range
        max_range = self.max_Range
        
        components_Count = PCA(n_components=components_Count_Value)
        principal_Components = components_Count.fit_transform(self.dataFrame)
        features = range(components_Count.n_components_)
        
        plt.bar(features, components_Count.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.show()
        
        PCA_components = pd.DataFrame(principal_Components)
        range_Count = range( min_range, max_range)
        
        inertias = []
        for k in range_Count:
            model = KMeans(n_clusters=k)
            model.fit(PCA_components.iloc[:,:2])
            inertias.append(model.inertia_)
            
        optimal_Clusters= self.optimal_number_of_clusters(inertias)
        model = KMeans(n_clusters=optimal_Clusters)
        model.fit(PCA_components.iloc[:,:2])
        clusters = model.fit_predict(PCA_components.iloc[:,:2])
        
        self.clusters_Model2=clusters
        self.silhouette_Model2=silhouette_score(PCA_components.iloc[:,:2], model.labels_, metric='euclidean')
        self.prediction_Model2 = model.predict(PCA_components.iloc[:,:2])
        
    
    def plots_Clusters(self,component1,component2,component3):
        
        """
        compares the silhouette score of both the cluster models and prints the clusters of the model with highest silhouette score
        input: cluster model
        returns: None
        """
        if(self.silhouette_Model1 < self. silhouette_Model2):
           
            print("Cluster Model 2")
            self.dataFrame["label"] = self.clusters_Model2
            
        else:
           
           print("cluster model1") 
           self.dataFrame["label"] = self.clusters_Model1
           
        fig = plt.figure(figsize=(21,10))
        plot = fig.add_subplot(111, projection='3d')
        plot.scatter(self.dataFrame[component1][self.dataFrame.label == 0], self.dataFrame[component2][self.dataFrame.label == 0], self.dataFrame[component3][self.dataFrame.label == 0], c='blue', s=60)
        plot.scatter(self.dataFrame[component1][self.dataFrame.label == 1], self.dataFrame[component2][self.dataFrame.label == 1], self.dataFrame[component3][self.dataFrame.label == 1], c='red', s=60)
        plot.scatter(self.dataFrame[component1][self.dataFrame.label == 2], self.dataFrame[component2][self.dataFrame.label == 2], self.dataFrame[component3][self.dataFrame.label == 2], c='green', s=60)
        plot.scatter(self.dataFrame[component1][self.dataFrame.label == 3], self.dataFrame[component2][self.dataFrame.label == 3], self.dataFrame[component3][self.dataFrame.label == 3], c='orange', s=60)
        plot.view_init(30, 185)
        plt.title("Clusters")
        plot.set(xlabel=component1)
        plot.set(ylabel=component2)
        plot.set(zlabel=component3)
            
            
           
    def plots_Analysis(self,component1,component2,component3):
        
        """
        prints bar plots for cluster analysis 
        input : clusters mapped to the dataframe
        returns: None
        
        """
        if(self.silhouette_Model1 > self. silhouette_Model2):
            prediction=self.prediction_Model1
        else:
            prediction=self.prediction_Model2
            
        data_frame = self.dataFrame.drop(['CustomerID'],axis=1)
        frame = pd.DataFrame(data_frame)
        frame['cluster'] = prediction
        frame.head()
        data_plot = data_frame.groupby(['cluster'], as_index=False).mean()
        sns.catplot(x='cluster', y=component1,data=data_plot, kind='bar')
        sns.catplot(x='cluster', y=component2,data=data_plot, kind='bar')
        sns.catplot(x='cluster', y=component3,data=data_plot, kind='bar')
       
         
        
        
         
        
        

    
        
        
        
        
       
        
        

    

    
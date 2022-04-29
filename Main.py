# -*- coding: utf-8 -*-


import pandas as pd
 
from Clustering import *


class Analysis:
    
    
    def mall_customers():

        DataFrame = pd.read_csv('.\Mall_Customers.csv')
        print(DataFrame.head(5))
       
        # updating columns with null values to 0
        DataFrame=DataFrame.fillna(0)
        
        # Converting categorical data to numeric
        DataFrame['Gender']=DataFrame['Gender'].replace(['Male', 'Female'],[0,1])
    
        # Specify range for clusters as  min_range and max_range
        min_range=2
        max_range=10
        
        # creating object for Clustering class
        obj1= Clustering(DataFrame,min_range,max_range)
        
        #invoking visualizations function for visualizing the bar plots of customer data
        obj1.visualizations('Age')
        obj1.visualizations('Spending Score (1-100)')
        obj1.visualizations('Annual Income (k$)')
        
        # invoking kmeans function to calculate the inertia score of clusters and append the score to a list
        inertia_Score= obj1.kmeans()
        
        # invoking print_clusters to print the cluster
        obj1.print_clusters()
    
        # invoking optimal_number_of_clusters method to calculate the optimal clusters
        Optimal_Clusters=obj1.optimal_number_of_clusters(inertia_Score)
        print("Number of optimal clusters is : "+ str(Optimal_Clusters))
        
        # invoking clustermodel1 for segmentation
        obj1.ClusterModel1()
        
        # specify the number of components for cluster model2
        components_Count=4
        
        # invoking clustermodel2 for segmentation
        obj1.ClusterModel2(components_Count)
        
        # columns to be selected for displaying the clusters
        component1="Age"
        component2="Annual Income (k$)"
        component3= "Spending Score (1-100)"
        
        # printing the clusters
        obj1.plots_Clusters(component1,component2,component3)
        
        # printing bar plots for cluster analysis
        obj1.plots_Analysis(component1,component2,component3)
        
    # invoking mall_customers function for segmentation of mall customers dataset    
    mall_customers()

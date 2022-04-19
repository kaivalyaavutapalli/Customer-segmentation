# -*- coding: utf-8 -*-




import pandas as pd
 
from Clustering import *


class Analysis:
    
    df = pd.read_csv(r'"C:\Desktop\programming\New folder1\Mall_Customers.csv"')
    
    # updating columns with null values to 0
    df=df.fillna(0)

    # Converting categorical data to numeric
    df['Gender']=df['Gender'].replace(['Male', 'Female'],[0,1])
    
    # Range of Clusters should be specified
    min_range=2
    max_range=10
    
    obj1= Clustering(df,min_range,max_range)
    
    # Specify range for clusters as  min_range and max_range
    
    obj1.kmeans()
    
    var1=obj1.calculate_wcss() 
    var2=obj1.optimal_number_of_clusters(var1)
    print("Number of optimal clusters is : "+ str(var2))
   
    obj1.ClusterModel1("Annual Income (k$)","Age")
    obj1.ClusterModel1("Age","Spending Score (1-100)")
    obj1.ClusterModel1("Annual Income (k$)","Spending Score (1-100)")
    

    

    

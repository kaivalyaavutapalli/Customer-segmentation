# -*- coding: utf-8 -*-


import pandas as pd
 
from Clustering import *

class Analysis:
    
    df = pd.read_csv(r'C:\Users\bhavi\Downloads\Mall_Customers.csv')
    
               
    print(df.head(10))
    
    # updating columns with null values to 0
    df=df.fillna(0)

    # Converting categorical data to numeric
    df['Gender']=df['Gender'].replace(['Male', 'Female'],[0,1])
    
    obj1= Clustering(df)
    
    # Specify range for clusters as  min_range and max_range
    
    min_range=2
    max_range=10
    obj1.clusters(min_range,max_range)
    
    var1=obj1.calculate_wcss() 
    var2=obj1.optimal_number_of_clusters(var1)
    print("Number of optimal clusters is : "+ str(var2))
   
    obj1.kmeans("Annual Income (k$)","Age")
    obj1.kmeans("Age","Spending Score (1-100)")
    obj1.kmeans("Annual Income (k$)","Spending Score (1-100)")
    

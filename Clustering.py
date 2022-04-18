# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt


class Clustering:
        
    def __init__(self,df):
        self.df=df
    
    def clusters(self,value1,value2):
        MinimumRange = value1
        MaximumRange = value2

        InertiaScore= []
        RangeList = range(MinimumRange, MaximumRange)

        for k in  RangeList:
            kmeans = KMeans(n_clusters = k, init='k-means++', random_state= 0)
            kmeans.fit(self.df) 
            Score = kmeans.inertia_
            InertiaScore.append(Score)

        plt.figure(1,figsize = (10 ,8))
        plt.plot(np.arange(MinimumRange , MaximumRange) , InertiaScore , '-' ,marker='o', alpha = 0.5)
        plt.xlabel('Number of Clusters') , plt.ylabel('Inertia score')
        plt.show()
        
    
    def calculate_wcss(self):
        min_range = 2
        max_range = 10
        wcss = []
        for n in range(min_range, max_range):
            kmeans = KMeans(n_clusters=n,random_state=0)
            kmeans.fit(self.df)
            wcss.append(kmeans.inertia_)
        
        return wcss


    
    def optimal_number_of_clusters(self,var1):
       
        min_range = 2
        max_range = 10
        x1, y1 = min_range, var1[0]
        x2, y2 = max_range, var1[len(var1)-1]

        distances = []
        for i in range(len(var1)):
            x0 = i+2
            y0 = var1[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
    
        return distances.index(max(distances)) + 2
    
    def kmeans(self,label1,label2):

        sum_of_squares = self.calculate_wcss()

        n = self.optimal_number_of_clusters(sum_of_squares)
    
        kmeans = KMeans(n_clusters=n, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
        kmeans.fit_predict(self.df)
        
        clusters = kmeans.fit_predict(self.df)
        self.df["label"] = clusters
     
        fig = plt.figure(figsize=(21,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter( self.df[label1][self.df.label == 0], self.df[label2][self.df.label == 0], c='blue', s=60)

        ax.scatter( self.df[label1][self.df.label == 1], self.df[label2][self.df.label == 1], c='red', s=60)
        ax.scatter( self.df[label1][self.df.label == 2], self.df[label2][self.df.label == 2], c='green', s=60)
        ax.scatter( self.df[label1][self.df.label == 3], self.df[label2][self.df.label == 3], c='orange', s=60)

        ax.view_init(30, 185)
        plt.show()

        

    

    
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.decomposition import PCA

class Clustering:
        
    def __init__(self,df,Value1,Value2):
        self.df=df
        self.value1=Value1
        self.value2=Value2
    
    def kmeans(self):
        MinimumRange = self.value1
        MaximumRange = self.value2

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
        wcss = []
        for n in range(self.value1,self.value2):
            kmeans = KMeans(n_clusters=n,init='k-means++',random_state=0)
            kmeans.fit(self.df)
            wcss.append(kmeans.inertia_)
        return wcss

    def optimal_number_of_clusters(self,var1):
       
        min_range = self.value1
        max_range = self.value2
        x1, y1 = min_range, var1[0]
        x2, y2 = max_range, var1[len(var1)-1]

        distance = []
        for i in range(len(var1)):
            x = i+2
            y = var1[i]
            numerator = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
            denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distance.append(numerator/denominator)
    
        return distance.index(max(distance)) + 2
    
    def ClusterModel1(self,label1,label2):

        sum_of_squares = self.calculate_wcss()

        n = self.optimal_number_of_clusters(sum_of_squares)
    
        kmeans = KMeans(n_clusters=n, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
        kmeans.fit_predict(self.df)
        
        clusters = kmeans.fit_predict(self.df)
        self.df["label"] = clusters
     
        figure = plt.figure(figsize=(20,15))
        axis = figure.add_subplot(111, projection='3d')
        axis.scatter( self.df[label1][self.df.label == 0], self.df[label2][self.df.label == 0], c='blue', s=50)

        axis.scatter( self.df[label1][self.df.label == 1], self.df[label2][self.df.label == 1], c='red', s=50)
        axis.scatter( self.df[label1][self.df.label == 2], self.df[label2][self.df.label == 2], c='green', s=50)
        axis.scatter( self.df[label1][self.df.label == 3], self.df[label2][self.df.label == 3], c='orange', s=50)

        axis.view_init(35, 140)
        plt.show()
        
    def ClusterModel2(self,label1,label2):
        sum_of_squares = self.calculate_wcss()
        n = self.optimal_number_of_clusters(sum_of_squares)
        pca = PCA(n_components=n)
        principalComponents = pca.fit_transform(self.df)

        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)

        PCA_components = pd.DataFrame(principalComponents)
        ks = range(self.value1, self.value2)
        inertias = []

        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(PCA_components.iloc[:,:2])
            inertias.append(model.inertia_)

        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.show()
        model = KMeans(n_clusters=n)
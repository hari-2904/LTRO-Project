"""
@Project: LTRO Project - Clustering
@author: Harihareshwar Kumaravel
"""
# ///LIBRAIRIES///

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import os

# ///USER INPUTS///
date = '2022-07-19' #Date on which the clustering need to be done
m = 0 # Input type of waste - 0,1,2,3 for BX,Verre,OM,Carton respectively

# ///OTHER VARIABLES///
dist_mat = "input/Distances.xlsx"
dist_mat = os.path.abspath(dist_mat)
dist = pd.read_excel(dist_mat, sheet_name=1)

# ///FUNCTIONS///

def clean_column(value):  #To clean the column to get int datatype
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value

def Selectpoints(date):   #Selects datapoints from the predicted data for a single date
    date = date
    pred_loc = "input/Predictions.xlsx"
    pred_loc = os.path.abspath(pred_loc)
    df = pd.read_excel(pred_loc, sheet_name = m )
    df['date']=pd.to_datetime(df['date'])
    df['n_point'] = df['n_point'].astype(str)
    df['day'] = [i.day for i in df['date']]
    df['month'] = [i.month for i in df['date']]
    df['year'] = [i.year for i in df['date']]
    df['week'] = [i.week for i in df['date']]
    df = df.loc[df['date'] == date]
    df[['points', 'bin_no']] = df['n_point'].str.extract(r'(\d+)-?(\d*)')
    df['points'] = df['points'].fillna(value=df['n_point'])
    df = df.loc[:,('date','n_point','prediction','points','bin_no')]
    df = df.query('prediction >= 0.7')
    df= df.reset_index()
    df['points'] = df['points'].apply(clean_column)
    df['dist'] = 0
    for i in range (len(df)):
        for j in range(len(dist)):
            if df['points'][i] == dist['n_point'][j]:
                df.loc[i, 'dist'] = dist.loc[j,'dist']
    return df

def OptimumClusters(df):
    X = df[['dist','prediction']]

    km = KMeans()
    visualizer = KElbowVisualizer(km, k=(2,8)) #Number of available trucks is 7
    
    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure
    return visualizer.elbow_value_

def Clustering(df):  #Clusters the data based on prediction and distance from depot using k-means algorithm 
    X = df[['dist','prediction']]
    km = KMeans(n_clusters = OptimumClusters(df))
    y_predicted = km.fit_predict(X)

    df['cluster'] = y_predicted
    b = df['cluster'].unique()
    a =[]
    for i in range(len(b)):
        c = df[df.cluster == i]
        a.append(c['points'].unique())
    data = []
    for arr in a:
        data.append(arr.tolist())
    data = [[elem-1 for elem in sublist] for sublist in data]
    return data

df = Selectpoints(date)

print(Clustering(df))
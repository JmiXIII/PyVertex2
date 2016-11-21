# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:24:47 2016

@author: user11
"""

import os
import glob
import numpy as np
import pandas as pd
import xlwings as xw
import seaborn as sn
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from bokeh.charts import Histogram, output_file, show
from plotly.tools import FigureFactory as FF


plotly.offline.init_notebook_mode()

# %% extraction des données
path = os.getcwd()
path = "G:/01-SUIVI PROCESSUS/GMM/Micro-Vu Vertex/Schrader/43418-820"
fcsv = glob.glob(path + "/CSV Data/*.txt")
datas = []
for file in fcsv:
    try:
        data = pd.read_csv(file, header=None, sep=None, index_col=False)
        b = {x: 'c'+str(x) for x in data.columns}
        data.rename(columns=b, inplace=True)
        datas.append(data)
    except:
        print("error" + file)
datas = pd.concat(datas)
datas = datas.loc[:,:'c1']
items = list(datas['c0'].unique())
gdatas = [datas.groupby('c0').get_group(item) for item in items]
pdatas = gdatas[3]['c1'].dropna()
n = len(pdatas)


# %% Excel export
#wb = xw.Book()
#for i in range(len(items)):
#    xw.Range((1, i+1), index=False).value = gdatas[i].loc[:, 1]
#    xw.Range((1,i+1)).value = gdatas[i][0].unique()
# %% seaborn graph

plt.figure(1)
sn.distplot(pdatas)
plt.figure(2)
sn.tsplot(pdatas)
# %% plotly graph

hist_data2 = [pdatas.tolist()]
hist_data = [[1.3,1,2,3,5,4,1,2,6,3,5,2]]
group_labels = ['distplot']
fig = FF.create_distplot(hist_data2, group_labels,bin_size=0.005)
py.offline.plot(fig, filename='Simple Distplot', validate=False)


trace = go.Scatter(
    x = np.linspace(0,n,n+1),
    y = pdatas
)

data = [trace]

# Plot offline
py.offline.plot(data, filename='basic-line')

# %% Bokeh graph

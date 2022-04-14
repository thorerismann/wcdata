#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import os
import sys
import datetime as dt
import csv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib.dates as mdates


# In[ ]:





# In[73]:


past = pd.read_csv("stats-downscaling-2.csv")
future = pd.read_csv("stats-downscaling-1.csv")


# In[74]:


future.columns


# In[79]:


for i in future.index:
    distances = []
    mydict = {}
    for j in past.index:
        xdist = (past.at[j,"dZ500/dx"]-future.at[i,"dZ500/dx"])*(past.at[j,"dZ500/dx"]-future.at[i,"dZ500/dx"])
        ydist = (past.at[j,"dZ500/dy"]-future.at[i,"dZ500/dy"])*(past.at[j,"dZ500/dy"]-future.at[i,"dZ500/dy"])
        zdist = (past.at[j,"Z500"]-future.at[i,"z500"])*(past.at[j,"Z500"]-future.at[i,"z500"])
        distance = ydist + xdist + zdist
        distance = (math.sqrt(distance))
        distance = round(distance,4)
        distances.append(distance)
        mydict[distance] = j
    mymin = min(distances)
    mymin_index = mydict[mymin]
    mymin_rain = past.at[mymin_index,"PREC [mm]"]
    future.at[i,"MIN_DISTANCE"] = mymin
    future.at[i,"ROW_NUMBER"] = mymin_index
        


# In[80]:


future.head()


# In[78]:


past.head()


# In[ ]:





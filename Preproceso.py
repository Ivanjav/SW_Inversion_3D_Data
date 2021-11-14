#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[1]:
#from IPython.display import display
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data
# 
# Read the file **sou_rec_pos.pkl** with pandas as DataFrame and the file **seismic_data_source_1000.nc** with xarray as DataSet

# In[2]:


sou_rec_pos = pd.read_pickle("sou_rec_pos.pkl")
seismic_data= xr.open_dataset("seismic_data_source_1000.nc")
seismic_data


# ## Create new dimensions
# Compute **Azimuth** and **Offset** from sources and receivers positions.

# In[3]:


pos_x=np.array(sou_rec_pos['SourceX']-sou_rec_pos['GroupX'], dtype='f')
pos_y=np.array(sou_rec_pos['SourceY']-sou_rec_pos['GroupY'], dtype='f')
azimuth=np.arctan2(-pos_x, pos_y) * 180 / np.pi+180
offset = np.linalg.norm([pos_x,pos_y], axis=0)  
sou_rec_pos['Azimuth']=azimuth
sou_rec_pos['Offset']=offset
sou_rec_pos


# Histogram of the **Azimuth** for the source 1000.

# In[4]:
source=1000
shot_pos=sou_rec_pos[sou_rec_pos['EnergySourcePoint']==source]
shot_pos[['Azimuth']].plot.hist()
#plt.show()


# ## Highligth position by azimuth bin
# Show receivers fro the source 1000 with azimuth from 90 to 110.

# In[5]:

az_1=90 
az_2=110                  #Limit values of azimuth
shot_azimuth_pos=shot_pos[(shot_pos['Azimuth']>az_1)&(shot_pos['Azimuth']<az_2)]
sns.scatterplot(data=sou_rec_pos, x="GroupX", y="GroupY")
sns.scatterplot(data=shot_azimuth_pos, x="GroupX", y="GroupY")
sns.scatterplot(data=shot_azimuth_pos, x="SourceX", y="SourceY")
#plt.show()


# ## Selecting traces for the azimuth bin
# Create the azimuth coordinate for the DataSet **seismic_data**

# In[6]:


seismic_data["azimuth"] = ("azimuth",  shot_pos['Azimuth'])
seismic_data["offset"] = ("azimuth",  shot_pos['Offset'])
seismic_data["data"] = (("azimuth", "time"), np.array(seismic_data.data))
seismic_data


# Select DataArray **data_azimuth** from the DataSet **seimisc_data** in the azimuth bin.

# In[7]:


data = seismic_data.where((seismic_data.azimuth>az_1)&(seismic_data.azimuth<az_2), drop=True)  #Select traces for the azimuth bin
data_azimuth = data.data.sortby('azimuth') #Sort DataArray by azimuth
data_azimuth


# Plot the shot gather foe the azimuth bin sort by azimuth.

# In[8]:
plt.figure()
data_azimuth.plot.imshow('azimuth','time',cmap='gray', vmin=-1000, vmax=1000, origin='upper')

# ## Sort the data by offset
# Select DataArray **data_offset** from the DataSet **data** in the azimuth bin.

# In[9]:


data=data.swap_dims({"azimuth": "offset"})
data_offset = data.data.sortby('offset') #Sort DataArray by offset
data_offset


# In[10]:

plt.figure()
data_offset.plot.imshow('offset','time',cmap='gray', vmin=-1000, vmax=1000, origin='upper')


# Plot the shot gather for the azimuth bin sort by offset.

# In[29]:


import masw_functions as mf
vmin=100.0 
vmax=4001.0
fmin=1.0 
fmax=8.0
pstep=5e-5
Nf=4001
v=np.arange(vmin,vmax)
D=mf.DispersionSpectrum(data_offset,v,fmin,fmax,Nf,pstep)
plt.rcParams['figure.figsize'] = [12, 6]
D.plot.imshow(D.dims[1],D.dims[0],cmap='viridis', origin='lower')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase velocity (m/s)")


# In[30]:
import masw_functions as mf
Xobs=mf.AutomaticPicking(D,threshold=0.8,num=30,fig=True)


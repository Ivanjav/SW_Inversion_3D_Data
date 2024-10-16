from segysak.segy import segy_loader
import pathlib
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import pickle
from matplotlib import rc
import masw_functions as mf
from matplotlib.ticker import FuncFormatter
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



# specify the example file and check we have the example data
segy_file = pathlib.Path("/home/ivan/data/NISCOTA/Niscota_4B.sgy")
print("SEG-Y exists:", segy_file.exists())

fileheader="header_new.pkl"
header = pd.read_pickle(fileheader)
print("header file loaded")

# Import topography interpolator
filetopo = "inter_topo.pkl"
with open(filetopo, 'rb') as f:
     rbf = pickle.load(f)

# New header dimensions
pos_x=np.array(header['ReceiverX']-header['SourceX_new'], dtype='f')
pos_y=np.array(header['ReceiverY']-header['SourceY_new'], dtype='f')
azimuth = np.degrees(np.arctan2(pos_x, pos_y))
ida = np.where(azimuth < 0)
azimuth[ida] = azimuth[ida] + 360
backazimuth = np.mod(azimuth + 180, 360)
offset = np.linalg.norm([pos_x,pos_y], axis=0)  
header['Azimuth']=azimuth
header['Backazimuth']=backazimuth
header['Offset']=offset
print("New header dimensions created (Azimuth, Backazimuth, Offset)")

# Parameters for dispersion image computing
vmin, vmax = 600,4000
fmin, fmax = 1,5
minof=500           # Limit minimum offset
dbaz=30   # backazimuth bin range
p=np.linspace(0, 1/vmin, 101)
theta = np.linspace(0, 360, 201)
dimage = []
error = []
header_dispersion = pd.DataFrame(columns=['Shot','Backazimuth', 'Sx', 'Sy','MPx','MPy','MPazimuth'])
# Load shot 

for shot in range(1,3646,1):
    shot_gather = segy_loader(
        segy_file,
        head_df=header[header['EnergySourcePoint'] == shot].copy()
    )
    # Change shot time dimension
    shot_gather["time"] = ("time",  np.array(shot_gather.twt*1e-3, dtype='f'))
    shot_gather["data"] = (("cdp", "time"), np.array(shot_gather.data))

    # Select header shot gather
    shot_pos=header[header['EnergySourcePoint']==shot]

    # Add geodesic dimension in the shot gather
    x_pts=shot_pos[['SourceX_new','ReceiverX']].to_numpy().astype('f')
    y_pts=shot_pos[['SourceY_new','ReceiverY']].to_numpy().astype('f')
    xt = np.transpose(np.linspace(x_pts[:,0],x_pts[:,1],100))
    yt = np.transpose(np.linspace(y_pts[:,0],y_pts[:,1],100))
    zt = rbf(xt,yt)
    geodesic = np.sum(np.linalg.norm([np.diff(xt),np.diff(yt),np.diff(zt)], axis=0), axis=1)
    shot_pos['Geodesic']=geodesic

    # New dimensions for shot data
    shot_gather["backazimuth"] = ("backazimuth",  shot_pos['Backazimuth'])
    shot_gather["azimuth"] = ("backazimuth",  shot_pos['Azimuth'])
    shot_gather["geodesic"] = ("backazimuth",  shot_pos['Geodesic'])
    shot_gather["offset"] = ("backazimuth",  shot_pos['Offset'])
    shot_gather["recx"] = ("backazimuth",  shot_pos['ReceiverX'])
    shot_gather["recy"] = ("backazimuth",  shot_pos['ReceiverY'])
    shot_gather["data"] = (("backazimuth", "time"), np.array(shot_gather.data))


    for bazi in range(0,360,30):
        print('Analyzing Shot:',shot,'Backazimuth:',bazi)
        try :
            bazf=bazi+dbaz      # Limit values of azimuth
            shot_backazimuth_pos=shot_pos[(shot_pos['Backazimuth']>bazi)&(shot_pos['Backazimuth']<bazf)&(shot_pos['Offset']>minof)]

            if shot_backazimuth_pos.shape[0] > 60:
                # Select data for azimuth bin
                data  = shot_gather.where((shot_gather.backazimuth>=bazi)&(shot_gather.backazimuth<bazf)&(shot_gather.offset>minof), drop=True) 
                # Preprocess data (Velocity filtering)
                [Time,Off]=np.meshgrid(data.time,data.offset)
                dt= shot_gather.sample_rate*1e-3
                Vel = Off/(Time+dt)
                alpha=5e-3
                data.data.values = data.data.values*(1+np.tanh(alpha*(Vel-vmin)))*(1+np.tanh(alpha*(vmax-Vel)))
                data_geodesic = data.swap_dims({"backazimuth": "geodesic"}).data.sortby('geodesic') #Sort DataArray by offset

                ## Beamforming computation
                rec_x = data.geodesic*np.sin(data.azimuth* 2 * np.pi / 360)
                rec_y = data.geodesic*np.cos(data.azimuth* 2 * np.pi / 360)

                [m_LRT, freq, Pw]=mf.adj_beam_LRT(data.data.T.values, dt, rec_x, rec_y, p, theta, fmin, fmax)

                # Beamforming masking
                alpha_theta=2
                P, Theta = np.meshgrid(p[::-1], np.deg2rad(theta),  indexing='ij')
                Mask_Theta= (1+np.tanh(alpha_theta*(Theta-np.deg2rad(bazi))))*(1+np.tanh(alpha_theta*(np.deg2rad(bazf)-Theta)))
                Mask_Theta_expanded = Mask_Theta[:, :, np.newaxis]
                Pwm = Pw*Mask_Theta_expanded

                ## Dsipersion image computation
                sumP = np.max(Pwm, axis=1)  # Summing along the first axis
                sumPn = (sumP - np.min(sumP, axis=0)) / (np.max(sumP, axis=0) - np.min(sumP, axis=0))
                dimage.append(sumPn.astype(np.float32))

                Sx = shot_backazimuth_pos['SourceX_new'].iloc[0]
                Sy = shot_backazimuth_pos['SourceY_new'].iloc[0]
                MPx = np.mean(data.recx)
                MPy = np.mean(data.recy)
                MPaz = np.degrees(np.arctan2(MPx-Sx, MPy-Sy))
                MPaz = MPaz + 360 if MPaz < 0 else MPaz
                new_row = {'Shot': shot,'Backazimuth': bazi, 'Sx': Sx, 'Sy': Sy, 'MPx': MPx, 'MPy': MPy, 'MPazimuth': MPaz}
                header_dispersion = pd.concat([header_dispersion, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"An error occurred: {e}")
            error.append([shot,bazi])

if 0:
    with open('dispersion_images.pkl', 'wb') as file:
        pickle.dump(dimage, file)

    with open('dispersion_header.pkl', 'wb') as file:
        pickle.dump(header_dispersion, file)
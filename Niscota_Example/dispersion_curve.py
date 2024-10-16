import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import masw_functions as mf
import pickle
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Load dispersin images and header
fileheader="dispersion_header.pkl"
header = pd.read_pickle(fileheader)
filedimage="dispersion_images.pkl"
dimage = pd.read_pickle(filedimage)

# Header editing
header['MPx'] = header['MPx'] .apply(lambda x: float(x.values))
header['MPy'] = header['MPy'] .apply(lambda x: float(x.values))
header['MPazimuth'] = header['MPazimuth'] .apply(lambda x: float(x.values))
header['MPxn'] = header['MPx'] - header['MPx'].min()
header['MPyn'] = header['MPy'] - header['MPy'].min()

##
vmin, vmax = 600,4000
fmin, fmax = 1,5
freq = np.linspace(fmin,fmax,80)
p=np.linspace(0, 1/vmin, 101)
phavel=[]
mderror=[]
midpoint=[]

## Select bin
dmx = 500.
dmy = 500.
for mx in np.arange(header['MPxn'].min()+dmx,header['MPxn'].max()-dmx,dmx) :
    for my in np.arange(header['MPyn'].min()+dmy,header['MPyn'].max()-dmy,dmy) :  
        try :  
            print(f'mx: {mx}    my:{my}')
            binx = (header['MPxn']>mx) & (header['MPxn']<mx+dmx)
            biny = (header['MPyn']>my) & (header['MPyn']<my+dmy)
            headerbin = header[binx & biny]
            idbin = header[binx & biny].index

            ## Stacking an smoothing of Dispersion Image
            dimageb = np.mean(np.stack([dimage[i] for i in idbin], axis=0), axis=0)
            dimagebi = gaussian_filter(dimageb**2, sigma=10.0)
            dimagebin = dimagebi/np.max(dimagebi,axis=0, keepdims=True)

            ## # Dispersion curve extraction
            pr = p[np.argmax(dimagebin, axis=0)]
            freq = np.linspace(fmin,fmax,dimagebin.shape[1])
            phavel.append(1/pr)
            midpoint.append((mx+dmx/2,my+dmy/2))
        except Exception as e:
            print(f"An error occurred: {e}")
            mderror.append([mx,my])


plt.figure()
for id in range(len(phavel)):  #  range(len(phavel)):
    if phavel[id][-1] < 1400 and np.all(phavel[id]>600) :
        plt.plot(phavel[id])
plt.show()

## inversion
Nh = 10
rps=2.5
alpha=100
n_iter=10
dh=10.0
invmodel = []
errorinv = []
for id in range(len(phavel)):  #  range(len(phavel)):
    if phavel[id][-1] < 1400 and np.all(phavel[id]>600) :
            mx, my = midpoint[id]
            print(f'mx: {mx}    my:{my}')
            Xobs = mf.decreasing(phavel[id], freq)
            try:
                (vsi,hi,zi)=mf.InitialModel(Xobs,Nh)
                rhoi=2*np.ones(np.size(vsi))
                hi=np.round(hi/dh)*dh
                (vsf,vpf,error)=mf.sw_inversion(Xobs[:,1],Xobs[:,0],vsi,rhoi,hi,rps,n_iter,alpha,dh)
                invmodel.append([vsf, hi, mx, my])
            except Exception as e:
                errorinv.append(id)
                print(f"An error occurred: {e}")

if 1:
    with open('inversion_result.pkl', 'wb') as file:
        pickle.dump(invmodel, file)
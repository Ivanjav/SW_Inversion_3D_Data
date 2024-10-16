import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import masw_functions as mf
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


fileheader="dispersion_header.pkl"
header = pd.read_pickle(fileheader)

filedimage="dispersion_images.pkl"
dimage = pd.read_pickle(filedimage)


##
vmin, vmax = 600,4000
fmin, fmax = 1,5
p=np.linspace(0, 1/vmin, 101)


## Scatter Midpoints
plt.figure()
plt.scatter(header['MPx'],header['MPy'])
plt.show()

plt.figure()
plt.scatter(header['Sx'],header['Sy'])
plt.show()



import time
plt.figure()
for idim in range(0,len(dimage),len(dimage)):
   
    plt.imshow(dimage[idim], aspect='auto', cmap=mf.parula_cmap, origin='lower', extent=[fmin, fmax, p[0], p[-1]])
    plt.title('Dispersion Image $D(p,f)$', fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Slowness (s/m)', fontsize=16)
    plt.xticks(fontsize=14) 
    plt.yticks(fontsize=14) 
    plt.show()
    time.sleep(1.0)
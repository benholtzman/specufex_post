# plot a simple spistogram, just to check...

import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib
import csv
from matplotlib import cm
#not sure what this does-- maybe closes any h5s?
import tables
tables.file._open_files.close_all()

# our functions:
sys.path.append('./functions/')
from importlib import reload
import spistogram_mat_f as spisto
import timeFractions_f as timefrac
reload(spisto)

# read in the paths:
# this will be replaced with the path to the config file.
sys.path.append('../../specufex_preprocessing/functions/')
from setParams import setParams
key = sys.argv[1]
print(key)

# pick the operating system, for pandas.to_csv (this will be in the config file, if necessary)
OSflag = 'linux'
#OSflag = 'mac'

# -------------
pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)

path_clustercat = pathProj + f'cat_Clusters_{key}.csv'
cat0c = pd.read_csv(path_clustercat)
print(cat0c.columns)

#cat0c['datetime'].to_datetime
# PROBABLY better to just create this from each column... year month day hour min sec, put into datetime objects,
# rather than going from the date string created by pandas "to_csv()".
cat0c['DateObj'] = pd.to_datetime(cat0c['datetime'],format= "%Y-%m-%d %H:%M:%S")
#cat = cat.set_index('Date')
# ==========================================
# calculate time fractions:
reload(timefrac)
yearFrac,hourFrac = timefrac.calc_YearFloat_HourFloat(cat0c)
#print(yearFrac[0:5],hourFrac[2000:2005])

cat0c['YearFrac'] = yearFrac
cat0c['HourFrac'] = hourFrac

# if that's what we're going to use instead of YearFrac
cat0c['time_4plot']=yearFrac

cat0c = cat0c.sort_values('YearFrac')
cat0c = cat0c.reset_index()

#index = cat0c.index
yr_frac_out = cat0c.YearFrac

plt.subplot(2,1,1)
plt.plot(yr_frac_out)
plt.show()

# =================================================
#  Histogram of event times
histAll, bin_edgesAll = np.histogram(yearFrac, bins = 300, density=False)
time_histAll = bin_edgesAll[0:-1]
#print(time_hr0[0:5])
# plt.plot(time_histAll, histAll)

# ==========================================================
# Make the spisto matrixes:

NC = 5
n_bins = 212
cluster_list = list(range(1,NC+1))
timeEvts_mat, xgrid_spisto, ygrid_spisto = spisto.makeSpistoMatrix(cat0c,NC,n_bins,cluster_list)

# ========================================================================
## PLOT THE SPISTOGRAM !
#fig = plt.figure(figsize=(10,9))

legendfontsize = 14
tickfontsize = 12

# refine the size ! make thinner !
fig, (ax0, ax1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[1,5]},figsize=(10,9))


ax0.tick_params(axis='both', which='major', labelsize=tickfontsize)
ax0.plot(time_histAll, histAll,'k')
ax0.set_xlim([time_histAll[0],time_histAll[-1]])

# ============================
ax1.tick_params(axis='both', which='major', labelsize=tickfontsize)
# gotta fix the plot labels so it shows plt

plt.pcolormesh(xgrid_spisto,ygrid_spisto,timeEvts_mat,cmap='YlOrRd')
#plt.pcolormesh(H,cmap='YlOrRd')
ax1.set_ylabel('Cluster', fontsize=legendfontsize)
bar = plt.colorbar(orientation='horizontal',fraction=0.05, pad=0.1)#, anchor=(0.5,1))
bar.set_label('Number of events per bin', fontsize=legendfontsize)

ax1.set_xlabel('time [units here]', fontsize=legendfontsize)

plt.axis('tight')
plt.show()

fig.savefig(pathProj+'spisto.png')

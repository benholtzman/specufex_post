# extract the fingerprints from the h5 SpecUFEx file,
# run k-means on them and merge with a catalog
import sys

import numpy as np
import h5py
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
# read in the paths:

import pandas as pd

sys.path.append('functions/')
from setParams import setParams
#from generators import gen_wf_from_folder

import tables
tables.file._open_files.close_all()


key = sys.argv[1]
print(key)

# pick the operating system, for pandas.to_csv
OSflag = 'linux'
#OSflag = 'mac'

# -------------
pathProj, pathCat, pathWF, network, station, channel, channel_ID, filetype, cat_columns = setParams(key)

SpecUFEx_H5_name = f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = pathProj + '/H5files/' + SpecUFEx_H5_name

#wf_cat_out = pathProj + 'wf_cat_out.csv'
sgram_catPath = pathProj + f'sgram_cat_out_{key}.csv' #sgram_cat_out_GeysersNW.csv
cat = pd.read_csv(sgram_catPath)
print(cat)


#### After linearizing FPs into data array X, you can then do Kmeans on X
## Optimal cluster is chosen by mean Silh scores, but euclidean distances are saved also


def linearizeFP(SpecUFEx_H5_path,cat00):

    X = []
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        print("Keys: %s" % MLout.keys())
        key_list = list(MLout.keys())
        print(key_list)

        #for evID in cat00.event_ID:
            #print(evID)
            #fp = MLout['SpecUFEX_output/fprints'].get(str(evID))[:]
            #linFP = fp.reshape(1,len(fp)**2)[:][0]
            #X.append(linFP)

    X = np.array(X)

    return X


X = linearizeFP(SpecUFEx_H5_path,cat)
print(np.shape(X))

# ==================
sys.exit()
# ==================

"""
##def calcSilhScore(range_n_clusters,numPCA,Xtype='fingerprints'):
    Parameters
    ----------

    range_n_clusters : range type - 2 : Kmax clusters
    numPCA : number of principal components to perform clustering on (if not on FPs)
    Xtype : cluster directly on fingerprints or components of PCA. The default is 'fingerprints'.


    Returns
    -------
    Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters.

"""
maxSilScore = 0

sse = []
avgSils = []
centers = []

for n_clusters in range_n_clusters:

    print(f"kmeans on {n_clusters} clusters...")

    kmeans = KMeans(n_clusters=n_clusters,
                       max_iter = 500,
                       init='k-means++', #how to choose init. centroid
                       n_init=10, #number of Kmeans runs
                       random_state=0) #set rand state

    #get cluster labels
    cluster_labels_0 = kmeans.fit_predict(X)

    #increment labels by one to match John's old kmeans code
    cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]

    #get euclid dist to centroid for each point
    sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
    sum_sqr_dist = sqr_dist.sum(axis=1)
    euc_dist = np.sqrt(sum_sqr_dist)

    #save centroids
    centers.append(kmeans.cluster_centers_ )

    #kmeans loss function
    sse.append(kmeans.inertia_)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    #%  Silhouette avg
    avgSil = np.mean(sample_silhouette_values)
    avgSils.append(avgSil)
    if avgSil > maxSilScore:
        Kopt = n_clusters
        maxSilScore = avgSil
        cluster_labels_best = cluster_labels
        euc_dist_best = euc_dist
        ss_best       = sample_silhouette_values


print(f"Best cluster: {Kopt}")


## return Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best





#return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best

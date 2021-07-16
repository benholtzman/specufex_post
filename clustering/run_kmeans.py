# extract the fingerprints from the h5 SpecUFEx file,
# run k-means on them and merge with a catalog
import sys

import numpy as np
import h5py
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import pandas as pd

sys.path.append('./functions/')
from importlib import reload
import functions_clustering as funclust
reload(funclust)

import tables
tables.file._open_files.close_all()

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

SpecUFEx_H5_name = f'SpecUFEx_{key}.hdf5'
SpecUFEx_H5_path = pathProj + '/H5files/' + SpecUFEx_H5_name

with h5py.File(SpecUFEx_H5_path,'r') as MLout:
    #key_list = list(MLout.keys())
    #print(key_list)
    print("Keys: %s" % MLout.keys())
    #print("fingerprints Keys: %s" % MLout['fingerprints'].keys())
    #print("fingerprints Keys: %s" % MLout['fingerprints'].keys())
    #print("Spectrogram Keys: %s" % MLout['spectrograms'].keys())
    fprintIDkey_list = list(MLout['fingerprints'].keys())
    print(MLout['fingerprints'].get(fprintIDkey_list[12])[:])


#wf_cat_out = pathProj + 'wf_cat_out.csv'
sgram_catPath = pathProj + f'sgram_cat_out_{key}.csv' #sgram_cat_out_GeysersNW.csv
cat0 = pd.read_csv(sgram_catPath)
print(cat0[0:5])
print(len(cat0))



#### After linearizing FPs into data array X, you can then do Kmeans on X
## Optimal cluster is chosen by mean Silh scores, but euclidean distances are saved also

# move this to ./functions/functions_clustering.py
def linearizeFP(SpecUFEx_H5_path,cat0):

    X = []  # consider changing to numpy method?
    count = 0
    with h5py.File(SpecUFEx_H5_path,'r') as MLout:
        for evID in cat0.event_ID:
            if count%500 == 0:
                print(count,evID)

            fp = MLout['fingerprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)  # consider changing to numpy method?
            count+=1

    X = np.array(X)

    return X


X = linearizeFP(SpecUFEx_H5_path,cat0)
print(np.shape(X))

# ====================================================================
# Do the PCA (in case of clustering on PCA but also for visualization)
# ====================================================================

numPCA = 3
sklearn_pca, PCA_df, Y_pca = funclust.PCAonFP(X,cat0,numPCA=3,stand=True)
# why return sklearn_pca ?

# ====================================================================
# calculate silhouette scores
# adapt to use the function in functions_clustering
# ====================================================================
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

# maxSilScore = 0
#
# sse = []
# avgSils = []
# centers = []
#
# for n_clusters in range_n_clusters:
#
#     # this can be a function:
#     print(f"kmeans on {n_clusters} clusters...")
#
#     kmeans = KMeans(n_clusters=n_clusters,
#                        max_iter = 500,
#                        init='k-means++', #how to choose init. centroid
#                        n_init=10, #number of Kmeans runs
#                        random_state=0) #set rand state
#
#     #get cluster labels
#     cluster_labels_0 = kmeans.fit_predict(X)
#
#     #increment labels by one to match John's old kmeans code
#     cluster_labels = [int(ccl)+1 for ccl in cluster_labels_0]
#
#     #get euclid dist to centroid for each point
#     sqr_dist = kmeans.transform(X)**2 #transform X to cluster-distance space.
#     sum_sqr_dist = sqr_dist.sum(axis=1)
#     euc_dist = np.sqrt(sum_sqr_dist)
#
#     #save centroids
#     centers.append(kmeans.cluster_centers_ )
#
#     #kmeans loss function
#     sse.append(kmeans.inertia_)
#
#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)
#
#     #%  Silhouette avg
#     avgSil = np.mean(sample_silhouette_values)
#     avgSils.append(avgSil)
#     if avgSil > maxSilScore:
#         Kopt = n_clusters
#         maxSilScore = avgSil
#         cluster_labels_best = cluster_labels
#         euc_dist_best = euc_dist
#         ss_best       = sample_silhouette_values
#
#
# print(f"Best cluster: {Kopt}")

runSilhouette = 'False'
if runSilhouette == 'True':
    Kmax = 7
    range_n_clusters = range(2,Kmax)


    # NOTE that i modified this so that you pass in X, be it fingerprints or PCA-- do that outside the function.
    pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best = funclust.calcSilhScore(X,cat0,range_n_clusters,distMeasure = "SilhScore",stand=True)


# ================================================================
# Rerun k-means at a single K value and write that to a catalog:

Ksave = 5  # could equal Kopt or not.
print(f"Rerunning kmeans on {Ksave} clusters, to save to catalog")

sse = []
centers = []
kmeans = KMeans(n_clusters=Ksave,
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

#
#
print(cluster_labels[0:20])

column_title = 'ClusterLbls_NC'+str(Ksave)
df_tmp = pd.DataFrame({column_title:cluster_labels,'event_ID':cat0['event_ID']}, index = cat0.index)
cat0c = pd.merge(cat0,df_tmp)


path_clustercat = pathProj + f'cat_Clusters_{key}.csv'
# not sure this is necessary, but keep for now: 
print('formatting CSV catalog for ',OSflag)
if OSflag=='linux':
    cat0c.to_csv(path_clustercat,line_terminator='\n')
elif OSflag=='mac':
    cat0c.to_csv(path_clustercat)

# ==================
sys.exit()
# ==================

## return Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best
# return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best

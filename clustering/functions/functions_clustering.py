

import h5py
import pandas as pd
import numpy as np
#from obspy import read

import datetime as dtt

import datetime
from scipy.stats import kurtosis
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import MinMaxScaler
from scipy import spatial

from scipy.signal import butter, lfilter
#import librosa
# # sys.path.insert(0, '../01_DataPrep')
from scipy.io import loadmat
from sklearn.decomposition import PCA
# sys.path.append('.')
from sklearn.metrics import silhouette_samples
import scipy as sp
import scipy.signal

from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sklearn.metrics






##################################################################################################
# ##################################################################################################
#        _           _            _
#       | |         | |          (_)
#    ___| |_   _ ___| |_ ___ _ __ _ _ __   __ _
#   / __| | | | / __| __/ _ \ '__| | '_ \ / _` |
#  | (__| | |_| \__ \ ||  __/ |  | | | | | (_| |
#   \___|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
#                                          __/ |
#                                         |___/
##################################################################################################



def linearizeFP(path_proj,outfile_name,cat00):

    X = []
    with h5py.File(path_proj + outfile_name,'r') as MLout:
        for evID in cat00.event_ID:
            fp = MLout['SpecUFEX_output/fprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

def PCAonFP(X,cat00,numPCA=3,stand=True):
    ## performcs pca on fingerprints, returns catalog with PCs for each event


    # X = linearizeFP(path_proj,outfile_name,cat00)


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    sklearn_pca = PCA(n_components=numPCA)

    Y_pca = sklearn_pca.fit_transform(X_st)

    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    pca_df = pd.DataFrame(data=Y_pca,columns=pc_cols,index=cat00.index)

    PCA_df = pd.concat([cat00,pca_df], axis=1)  # not merge on IDs? no chance of re-sequencing ?

    return sklearn_pca, PCA_df, Y_pca # why return sklearn_pca ?






# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
# Percent variance explained?

def PVEofPCA(path_proj,outfile_name,cat00,numPCMax=100,cum_pve_thresh=.8,stand='MinMax'):

    X = linearizeFP(path_proj,outfile_name,cat00)


    if stand=='StandardScaler':
        X_st = StandardScaler().fit_transform(X)
    elif stand=='MinMax':
        X_st = MinMaxScaler().fit_transform(X)
    else:
        X_st = X


    numPCA_range = range(1,numPCMax)


    for numPCA in numPCA_range:

        sklearn_pca = PCA(n_components=numPCA)

        Y_pca = sklearn_pca.fit_transform(X_st)

        pve = sklearn_pca.explained_variance_ratio_

        cum_pve = pve.sum()
        print(numPCA,cum_pve)
        if cum_pve >= cum_pve_thresh:

            print('break')
            break



    pc_cols = [f'PC{pp}' for pp in range(1,numPCA+1)]

    PCA_df = pd.DataFrame(data = Y_pca, columns = pc_cols)


    return PCA_df, numPCA, cum_pve


# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo


def getTopFCat(cat0,topF,startInd=0,distMeasure = "SilhScore"):
    """


    Parameters
    ----------
    cat00 : all events
    topf : get top F events in each cluster
    startInd : can skip first event if needed
    Kopt : number of clusters
    distMeasure : type of distance emtrix between events. Default is "SilhScore",
    can also choose euclidean distance "EucDist"

    Returns
    -------
    catall : TYPE
        DESCRIPTION.

    """


    cat0['event_ID'] = [int(f) for f in  cat0['event_ID']]
    if distMeasure == "SilhScore":
        cat0 = cat0.sort_values(by='SS',ascending=False)

    if distMeasure == "EucDist":
        cat0 = cat0.sort_values(by='euc_dist',ascending=True)

    # try:
    cat0 = cat0[startInd:startInd+topF]
    # except: #if less than topF number events in cluster
    #     print(f"sampled all {len(cat0)} events in cluster!")

    # overwriting cat0 ?????
    return cat0



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo

#def calcSilhScore(path_proj,outfile_name,cat00,range_n_clusters,numPCA,distMeasure = "SilhScore",Xtype='fingerprints',stand=True):
def calcSilhScore(X,cat00,range_n_clusters,distMeasure = "SilhScore",stand=True):

    """


    Parameters
    ----------

    range_n_clusters : range type - 2 : Kmax clusters
    numPCA : number of principal components to perform clustering on (if not on FPs)
    Xtype : cluster directly on fingerprints or components of PCA. The default is 'fingerprints'.


    Returns
    -------
    Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters.

    """

## Return avg silh scores, avg SSEs, and Kopt for 2:Kmax clusters
## Returns altered cat00 dataframe with cluster labels and SS scores,
## Returns NEW catall dataframe with highest SS scores



## alt. X = 'PCA'

    # if X == 'fingerprints':
    #     X = linearizeFP(path_proj,outfile_name,cat00)
    #     pca_df = cat00
    # elif X == 'PCA':
    #     __, pca_df, X = PCAonFP(path_proj,outfile_name,cat00,numPCA=numPCA,stand=stand);

    pca_df = cat00

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
    pca_df['Cluster'] = cluster_labels_best
    pca_df['SS'] = ss_best
    pca_df['euc_dist'] = euc_dist_best


    ## make df for  top SS score rep events

    catall = pd.DataFrame();

    for k in range(1,Kopt+1):

        pca_df_top00 = pca_df.where(pca_df.Cluster==k).dropna();

        pca_df_top0 = getTopFCat(pca_df_top00,topF=1,startInd=0,distMeasure = distMeasure);


        catall = catall.append(pca_df_top0);


    catall['datetime_index'] = [pd.to_datetime(d) for d in catall.datetime];
    catall.sort_values(by='datetime_index');



    return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best



# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo




##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.OTHERoOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################
############ OTHER
############ OTHER
############ OTHER
############ OTHER

# Mean Absolute Error

def CalcScaleMAE(path_proj,cat00,Kopt,scale_range,RMM,sel_state,station,normed='median'):


    R2_all = []
    mae_Cbest= 1e12 #best of all clusters
    sca_Cbest = 1
    for k in range(1,Kopt+1):
#     for k in [2]:

        print(k)
        mae_min = 1e12 #best within a cluster
        sca_keep = 1


        if normed=='median':
            specMatsum_med_orig=getSpectraMedian(path_proj,cat00,k,station,normed=True)


        rec_state = RMM[:,sel_state[k-1]]

        if normed == 'max':
            specMatsum_med_orig=getSpectraMedian(path_proj,cat00,k,station,normed=True)
            specMatsum_med_orig = specMatsum_med_orig / np.max(specMatsum_med_orig)
            rec_state = rec_state / np.max(rec_state)

        for i, sca in enumerate(scale_range):

            specMatsum_med = specMatsum_med_orig * sca


#             mae_temp = r2_score(specMatsum_med, rec_state)
            mae_temp = sklearn.metrics.mean_absolute_error(specMatsum_med, rec_state)


            if mae_temp < mae_min:

                mae_min = mae_temp
                sca_keep = sca

        if mae_min < mae_Cbest:

            kmax = k
            mae_Cbest = mae_min
            sca_Cbest = sca_keep

            print(k, ': ', mae_Cbest, sca_Cbest)

        R2_k=r2_score(specMatsum_med, rec_state)

        R2_all.append(R2_k)

    return mae_Cbest,sca_Cbest, R2_all, kmax






##################################################################################################
# .oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo..oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo.oOo
##################################################################################################



def CalcDiffPeak(path_proj,cat00,k,RMM,sel_state,station):


    specMatsum=getSpectraMedian(path_proj,cat00,k,station,normed=True)
    rec_state = RMM[:,sel_state[k-1]]



    maxIDR = np.argwhere(rec_state==np.max(rec_state))
    maxIDS = np.argwhere(specMatsum==np.max(specMatsum))

    peak_rec_state = rec_state[maxIDR]
    peak_spec      = specMatsum[maxIDS]

    scale = peak_rec_state - peak_spec

    return int(peak_rec_state), int(peak_spec), int(scale)

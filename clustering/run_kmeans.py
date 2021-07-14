
#### After linearizing FPs into data array X, you can then do Kmeans on X
## Optimal cluster is chosen by mean Silh scores, but euclidean distances are saved also



def linearizeFP(path_proj,outfile_name,cat00):

    X = []
    with h5py.File(path_proj + outfile_name,'r') as MLout:
        for evID in cat00.event_ID:
            fp = MLout['SpecUFEX_output/fprints'].get(str(evID))[:]
            linFP = fp.reshape(1,len(fp)**2)[:][0]
            X.append(linFP)

    X = np.array(X)

    return X



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





        return pca_df,catall, Kopt, maxSilScore, avgSils, sse,cluster_labels_best,ss_best,euc_dist_best

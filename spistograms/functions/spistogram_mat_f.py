# spistogram maker ! module 

import numpy as np
#print('makeSpistoMatrix ready to work')


# 12/18-- gotta make a new column in cat_ML for "time_4plot", 
# the time as you want it to appear in the spistogram-- yearfrac is not generic enough. 

def makeSpistoMatrix(cat_ML,NC,n_bins,cluster_list):
    cn_str = 'Cluster' # 'cn_NC'+str(NC)
    
    #timeEvts_mat = np.zeros([len(cluster_list),n_bins-1])
    #bins_time = np.linspace(min(cat_ML.YearFrac),max(cat_ML.YearFrac),n_bins)
    bins_time = np.linspace(min(cat_ML.time_4plot),max(cat_ML.time_4plot),n_bins)
    #print(bins_time)
    
    for ic,cn in enumerate(cluster_list): #range(0,cn_max): 
    #for ic in range(len(cluster_list)): #range(0,cn_max): 
        #print(ic, cluster_list[ic])
        
        cat_tmp = cat_ML[cat_ML[cn_str]==cluster_list[ic]]
        #print(len(cat_tmp))
        timeEvts = cat_tmp.time_4plot
        #print(timeEvts)
        
        # density True makes a big difference !  check what it means ! 
        # normalize the hist ! a big decision ! 
        hist, bin_edges = np.histogram(timeEvts, bins=bins_time, density=False) 
        #hist, bin_edges = np.histogram(timeEvts_YrFrac, bins=bins_time, density=True) 

        if ic==0:
            timeEvts_mat = hist
        else:
            timeEvts_mat = np.vstack([timeEvts_mat,hist])

        del cat_tmp

    cl_extra = [0] +  list(range(1,len(cluster_list)+1)) # [0] + 
    #xgrid_spisto, ygrid_spisto = np.meshgrid(bin_edges,cl_extra)
    xgrid_spisto, ygrid_spisto = np.meshgrid(bins_time,cl_extra)
    
    return timeEvts_mat, xgrid_spisto, ygrid_spisto

# useful: from re-ordering: 
# for io,oldposn in enumerate(new_order):
#     newposn = io
#     #print(io,oldposn)
#     xhisto = timeEvts_mat[oldposn-1,:]
#     if io==0:
#         timeEvts_mat_NewO = xhisto
#     else:
#         timeEvts_mat_NewO = np.vstack([timeEvts_mat_NewO,xhisto])
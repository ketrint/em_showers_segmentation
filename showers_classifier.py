#!/usr/bin/env python
# coding: utf-8
import comet_ml
from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, average_precision_score

from utils.clustering_metrics import class_disbalance_graphx, class_disbalance_graphx__
from utils.clustering_metrics import estimate_e, estimate_start_xyz, estimate_txty
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


from xgboost import XGBClassifier


def predict_one_shower(shower, graph_embedder, edge_classifier):
    embeddings = graph_embedder(shower)
    edge_labels_true = (shower.y[shower.edge_index[0]] == shower.y[shower.edge_index[1]]).view(-1)
    edge_data = torch.cat([
        embeddings[shower.edge_index[0]],
        embeddings[shower.edge_index[1]]
    ], dim=1)
    edge_labels_predicted = edge_classifier(edge_data).view(-1)

    return edge_labels_true, edge_labels_predicted


# In[2]:


sns.set(context='paper', style="whitegrid", font_scale=3, font = 'serif')
colors = [
    'skyblue', 'orange', 'steelblue', 'gold', '#f58231', 'red' 
]
#get_ipython().run_line_magic('matplotlib', 'inline')
linewidth = 3

device = torch.device("cpu")

datafile_test='./data/clusters_test.pt'
datafile_val='./data/clusters_val.pt'



clusterized_bricks_test = torch.load(datafile_test)

clusterized_bricks_val = torch.load(datafile_val)


print("test data: ", len(clusterized_bricks_test), "\n val data: ", len(clusterized_bricks_val))


def class_disbalance_graphx(graphx):
    signal = []
    for _, node in graphx.nodes(data=True):
        signal.append(node['signal'])
    return list(zip(*np.unique(signal, return_counts=True)))



def data_collection(clusterized_bricks):

    selected_tracks = 0
    total_tracks = 0
    E = []
    E_true_all = []
    n_showers = []

    total_number_of_showers = 0
    number_of_lost_showers = 0
    second_to_first_ratios = 0
    number_of_good_showers = 0
    number_of_stucked_showers = 0
    number_of_broken_showers = 0

    second_to_first_ratios = []
    x_raw = []
    x_true = []

    y_raw = []
    y_true = []

    z_raw = []
    z_true = []

    tx_raw = []
    tx_true = []

    ty_raw = []
    ty_true = []
    n_showers = []

    e_stucked_10_list = []
    e_stucked_30_list = []
    e_stucked_50_list = []

    e_broken_10_list = []
    e_broken_30_list = []
    e_broken_50_list = []

    e_good_10_list = []
    e_good_30_list = []
    e_good_50_list = []

    for clusterized_brick in clusterized_bricks:
        showers_data = clusterized_brick['graphx'].graph['showers_data']
        #print(len(showers_data))


        clusters = clusterized_brick['clusters']
        raw_clusters = clusterized_brick['raw_clusters']
        #print('N predicted showers per brick:', len(raw_clusters))

        for shower_data in showers_data:
            shower_data['clusters'] = []
            shower_data['raw_clusters'] = []

        for cluster, raw_cluster in zip(clusters, raw_clusters):
            selected_tracks += len(cluster)
            for label, label_count in class_disbalance_graphx(cluster):
                if label_count / showers_data[label]['numtracks'] >= 0.1:
                    showers_data[label]['clusters'].append(cluster)
                    showers_data[label]['raw_clusters'].append(raw_cluster)

        for shower_data in showers_data:
            total_tracks += shower_data['numtracks']
            E.append(shower_data['numtracks'])
            E_true_all.append(shower_data['ele_P'])
            n_showers.append(len(showers_data)) 

        for shower_data in showers_data:
                total_number_of_showers += 1

                signals_per_cluster = []
                signals_per_cluster_bad = []
                idx_cluster = []
                for i, cluster in enumerate(shower_data['clusters']):
                    labels, counts = class_disbalance_graphx__(cluster)           
                    signals_per_cluster.append(counts[labels == shower_data['signal']][0])
                    idx_cluster.append(i)
                signals_per_cluster = np.array(signals_per_cluster)



                if len(signals_per_cluster) == 0:
                    number_of_lost_showers += 1               
                    continue
                if len(signals_per_cluster) == 1:
                    second_to_first_ratio = 0.
                    second_to_first_ratios.append(second_to_first_ratio)
                else:
                    second_to_first_ratio = np.sort(signals_per_cluster)[-2] / signals_per_cluster.max()
                    second_to_first_ratios.append(second_to_first_ratio)



                cluster = shower_data['clusters'][np.argmax(signals_per_cluster)]



                # not enough signal
                if (signals_per_cluster.max() / shower_data['numtracks']) <= 0.1:
                    number_of_lost_showers += 1
                    continue

                labels, counts = class_disbalance_graphx__(cluster)
                counts = counts / counts.sum()
                # high contamination
                if counts[labels == shower_data['signal']] < 0.9:
                    number_of_stucked_showers += 1
                    #cluster = shower_data['clusters'][0]
                    e_stucked_10 = estimate_e(cluster, angle=0.1)
                    e_stucked_10_list.append(e_stucked_10)

                    e_stucked_30 = estimate_e(cluster, angle=0.3)
                    e_stucked_30_list.append(e_stucked_30)

                    e_stucked_50 = estimate_e(cluster, angle=0.5)
                    e_stucked_50_list.append(e_stucked_50)

                    #print('stuck', shower_data['raw_clusters'])             
                    #stability_stucked.append(shower_data['raw_clusters'].stability)
                    continue

                if second_to_first_ratio > 0.3:
                    number_of_broken_showers += 1
                    #cluster = shower_data['clusters'][0]

                    e_broken_10 = estimate_e(cluster, angle=0.1)
                    e_broken_10_list.append(e_broken_10)

                    e_broken_30 = estimate_e(cluster, angle=0.3)
                    e_broken_30_list.append(e_broken_30)

                    e_broken_50 = estimate_e(cluster, angle=0.5)
                    e_broken_50_list.append(e_broken_50)

                    #print('broken', shower_data['raw_clusters'])
                    #stability_broken.append(shower_data['raw_clusters'][0].stability)
                    continue



                # for good showers
                number_of_good_showers += 1


                # x, y, z
                x, y, z = estimate_start_xyz(cluster)

                e_good_10 = estimate_e(cluster, angle=0.1)
                e_good_30 = estimate_e(cluster, angle=0.3)
                e_good_50 = estimate_e(cluster, angle=0.5)

                e_good_10_list.append(e_good_10)
                e_good_30_list.append(e_good_30)
                e_good_50_list.append(e_good_50)

                #print('good', shower_data['raw_clusters'])
                #stability_good.append(shower_data['raw_clusters'][0].stability)

                x_raw.append(x)
                x_true.append(shower_data['ele_SX'])

                y_raw.append(y)
                y_true.append(shower_data['ele_SY'])

                z_raw.append(z)
                z_true.append(shower_data['ele_SZ'])

                # tx, ty
                tx, ty = estimate_txty(cluster)

                tx_raw.append(tx)
                tx_true.append(shower_data['ele_TX'])

                ty_raw.append(ty)
                ty_true.append(shower_data['ele_TY'])

    e_stucked_10_list = np.array(e_stucked_10_list)
    e_stucked_30_list = np.array(e_stucked_30_list)
    e_stucked_50_list = np.array(e_stucked_50_list)

    e_broken_10_list = np.array(e_broken_10_list)
    e_broken_30_list = np.array(e_broken_30_list)
    e_broken_50_list = np.array(e_broken_50_list)

    e_good_10_list = np.array(e_good_10_list)
    e_good_30_list = np.array(e_good_30_list)
    e_good_50_list = np.array(e_good_50_list)
    
    e_10 = np.hstack((e_stucked_10_list, e_broken_10_list, e_good_10_list))
    
    labels = np.hstack((np.array([0]*len(e_stucked_10_list)), np.array([0]*len(e_broken_10_list)),
                   np.array([1]*len(e_good_10_list))))
    
    assert (len(labels)==len(e_10))

    e_30 = np.hstack((e_stucked_30_list, e_broken_30_list, e_good_30_list))
    e_50 = np.hstack((e_stucked_50_list, e_broken_50_list, e_good_50_list))
    
    data = np.vstack((e_10,e_30,e_50)).T
    
    return data, labels


X_train, y_train =  data_collection(clusterized_bricks_test)
X_test, y_test =  data_collection(clusterized_bricks_val)



xgb = XGBClassifier(n_estimators=2000,
                    verbose=0,
                    n_jobs=6,
                    random_state=1234)


from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


m = xgb
m.fit(X_train, y_train)

   
y_pred = m.predict(X_test) 
average_precision = average_precision_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)   
   

experiment = Experiment(project_name='classifier', workspace='ketrint', api_key = 'abUSnAytqEzSzLOxNLP1ohibs')


def plot_aucs(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(25, 9), dpi=100)

    ax = axs[0]
    ax.plot([0, 1], [0, 1], linestyle='--', rasterized=True)
    # plot the roc curve for the model
    auc = np.trapz(tpr, fpr)
    ax.plot(fpr, tpr, marker='.', rasterized=True)
    ax.set_title('ROC curve')
    ax.set_xlabel('FPR (Background efficiency)')
    ax.set_ylabel('TPR (Signal efficiency)')
    ax.legend()
    print(auc)

    ax = axs[1]
    average_precision = average_precision_score(y_true, y_pred)

    print(average_precision)
    ax.plot([0, 1], [0.5, 0.5], linestyle='--', rasterized=True)
    ax.plot(recall, precision, marker='.', rasterized=True)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall (efficiency)')
    ax.set_ylabel('\n \n  Precision (Purity)')
    ax.legend(loc = 'lower right')

    plt.savefig("Classificator_metrics.pdf", bbox_inches='tight')

    return fig


np.save('y_test.npy',y_test)
np.save('y_pred.npy',m.predict_proba(X_test)[:,1])


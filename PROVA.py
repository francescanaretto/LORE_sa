import sys

import gzip
import pickle
import datetime
import pandas as pd
from collections import defaultdict

from experiments.exputil import *
from experiments.expconfig import *
import bz2
from SuperLore.lorem_new import LOREM


def main():

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


    cxpb = 0.7
    mutpb = 0.5
    ngen = 20
    neigh_type = 'rndgenp'

    X_2e = np.load('../adult/adult-blackbox-data2.npz')['X_distance_separated']
    K = np.load('../adult/adult-blackbox-data2.npz')['X_train']
    class_name = 'Target'
    class_values = [0, 1]
    feature_names = list()
    for i in range(0, X_2e.shape[1]):
        feature_names.append(str(i))
    real_feature_names = feature_names
    numeric_columns = list()

    with bz2.BZ2File('../adult/adult_randfor.bz2') as f:
        bb = pickle.load(f)
    features_map = defaultdict(dict)
    i = 0
    j = 0

    while i < len(feature_names) and j < len(real_feature_names):
        if feature_names[i] == real_feature_names[j]:
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
            j += 1

        elif feature_names[i].startswith(real_feature_names[j]):
            features_map[j][feature_names[i].replace('%s=' % real_feature_names[j], '')] = i
            i += 1
        else:
            j += 1


    neigh_kwargs = {
        "balance": False,
        "sampling_kind": "gaussian",
        "kind": "gaussian",
        "downward_only": True,
        "redo_search": True,
        "forced_balance_ratio": 0.5,
        "cut_radius": True,
        "n": 8000,
        "normalize": 'minmax',
        "forced_balance_ratio": 0.5,
        "n_batch": 10,
        "datas": X_2e
    }

    print(datetime.datetime.now(), 'building LOREM explainer')
    explainer = LOREM(K, bb.predict, bb.predict_proba, feature_names, class_name, class_values, numeric_columns,
                      features_map,
                      neigh_type=neigh_type, categorical_use_prob=True, continuous_fun_estimation=True, size=1000,
                      ocr=0.1, multi_label=False, one_vs_rest=False, random_state=random_state, verbose=False,
                      ngen=ngen, cxpb=cxpb, mutpb=mutpb, Kc=X_2e, bb_predict_proba=bb.predict_proba, K_transformed=K,
                      discretize=True,
                      encdec=None, binary=False, **neigh_kwargs)


    for x in X_2e[0:1]:
        print('inizio con ', x)
        res = explainer.explain_instance_stable(x)
        print('finisco con ', res)



if __name__ == "__main__":
    main()

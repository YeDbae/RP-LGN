import numpy as np
import pandas as pd
from sklearn import preprocessing


def process_dataset(fc_data, fc_id, id2gender, id2pearson, label_df):
    final_label, final_pearson = [], []
    for fc, l in zip(fc_data, fc_id):
        if l in id2gender and l in id2pearson:
            if not np.any(np.isnan(id2pearson[l])):
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])
    final_pearson = np.array(final_pearson)  # (7901, 360, 360)
    _, _, node_feature_size = final_pearson.shape  # node_feature_size: 360
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label_df["sex"])
    labels = encoder.transform(final_label)
    return final_pearson, labels



def load_data_abide(abide_path):
    data = np.load(abide_path + '/abide.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

def load_data_adhd(abide_path):
    data = np.load(abide_path + '/adhd.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

def load_data_kki(abide_path):
    data = np.load(abide_path + '/kki.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

def load_data_nyu(abide_path):
    data = np.load(abide_path + '/nyu.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

def load_data_um(abide_path):
    data = np.load(abide_path + '/um.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

def load_data_ucla(abide_path):
    data = np.load(abide_path + '/ucla.npy', allow_pickle=True).item()
    final_pearson = data["corr"]
    labels = data["label"]

    return final_pearson, labels

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_ensemble_score(scores_dict, weights=None):
    # scores_dict: {model_name: score_array}
    keys = list(scores_dict.keys())
    arrs = [np.array(scores_dict[k]).reshape(-1,1) for k in keys]
    arrs = [MinMaxScaler().fit_transform(a) for a in arrs]
    arrs = np.hstack(arrs)
    if weights is None:
        weights = np.ones(len(keys)) / len(keys)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()
    ensemble_score = arrs @ weights
    return ensemble_score 
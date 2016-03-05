def get_sunrise_sunset():
    month_length = [31,28,31,30,31,30,31,31,30,31,30,31]

    with open('dataset/sunrise_sunset.txt') as f:
        content = f.readlines()

        date = {}

        for (i, month) in enumerate(month_length):
            for day in range(0, month):

                if i >= 9:
                    m = str(i+1)
                else:
                    m = '0' + str(i+1)

                if day >= 9:
                    d = str(day+1)
                else:
                    d = '0' + str(day+1)

                line = content[day]
                line = line[4:]

                size = 11
                sunrise = line[i*size:i*size+4]
                sunset = line[i*size+5:i*size+9]
                date['2009-'+m+'-'+d] = [sunrise, sunset]

    return date

def get_changes(feature):
    import numpy as np
    new_feature = np.zeros(len(feature))
    for idx, val in enumerate(feature):
        if idx == 0:
            new_feature[idx] = 0
        else:
            new_feature[idx] = feature[idx] - feature[idx-1]

    return new_feature

import numpy as np
from random import choice


def SMOTE(T, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples:
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours.

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """
    n_minority_samples, n_features = T.shape

    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")

    N = N/100
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))

    from sklearn.neighbors import NearestNeighbors
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)

    #Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(T[i].reshape(1, T.shape[1]), return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it
            while nn_index == i:
                nn_index = choice(nn[0])

            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]

    return S
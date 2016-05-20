from __future__ import division
import cPickle as pickle
import os
import numpy as np
import pandas as pd
from configuration import *

list_choose_features = ['RELH', 'TAIR', 'WSSD', 'WDIR', 'TA9M', 'PRES', 'SRAD']
timeline_length = 24
safety_zone = 24
window_size = 6
threshold_count_1 = 10


def get_data():

    file_name = directory + 'predict_onto_data_timeline_length_'+str(timeline_length)+'_safety_zone_'+str(safety_zone)+'.pkl'
    if os.path.isfile(file_name):
        with open(file_name, 'r') as f:
            result = pickle.load(f)

    else:
        print('Get data for drawing')
        print('Read data file')
        df = pd.read_csv(directory_scratch + data_name)
        print('Complete read data file')
        df = df.sort_values(by=['STID', 'Date', 'Time'])

        list_features_raw = list(df.columns.values)

        rain_index = list_features_raw.index('RAIN')

        data = np.array(df)

        # print('Get Rainiest days')
        # # rainy_days = get_rainiest_days(list_stations, df, rain_index, date_values)
        # print('Complete Get rainiest days')

        print('Get rain indices')
        rain_indices, non_rain_indices = get_rain_indices(data, rain_index)
        print('Complete Get Rain Indices')

        index_choose_features = [list_features_raw.index(x) for x in list_choose_features]

        print('Get rain and non rain data')

        def get_data(indices, label):

            temp_data = []
            temp_labels = []

            for index in indices:

                range_data = data[index-timeline_length:index+timeline_length, index_choose_features]
                list_features_values = np.array([range_data[i:i+window_size,:] for i in range(0,range_data.shape[0]-window_size)])
                shape = list_features_values.shape
                temp_values = list_features_values.reshape((shape[0], -1), order='F')

                temp_data.append(temp_values)

                if label == 1:
                    temp_labels.append(np.ones(shape[0]))
                else:
                    temp_labels.append(np.zeros(shape[0]))

            return temp_data, temp_labels

        rain_data, rain_labels = get_data(rain_indices, 1)
        non_rain_data, non_rain_labels = get_data(non_rain_indices, 0)

        print(len(rain_data), len(non_rain_data))

        combined_data = np.array(rain_data+non_rain_data)

        combined_labels = np.array(rain_labels+non_rain_labels)

        print('Complete Get rain and non rain data')

        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            from sklearn.cross_validation import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(combined_data, combined_labels, test_size=0.2, random_state=42)

        result = (X_train, y_train, X_test, y_test)

        with open(file_name, 'w') as f:
            pickle.dump(result, f)

    return result


def get_rain_indices(data, rain_index):

    rain_indices = []
    non_rain_indices = []
    last_indices = -7

    for i, rows in enumerate(data):

        if rows[rain_index] > 0 and last_indices < i - 6 and data[i-1, rain_index] == 0:
            rain_indices.append(i)
            last_indices = i

    last_indices = -7

    for i, rows in enumerate(data):
        # get each example separately at least safety_zone*5
        if i < 24 or last_indices >= i - safety_zone: continue

        if len(non_rain_indices) >= len(rain_indices):
            break

        if np.sum(data[i-safety_zone:i+safety_zone, rain_index]) == 0:
            non_rain_indices.append(i)
            last_indices = i

    return rain_indices, non_rain_indices
    pass


def run_RandomForest(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import RandomForestClassifier

    mean_data = np.mean(train_data,axis=0)
    train_data -= mean_data
    std_data = np.std(train_data.astype('float32'),axis=0)
    train_data /= std_data
    test_data -= mean_data
    test_data /= std_data

    train_data = train_data.reshape(-1, train_data.shape[2])
    train_labels = train_labels.reshape(-1, 1)[:,0]

    print('RUN random forest')

    clf = RandomForestClassifier()

    clf.fit(train_data, train_labels)

    rain_cases = []
    non_rain_cases = []
    predict_Y = []

    from itertools import groupby
    for i, test_sample in enumerate(test_data):
        predict_results_each_time_step = []

        for time_step_data in test_sample:
            result = clf.predict(time_step_data.reshape(1, -1))[0]
            predict_results_each_time_step.append(result)

        count_1 = np.sum(predict_results_each_time_step)

        group = [(k, sum(1 for t in g)) for k, g in groupby(predict_results_each_time_step)]

        max_1 = 0
        for (label, count) in group:
            if label == 1 and count > max_1:
                max_1 = count

        if test_labels[i,0] == 1:
            rain_cases.append([count_1, max_1])
        else:
            non_rain_cases.append([count_1, max_1])

        if count_1 >= threshold_count_1:
            predict_Y.append(1)
        else:
            predict_Y.append(0)

    # plot_stat(rain_cases, 1)
    # plot_stat(non_rain_cases, 0)

    test_labels = test_labels[:, 0]

    [precision, recall, fscore] = compute_metrics(predict_Y, test_labels)

    return predict_Y, [precision, recall, fscore]


def plot_stat(data, label):

    data = np.array(data).reshape((-1, 2))

    import matplotlib.pyplot as plt

    n_groups = data.shape[0]

    count_1 = data[:, 0]
    max_1 = data[:, 1]

    plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    plt.bar(index, count_1, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='Count Number of 1')

    plt.bar(index + bar_width, max_1, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='Max Number of Cons. 1')

    plt.xlabel('Cases')
    plt.ylabel('Number')
    if label == 1:
        plt.title('Rain Case Statistics')
    else:
        plt.title('Non Rain Case Statistics')

    plt.legend(loc='best')

    plt.tight_layout()

    plt.show()

    if label == 1:
        plt.savefig('images/rain_stats.png')
    else:
        plt.savefig('images/non_rain_stats.png')


def compute_metrics(predict_Y, test_labels, predict_Y_proba=None):
    temp_test_labels = test_labels

    # compute metrics
    from sklearn.metrics import precision_recall_fscore_support, \
    roc_curve, auc, accuracy_score, confusion_matrix

    precision, recall, fscore, _ = precision_recall_fscore_support(temp_test_labels, predict_Y, average='binary',
                                                                   pos_label=1)
    accuracy = accuracy_score(temp_test_labels, predict_Y)
    confusion = confusion_matrix(temp_test_labels, predict_Y)

    print "precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore)
    print "accuracy = " + str(accuracy)
    print "Confusion matrix: "
    print confusion

    if predict_Y_proba:
        fpr_rf, tpr_rf, _ = roc_curve(temp_test_labels, predict_Y_proba, pos_label=1)
        auc_score = auc(fpr_rf, tpr_rf)
        print "AUC = " + str(auc_score)

    return [precision, recall, fscore]
    pass


if __name__ == "__main__":

    np.random.seed(1991)

    (train_data, train_labels, test_data, test_labels) = get_data()

    run_RandomForest(train_data, train_labels, test_data, test_labels)

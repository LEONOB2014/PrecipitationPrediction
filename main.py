from __future__ import division
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping
from keras import layers
from keras.utils import np_utils

from support_functions import *

# plt.close('all')


# Global variables
train_data = []
train_labels = []
test_data = []
test_labels = []
list_features = []
X = []
Y = []
station = []

# optional variables
visualize = 0
sample_mode = 0  # 1: undersample, 2: oversample
convert_1_hour_period = 0  # 1: if convert to one hour period, not necessary
filter_data = 1
test_mode = 0
run_rnn_mode = 0
nearby_station_mode = 1 # 1 if consider nearby station RAIN values
run_mode = 1  # 1: for classify mode, 2: for regression mode

n_features = 0
minority_scale = 1
rain_threshold = 0.4  # inch 0.4 in = 1 cm

# Hyper-parameter variables
time_series_length = 50
batch_size = 32
window_size = 6
percent_of_testing = 0.1
nb_epochs = 100
stations = ['CARL', 'LANE', 'VANO', 'WEBR']
test_station = 'LANE'
start_date = '2009-01-01'


def read_data(window_size=6):  # read data and load into global variables

    global list_features, station

    df = pd.read_csv('dataset/All_WEBR_VANO_KING_CARL_LANE_2009.csv');

    station = 'CARL'

    list_features_raw = list(df.columns.values)
    list_features = [x for x in list_features_raw if not x.startswith('Q')]
    list_soil_temp_features = ['RAIN','TS05', 'TB05', 'TS10', 'TB10', 'TS30', 'BATV']

    list_features[0:6] = []
    list_features = [x for x in list_features if x not in list_soil_temp_features]

    # Get complete data with clean RAIN
    df = df.sort_values(by=['STID','Year', 'Month', 'Day', 'Time'])
    date = '2009-01-01'

    # --------------------------------------------
    # Filter some information
    df = df[
        #         (df.QRAIN == 0)
        (df.STID != 'KING')
        #         & (df.Month >= 6)
        #        &  (df.Date == date)
        #         & (df.Day >= 11)
        #         & (df.Day <= 14)
        #         & (df.Month <= 6)
    ]
    #--------------------------------------------

    rain_index = list_features_raw.index('RAIN')
    time_index = list_features_raw.index('Time')
    date_index = list_features_raw.index('Date')
    station_index = list_features_raw.index('STID')

    data = np.array(df)

    n_rows = data.shape[0]

    real_Y = np.zeros(shape=(n_rows))

    # Get rain data: Y label
    for idx, val in enumerate(data):
        if idx == 0 or data[idx, time_index] == 5 or (data[idx, time_index] == 0 and data[idx,date_index] == '2009-01-01'):
            real_Y[idx] = data[idx,rain_index]
        else:
            real_Y[idx] = data[idx,rain_index] - data[idx-1,rain_index]

    list_feature_index = [list_features_raw.index(x) for x in list_features]

    X = data[:, list_feature_index]

    if run_mode == 1:
        Y = (real_Y > 0).astype(int)
    elif run_mode == 2:
        Y = real_Y

    # convert to 1 hour period
    if convert_1_hour_period:
        X_new = []
        Y_new = []

        count_example = 0

        for i in range(0, X.shape[0]):

            if np.mod(int(data[i, time_index]), 100) == 0:
                one_hour_data_X = np.mean(X[range(i,i+12),:], axis=0).reshape((1,len(list_features)))
                one_hour_data_Y = np.sum(real_Y[range(i, i+12)], axis=0)
                X_new.append(one_hour_data_X)
                Y_new.append(one_hour_data_Y)
                count_example += 1
            else:
                pass

        X = np.reshape(X_new,(count_example, len(list_features)))

        if run_mode == 1:
            Y = (np.array(Y_new) > 0).astype(int)
        else:
            Y = np.copy(Y_new)

        n_rows = count_example

    temp_list = []

    # get and add change feature to list feature
    for i, feature in enumerate(list_features):

        feature_data = X[:, i]
        change_feature = np.reshape(get_changes(feature_data), (n_rows,1))

        X = np.append(X, change_feature,1)
        temp_list.append('C'+feature)

    list_features.append(temp_list)

    # visualize_data('RAIN', Y)

    # Filter Features
    if filter_data:
        list_choose_features = ['RELH', 'TAIR', 'WSSD', 'TA9M', 'PRES', 'WDIR', ]

        index_choose_features = [list_features.index(x) for x in list_choose_features]

        X = X[:, index_choose_features]

        list_features = list_choose_features

    # end filter Features

    # add window size
    if window_size != 0:
        Y_new = np.zeros(Y.shape)
        for i, _ in enumerate(Y):
            print str(i)+'/'+str(len(Y))

            Y_sum = 0
            for k in range(-window_size, window_size):
                if i + k < 0 or i + k >= X.shape[0]: continue
                Y_sum += Y[i+k]

            if run_mode == 1:
                Y_new[i] = Y_sum > rain_threshold
            else:
                Y_new[i] = Y_sum

        Y = np.copy(Y_new)

    # convert the data to Recurrent Network

    station_values = {}
    for s in stations:
        station_values[s] = data[data[:, station_index] == s, rain_index]

    station_start_indices = {}
    for s in stations:
        station_start_indices[s] = np.where(
            (data[:, station_index] == s) & (data[:, date_index] == start_date) & (data[:, time_index] == 0))

    if run_rnn_mode == 1:

        X_new = []
        Y_new = []

        for station in stations:
            nearby_stations = get_nearby_stations(station, 3)

            for i, y in enumerate(station_values[station]):
                if i - 50 < 0: continue

                range_indices = range(i-time_series_length,i)

                station_own_values = X[range(i - time_series_length + station_start_indices[station],
                                             i + station_start_indices[station]), :]

                if nearby_station_mode:
                    nearby_stations_value = np.transpose([station_values[s][range_indices] for s in nearby_stations])
                    values = np.append(station_own_values, nearby_stations_value, axis=1)
                else:
                    values = station_own_values

                X_new.append(values)
                Y_new.append(y)

        X = np.array(X_new)
        Y = np.array(Y_new)

        if run_mode == 1:
            Y = np_utils.to_categorical(Y, 2)

    # Add data for model 2

    X_new = []
    Y_new = []

    for station in stations:
        nearby_stations = get_nearby_stations(station, 3)

        for i, y in enumerate(station_values[station]):
            values = [station_values[s][i] for s in nearby_stations]

            X_new.append(values)
            Y_new.append(y)

    X_2 = np.array(X_new)

    if run_mode == 1:
        Y_2 = (np.array(Y_new) > 0).astype(int)
    else:
        Y_2 = np.array(Y_new)

    # Add data for model 3

    X_new = []
    Y_new = []

    for station in stations:
        nearby_stations = get_nearby_stations(station, 3)

        for i, y in enumerate(station_values[station]):
            station_own_values = X[station_start_indices[station][0] + i, :][0]

            # station_own_values = X[range_indices,:]
            nearby_stations_value = np.transpose([station_values[s][i] for s in nearby_stations])
            values = np.append(station_own_values, nearby_stations_value)

            X_new.append(values)
            Y_new.append(y)

    X_3 = np.array(X_new)

    if run_mode == 1:
        Y_3 = (np.array(Y_new) > 0).astype(int)
    else:
        Y_3 = np.array(Y_new)

    return X, Y, X_2, Y_2, X_3, Y_3


def get_nearby_stations(station, number):
    # change later
    list_nearby_stations = []

    for s in stations:
        if s != station:
            list_nearby_stations.append(s)

    return list_nearby_stations
    pass


def visualize_data(feature_name, feature_data):
    n_rows = feature_data.shape[0]
    plt.figure()
    plt.title(feature_name + ' over time at station: '+ station)
    plt.plot(np.arange(0,n_rows), feature_data, 'r', label=feature_name)
    plt.legend(loc='best')
    plt.savefig('images/'+feature_name+'_over_time_at_station: '+ station)

    plt.show()

    pass


def process_data(X, Y, permutation, minority_scale=1):
    new_X = X[permutation]
    new_Y = Y[permutation]

    test_index = int(X.shape[0]*(1-percent_of_testing))

    train_data = new_X[:test_index]
    train_labels = new_Y[:test_index]
    test_data = new_X[test_index:]
    test_labels = new_Y[test_index:]

    mean_data = np.mean(train_data,axis=0)
    train_data -= mean_data
    std_data = np.std(train_data.astype('float32'),axis=0)
    train_data /= std_data
    test_data -= mean_data
    test_data /= std_data

    return train_data, train_labels, test_data, test_labels


def find_important_features(window_size):
    from sklearn.ensemble import RandomForestClassifier
    # train by Random Forest
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
    forest = clf

    X = np.concatenate((train_data, test_data), axis=0)
    Y = np.concatenate((train_labels, test_labels), axis=0)

    forest.fit(X, Y)

    importances = forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    sorted_list_features = [list_features[x] for x in indices]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(2)

    plt.title("Feature importances All Data at window size = " + str(window_size))

    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), sorted_list_features)
    plt.xlim([-1, X.shape[1]])

    plt.savefig('images/importance_features_data_window_size_' + str(window_size) + '.png')

    # plt.show()

    pass


def compute_metrics(predict_Y, predict_Y_proba, test_labels):
    if run_rnn_mode:
        temp_test_labels = np.argmax(test_labels, axis=1)
        predict_Y_proba = predict_Y_proba[:, temp_test_labels]
    else:
        temp_test_labels = test_labels

    # compute metrics
    from sklearn.metrics import precision_recall_fscore_support, \
    roc_curve, auc, accuracy_score, confusion_matrix

    fpr_rf, tpr_rf, _ = roc_curve(temp_test_labels, predict_Y_proba, pos_label=1)
    auc_score = auc(fpr_rf, tpr_rf)
    precision, recall, fscore, _ = precision_recall_fscore_support(temp_test_labels, predict_Y, average='binary',
                                                                   pos_label=1)
    accuracy = accuracy_score(temp_test_labels, predict_Y)
    confusion = confusion_matrix(temp_test_labels, predict_Y)

    print "precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore)
    print "AUC = " + str(auc_score)
    print "accuracy = " + str(accuracy)
    print "Confusion matrix: "
    print confusion

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve at window size = ' + str(window_size) )
    plt.legend(loc='best')
    plt.savefig('images/ROC_curve_window_size_' + str(window_size) + '.png')

    return [precision, recall, fscore]
    pass


def run_RandomForest(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if run_mode == 1: # Classifion mode
        clf = RandomForestClassifier()

        predict_Y = clf.fit(train_data, train_labels).predict(test_data)
        predict_Y_proba = clf.predict_proba(test_data)[:,1]

        [precision, recall, fscore] = compute_metrics(predict_Y, predict_Y_proba, test_labels)

        return predict_Y, predict_Y_proba, [precision, recall, fscore]

    elif run_mode == 2: # Regression Mode
        clf = RandomForestRegressor()
        predict_Y = clf.fit(train_data, train_labels).predict(test_data)

        from sklearn.metrics import mean_squared_error

        mse = mean_squared_error(test_labels, predict_Y)

        print "mean squared error: " + str(mse)

        return mse


def run_RNN(train_data, train_labels, test_data, test_labels):

    rnn = 'SimpleRNN'
    early_stopping = EarlyStopping(monitor='val_error', patience=1)

    if run_mode == 1: # classification mode
        model = Sequential()
        model.add(getattr(layers, rnn)(512, input_shape=(time_series_length, train_data.shape[2]), return_sequences=False))
        model.add(Dense(2, activation='softmax'))

        model.load_weights(
            'models/models_weights_windows_' + str(window_size) + '_time_length_' + str(time_series_length) + '.h5')

        model.compile(loss='binary_crossentropy', optimizer='adagrad')

        # model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epochs,
        #       callbacks=[early_stopping],validation_data=(test_data, test_labels),show_accuracy=True)

        # model.save_weights('models/models_weights_windows_' + str(window_size) + '_time_length_' + str(time_series_length)+'.h5')

        predict_Y = model.predict_classes(test_data)
        predict_Y_proba = model.predict_proba(test_data)
        [precision, recall, fscore] = compute_metrics(predict_Y, predict_Y_proba, test_labels)

        return predict_Y, predict_Y_proba, [precision, recall, fscore]

    else: # regression mode
        model = Sequential()
        model.add(
            getattr(layers, rnn)(512, input_shape=(time_series_length, train_data.shape[2]), return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adagrad')

        model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epochs,
              callbacks=[early_stopping],validation_data=(test_data, test_labels))

        predict_Y = model.predict(test_data)

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(test_labels, predict_Y)

        print "mean squared error: " + str(mse)

    pass


if __name__ == "__main__":

    np.random.seed(1991)
    window_sizes = range(0,30)

    thresholds = range(0, 10)

    # read_data(window_size)
    # process_data(minority_scale)
    #
    # if run_rnn_mode == 1:
    #     run_RNN()
    # else:
    #     run_RandomForest()

    # list_results = np.empty((len(thresholds),3))
    #
    # for i, rain_threshold in enumerate(thresholds):
    #
    #     list_results[i,:] = run_RandomForest()
    #
    # # # print different window size
    # plt.figure(2)
    # plt.plot(thresholds, list_results[:,0], label='precision')
    # plt.plot(thresholds, list_results[:,1], label='recall')
    # plt.plot(thresholds, list_results[:,2], label='fscore')
    # plt.xlabel('Threshold Size')
    # plt.ylabel('Percent')
    # plt.title('Precision, Recall, F1')
    # plt.legend(loc='best')
    # plt.savefig('images/Result_with_Thresholds.png')
    #
    # print list_results

    X, Y, X_2, Y_2, X_3, Y_3 = read_data()

    permutation = np.random.permutation(X.shape[0])

    print("Run with Station own data only")
    train_data, train_labels, test_data, test_labels = process_data(X, Y, permutation)

    predict_Y, predict_Y_proba, _ = run_RandomForest(train_data, train_labels, test_data, test_labels)

    print("Run with Nearby Stations data")
    train_data, train_labels, test_data, test_labels = process_data(X_2, Y_2, permutation)

    predict_Y_2, predict_Y_2_proba, _ = run_RandomForest(train_data, train_labels, test_data, test_labels)

    print("Combine 2 result internally")
    train_data, train_labels, test_data, test_labels = process_data(X_3, Y_3, permutation)

    predict_Y_3, predict_Y_3_proba, _ = run_RandomForest(train_data, train_labels, test_data, test_labels)

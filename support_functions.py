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

def process():
        # # sample data for balance between classes
    # train_rain_examples = train_data[train_labels == 1,:]
    # train_non_rain_examples = (train_data[train_labels == 0, :])
    #
    # if sample_mode == 1: # under sample
    #     n_train_non_rain_examples = train_non_rain_examples.shape[0]
    #     train_non_rain_examples_new = train_non_rain_examples[:int(n_train_non_rain_examples*1/50),:]
    #     train_data = np.concatenate((train_rain_examples, train_non_rain_examples_new))
    #     train_labels = np.concatenate((np.ones(train_rain_examples.shape[0]), np.zeros(train_non_rain_examples_new.shape[0])))
    #
    #     pass
    #
    # elif sample_mode == 2 and minority_scale != 1: # over sample
    #     train_rain_examples_new = SMOTE(train_rain_examples, minority_scale*100, 5)
    #     train_data = np.concatenate((train_data, train_rain_examples_new))
    #     train_labels = np.concatenate((train_labels, np.ones((train_rain_examples_new.shape[0]))))
    #     pass

    pass


# def run_RNN(train_data, train_labels, test_data, test_labels):
#     from keras.callbacks import EarlyStopping
#     rnn = 'SimpleRNN'
#     early_stopping = EarlyStopping(monitor='val_error', patience=1)
#
#     if run_mode == 1: # classification mode
#         model = Sequential()
#         model.add(getattr(layers, rnn)(512, input_shape=(time_series_length, train_data.shape[2]), return_sequences=False))
#         model.add(Dense(2, activation='softmax'))
#
#         model.load_weights(
#             'models/models_weights_windows_' + str(window_size) + '_time_length_' + str(time_series_length) + '.h5')
#
#         model.compile(loss='binary_crossentropy', optimizer='adagrad')
#
#         # model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epochs,
#         #       callbacks=[early_stopping],validation_data=(test_data, test_labels),show_accuracy=True)
#         #
#         # model.save_weights('models/models_weights_windows_' + str(window_size) + '_time_length_' + str(time_series_length)+'.h5', overwrite=True)
#
#         predict_Y = model.predict_classes(test_data)
#         predict_Y_proba = model.predict_proba(test_data)
#         [precision, recall, fscore] = compute_metrics(predict_Y, predict_Y_proba, test_labels)
#
#         return predict_Y, predict_Y_proba, [precision, recall, fscore]
#
#     else: # regression mode
#         model = Sequential()
#         model.add(
#             getattr(layers, rnn)(512, input_shape=(time_series_length, train_data.shape[2]), return_sequences=False))
#         model.add(Dense(1))
#         model.compile(loss='mse', optimizer='adagrad')
#
#         model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=nb_epochs,
#               callbacks=[early_stopping],validation_data=(test_data, test_labels))
#
#         predict_Y = model.predict(test_data)
#
#         from sklearn.metrics import mean_squared_error
#         mse = mean_squared_error(test_labels, predict_Y)
#
#         print "mean squared error: " + str(mse)
#
#     pass

#--------------------------
# Convert data to recurrent network

# in read data

# from keras.utils import np_utils
#
# if run_rnn_mode == 1:
#
#         station_values = {}
#
#         for s in stations:
#             station_values[s] = Y[data[:, station_index] == s]
#
#         X_new = []
#         Y_new = []
#
#         for station in stations:
#             nearby_stations = get_nearby_stations(station, 3)
#
#             for i, y in enumerate(station_values[station]):
#                 if i - 50 < 0: continue
#
#                 range_indices = range(i-time_series_length,i)
#
#                 station_own_values = X[range(i - time_series_length + station_start_indices[station][0],
#                                              i + station_start_indices[station][0]), :]
#
#                 if nearby_station_mode:
#                     nearby_stations_value = np.transpose([station_values[s][range_indices] for s in nearby_stations])
#                     values = np.append(station_own_values, nearby_stations_value, axis=1)
#                 else:
#                     values = station_own_values
#
#                 X_new.append(values)
#                 Y_new.append(y)
#
#         X = np.array(X_new)
#         Y = np.array(Y_new)
#
#         print Y.shape, Y[0]
#
#         if run_mode == 1:
#             Y = np_utils.to_categorical(Y,2)
#
#         return X, Y
#
#     else:

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


        # def find_important_features(train_data, train_labels, test_data, test_labels, window_size=0):
        #     from sklearn.ensemble import RandomForestClassifier
        #     # train by Random Forest
        #     clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
        #     forest = clf
        #
        #     X = np.concatenate((train_data, test_data), axis=0)
        #     Y = np.concatenate((train_labels, test_labels), axis=0)
        #
        #     forest.fit(X, Y)
        #
        #     importances = forest.feature_importances_
        #
        #     std = np.std([tree.feature_importances_ for tree in forest.estimators_],
        #                  axis=0)
        #     indices = np.argsort(importances)[::-1]
        #     sorted_list_features = [list_features[x] for x in indices]
        #
        #     # Print the feature ranking
        #     print("Feature ranking:")
        #
        #     for f in range(X.shape[1]):
        #         print("%d. feature %d (%f)" % (f, indices[f], importances[indices[f]]))
        #
        #     # Plot the feature importances of the forest
        #     plt.figure(2)
        #
        #     plt.title("Feature importances All Data at window size = " + str(window_size))
        #
        #     plt.bar(range(X.shape[1]), importances[indices],
        #            color="r", yerr=std[indices], align="center")
        #     plt.xticks(range(X.shape[1]), sorted_list_features)
        #     plt.xlim([-1, X.shape[1]])
        #
        #     plt.savefig('images/importance_features_data_window_size_' + str(window_size) + '.png')
        #
        #     # plt.show()
        #
        #     pass

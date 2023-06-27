import os
import argparse
import json
import numpy as np
import helpers as hp
import multiprocessing as mp
from functools import partial
from sklearn import linear_model
from sklearn.metrics import mutual_info_score, accuracy_score
from scipy.stats import sem, t


NEURONS = 256

def mutual_info(x, y, bins=np.arange(3)):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def fit_evaluate(x_train, x_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    predictions = np.argmax(regr.predict(x_test), axis=1)
    return mutual_info_score(predictions, np.dot(y_test, np.arange(y_test.shape[1])))

def shuffle_data(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx, :, :], y[idx, :, :]

def average(data):
    avg = np.full((3, data.shape[0], data.shape[2]), np.nan)
    avg[0, :, :] = np.nanmean(data, axis=1)

    for i, j in hp.shape_iter((data.shape[0], data.shape[2])):
        d = data[i, :, j]
        d = d[~np.isnan(d)]
        std_err = sem(d.flatten())
        avg[1, i] = avg[0, i] - std_err * t.ppf((1 + 0.95) / 2, data.shape[1] - 1)
        avg[2, i] = avg[0, i] + std_err * t.ppf((1 + 0.95) / 2, data.shape[1] - 1)
        avg[1:, i, j] = hp.ci(data[i, :, j])
    return avg

def flatten(data):
    shape = data.shape
    new_shape = (shape[0] * shape[1], shape[2], shape[3])
    return data.reshape(new_shape)

def main(indices, config):
    print(indices)
    path = os.path.join(basedir, config["network_spikefiles"][indices[0]][indices[1]])

    letter = config["pre_letter"] + config["post_letter"] + 1
    bins = np.arange(config["options"]["offset"],
                     config["options"]["offset"] + \
                             (letter + 1) * config["duration"],
                     config["duration"])

    mi = np.full(bins.size - 1, np.nan)
    activity = np.zeros((config["classes"], config["samples"],
                         NEURONS, bins.size - 1))
    labels = np.zeros((config["classes"], config["samples"],
                       config["classes"], bins.size - 1))
    
    try:
        spikes = np.load(path, allow_pickle=True)
        # create dense activity matrix
        for c in range(config["classes"]):
            for s in range(config["samples"]):
                for n in range(NEURONS):
                    mask = (spikes[c][s][:, 1] == n)
                    activity[c, s, n, :] = np.histogram(spikes[c][s][mask, 0],
                                                        bins=bins)[0]
        # create label array
        for c in range(config["classes"]):
            labels[c, :, c, :] = 1

        # split in train and test set
        split = int(0.8 * activity.shape[1])
        activity_train = activity[:, :split, :, :]
        activity_test = activity[:, split:, :, :]
        labels_train = labels[:, :split, :]
        labels_test = labels[:, split:, :]

        # flatten class and sample axis
        activity_train = flatten(activity_train)
        activity_test = flatten(activity_test)
        labels_train = flatten(labels_train)
        labels_test = flatten(labels_test)

        activity_train, labels_train = shuffle_data(activity_train, labels_train)
        activity_test, labels_test = shuffle_data(activity_test, labels_test)

        for l in range(bins.size - 1):
            try:
                mi[l] = fit_evaluate(activity_train[:, :, l],
                                     activity_test[:, :, l],
                                     labels_train[:, :, l],
                                     labels_test[:, :, l])
            except np.linalg.LinAlgError:
                pass
            except:
                pass
    except IOError:
        print("File {} not found!".format(path))
    return mi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    values0 = args.config["values0"]
    values1 = args.config["values1"]

    shape = (len(values0), len(values1[:10]))

    with mp.Pool(20) as pool:
        mi = np.array(pool.map(partial(main, config=args.config),
                      hp.shape_iter(shape)))
    mi = mi.reshape(shape + (-1, ))
    mi = hp.average_ci(mi)

    np.savez(args.save, mi=mi, values0=values0)

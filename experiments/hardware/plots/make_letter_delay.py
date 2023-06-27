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

def filter_spiketrain(data, tmin, tmax):
    mask = np.logical_and(data[:, 0] >= tmin, data[:, 0] < tmax)
    data = data[mask, :]
    data -= tmin
    return data

def mutual_info(x, y, bins=np.arange(3)):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def fit_evaluate(x_train, x_test, y_train, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    predictions = np.argmax(regr.predict(x_test), axis=1)
    return mutual_info(predictions, np.dot(y_test, np.arange(y_test.shape[1])))

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
    return avg

def flatten(data):
    shape = data.shape
    new_shape = (shape[0] * shape[1], shape[2], shape[3])
    return data.reshape(new_shape)

def main(indices, config):
    print(indices)
    path = os.path.join(args.basedir,
                        config["network_spikefiles"][indices[0]][indices[1]])

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
                    neuron_mask = (spikes[c][s][:, 1] == n)
                    tmp = spikes[c][s][neuron_mask, 0]
                    for b in range(bins.size - 1):
                        time_mask = np.logical_and(tmp >= (bins[b] - args.binwidth),
                                                   tmp <   bins[b])
                        activity[c, s, n, b] = time_mask.sum() > 0
        
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
    parser.add_argument("--binwidth", type=float, default=30e-6)
    parser.add_argument("--step", type=float, default=10e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    args.basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    values0 = args.config["values0"]
    values1 = args.config["values1"]

    shape = (len(values0), len(values1))
    
    start = args.config["duration"] * (args.config["pre_letter"] - 2)
    end   = args.config["duration"] * (args.config["pre_letter"] + args.config["post_letter"] + 1 - 6)
    bins = np.arange(args.config["options"]["offset"] + start,
                     args.config["options"]["offset"] +   end,
                     args.step)
    boarders = list()
    boarders.append(args.config["options"]["offset"] + (args.config["pre_letter"] + 0) * args.config["duration"] - start)
    boarders.append(args.config["options"]["offset"] + (args.config["pre_letter"] + 1) * args.config["duration"] - start)

    with mp.Pool(20) as pool:
        mi = np.array(pool.map(partial(main, config=args.config),
                      hp.shape_iter(shape)))
    mi = mi.reshape(shape + (-1, ))
    mi = average(mi)

    in_spikes_cl1 = np.load(os.path.join(args.basedir, args.config["input_spikefiles"][2][0]), allow_pickle=True)[0][0]
    net_spikes_cl1 = np.load(os.path.join(args.basedir, args.config["network_spikefiles"][2][0]), allow_pickle=True)[0][0]
    
    in_spikes_cl2 = np.load(os.path.join(args.basedir, args.config["input_spikefiles"][2][0]), allow_pickle=True)[1][0]
    net_spikes_cl2 = np.load(os.path.join(args.basedir, args.config["network_spikefiles"][2][0]), allow_pickle=True)[1][0]
    
    bins_raster = np.arange(start, end, 1e-6)

    np.savez(args.save,
             mi=mi, values0=values0, bins=bins-bins[0], boarders=boarders,
             raster_in_cl1=filter_spiketrain(in_spikes_cl1, start, end),
             raster_net_cl1=filter_spiketrain(net_spikes_cl1, start, end),
             raster_in_cl2=filter_spiketrain(in_spikes_cl2, start, end),
             raster_net_cl2=filter_spiketrain(net_spikes_cl2, start, end),
             act_in_cl1=np.histogram(in_spikes_cl1[:, 0], bins_raster)[0],
             act_net_cl1=np.histogram(net_spikes_cl1[:, 0], bins_raster)[0],
             act_in_cl2=np.histogram(in_spikes_cl2[:, 0], bins_raster)[0],
             act_net_cl2=np.histogram(net_spikes_cl2[:, 0], bins_raster)[0],
             bins_raster=bins_raster-bins_raster[0])

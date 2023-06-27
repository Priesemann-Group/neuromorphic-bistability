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

def fit_evaluate(x_train, x_test, y_train, y_test, mode):
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    predictions = np.argmax(regr.predict(x_test), axis=1)
    if mode == "mi":
        return mutual_info(predictions, np.dot(y_test, np.arange(y_test.shape[1])))
    elif mode == "acc":
        return accuracy_score(predictions, np.dot(y_test, np.arange(y_test.shape[1])))
    elif mode == "both":
        return ( mutual_info(predictions, np.dot(y_test, np.arange(y_test.shape[1]))) ,
                 accuracy_score(predictions, np.dot(y_test, np.arange(y_test.shape[1]))) )
    else:
        print("wrong mode")
        exit()

def shuffle_data(x, y):
    idx = np.random.permutation(x.shape[0])
    return x[idx, :, :], y[idx, :, :]

def flatten(data):
    shape = data.shape
    new_shape = (shape[0] * shape[1], shape[2], shape[3])
    return data.reshape(new_shape)

def main(indices, config):
    print(indices)
    path = os.path.join(args.basedir,
                        config["input_spikefiles"][indices[0]][indices[1]])

    mi = np.full((2, bins.size - 1), np.nan)
    activity = np.zeros((config["classes"], config["samples"],
                         NEURONS, bins.size - 1))
    labels = np.zeros((config["classes"], config["samples"],
                       config["classes"], bins.size - 1))
    
    ZEROS = 0
    try:
        spikes = np.load(path, allow_pickle=True)
        # create dense activity matrix
        for c, cls in enumerate(config["myclasses"]):
            for s in range(config["samples"]):
                for n in range(NEURONS):
                    neuron_mask = (spikes[cls][s][:, 1] == n)
                    tmp = spikes[cls][s][neuron_mask, 0]
                    for b in range(bins.size - 1):
                        time_mask = np.logical_and(tmp >= (bins[b] - args.binwidth),
                                                   tmp <   bins[b])
                        activity[c, s, n, b] = time_mask.sum() > 0

                        if (n==NEURONS-1 and b!=0 and activity[c, s, :, b].sum()==0):
                            ZEROS += 1

        ZEROS = ZEROS / config["classes"] / config["samples"] / (bins.size-1)
        if ZEROS>0: print(indices, "Zeros:", ZEROS)


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
                mi[:, l] = fit_evaluate(activity_train[:, :, l],
                                     activity_test[:, :, l],
                                     labels_train[:, :, l],
                                     labels_test[:, :, l],
                                     mode="both")
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
    parser.add_argument("--binwidth", type=float, default=20e-6)
    parser.add_argument("--step", type=float, default=10e-6)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    args.basedir = os.path.dirname(args.config)
    with open(args.config) as h:
        args.config = json.load(h)

    values0 = args.config["values0"]
    values1 = args.config["values1"]
    shape = (len(values0), len(values1))
    print(shape)
    
    args.config["myclasses"] = [args.config["digits_X"].index(i) for i in [8,7]]
    args.config["classes"] = len(args.config["myclasses"])

    start = 0#args.config["durations"][0][0]-2*args.step
    end   = args.config["durations"][0][-1]+2*args.step
    bins = np.arange(args.config["options"]["offset"] + start,
                     args.config["options"]["offset"] +   end,
                     args.step)

    #args.config["durations"] = np.array(args.config["durations"]) - args.config["durations"][0][0] + 2*args.step
    args.config["durations"] = [args.config["durations"][i] for i in args.config["myclasses"]]

    with mp.Pool(20) as pool:
        mi = np.array(pool.map(partial(main, config=args.config),
                      hp.shape_iter(shape)))
    mi = mi.reshape(shape + (2, -1, ))
    mi = hp.average_ci(mi)

    freq, seed = 0, 1
    in_spikes_cl1 = np.load(
            os.path.join(args.basedir, args.config["input_spikefiles"][freq][seed]),
            allow_pickle=True)[0][0]
    net_spikes_cl1 = np.load(
            os.path.join(args.basedir, args.config["network_spikefiles"][freq][seed]),
            allow_pickle=True)[0][0]

    in_spikes_cl2 = np.load(
            os.path.join(args.basedir, args.config["input_spikefiles"][freq][seed]),
            allow_pickle=True)[1][0]
    net_spikes_cl2 = np.load(
            os.path.join(args.basedir, args.config["network_spikefiles"][freq][seed]),
            allow_pickle=True)[1][0]

    bins_raster = np.arange(start, end, 1e-6)

    np.savez(args.save,
             mi=mi, values0=values0, bins=bins-bins[0],
             durations=args.config["durations"],
             raster_in_cl1=filter_spiketrain(in_spikes_cl1, start, end),
             raster_net_cl1=filter_spiketrain(net_spikes_cl1, start, end),
             raster_in_cl2=filter_spiketrain(in_spikes_cl2, start, end),
             raster_net_cl2=filter_spiketrain(net_spikes_cl2, start, end),
             act_in_cl1=np.histogram(in_spikes_cl1[:, 0], bins_raster)[0],
             act_net_cl1=np.histogram(net_spikes_cl1[:, 0], bins_raster)[0],
             act_in_cl2=np.histogram(in_spikes_cl2[:, 0], bins_raster)[0],
             act_net_cl2=np.histogram(net_spikes_cl2[:, 0], bins_raster)[0],
             bins_raster=bins_raster-bins_raster[0])

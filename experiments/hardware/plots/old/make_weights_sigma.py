import data
import numpy as np
import helpers as hp


def average(data):
    avg = np.zeros((3, data.shape[0], data.shape[1], data.shape[3]))
    avg[0, :, :, :] = np.nanmedian(data, axis=2)
    for i, j, k in hp.shape_iter((data.shape[0], data.shape[1], data.shape[3])):
        avg[1:, i, j, k] = hp.ci(data[i, j, :, k])
    return avg


if __name__ == "__main__":
    conf = data.Sigma()
    
    paths = [conf.weights_path, conf.weights_path_uncalib]
    
    bins = np.arange(1, 64, 1)

    weights_ext = np.full((len(paths), ) + conf.shape + (bins.size-1, ), np.nan)
    weights_int = np.full((len(paths), ) + conf.shape + (bins.size-1, ), np.nan)
    for p, path in enumerate(paths):
        for i, sigma in enumerate(conf.sigmas):
            for j, seed in enumerate(conf.seeds):
                d = np.load(path.format(sigma, seed))
                w = d["weights"]
                # inhibitory_mask = -1 * (2 * d["inhibitory_mask"] - 1)

                # w *= inhibitory_mask[:, None]
                weights_ext[p, i, j, :] = np.histogram(w[conf.Nneurons:, :], bins)[0]
                weights_int[p, i, j, :] = np.histogram(w[:conf.Nneurons, :], bins)[0]

    weights_ext = average(weights_ext)
    weights_int = average(weights_int)

    np.savez("data/weights_sigma.npz",
             weights_ext=weights_ext, weights_int=weights_int, bins=bins)

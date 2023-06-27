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
    conf = data.Standard()
    
    paths = [conf.weights_path, conf.weights_path_uncalib]
    
    bins = np.arange(0, 64, 1)
    bins_weighted = np.arange(100, 1000, 10)

    degree_ext = np.full((len(paths), ) + conf.shape + (bins.size-1, ), np.nan)
    degree_int = np.full((len(paths), ) + conf.shape + (bins.size-1, ), np.nan)
    
    degree_ext_w = np.full((len(paths), ) + conf.shape + (bins_weighted.size-1, ), np.nan)
    degree_int_w = np.full((len(paths), ) + conf.shape + (bins_weighted.size-1, ), np.nan)
    
    for p, path in enumerate(paths):
        for i, freq in enumerate(conf.freqs):
            for j, seed in enumerate(conf.seeds):
                d = np.load(path.format(freq, seed))
                w = d["weights"]
                # inhibitory_mask = -1 * (2 * d["inhibitory_mask"] - 1)
                
                degree_ext[p, i, j, :] = np.histogram((w[conf.Nneurons:, :] > 0).sum(axis=0), bins)[0]
                degree_int[p, i, j, :] = np.histogram((w[:conf.Nneurons, :] > 0).sum(axis=0), bins)[0]

                # w *= inhibitory_mask[:, None]
                degree_ext_w[p, i, j, :] = np.histogram(w[conf.Nneurons:, :].sum(axis=0), bins_weighted)[0]
                degree_int_w[p, i, j, :] = np.histogram(w[:conf.Nneurons, :].sum(axis=0), bins_weighted)[0]

    degree_ext = average(degree_ext)
    degree_int = average(degree_int)
    
    degree_ext_w= average(degree_ext_w)
    degree_int_w= average(degree_int_w)

    np.savez("data/degree.npz",
             degree_ext=degree_ext, degree_int=degree_int,
             degree_ext_w=degree_ext_w, degree_int_w=degree_int_w,
             bins=bins, bins_weighted=bins_weighted)

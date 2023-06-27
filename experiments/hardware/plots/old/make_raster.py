import data
import helpers as hp
import numpy as np


TMIN = 0.019
TMAX = 0.020


def rate(data, conf):
    times = data[:, 0]
    duration = float(conf.tmax - conf.tmin)
    return times.size / float(conf.Nneurons) / duration / conf.speedup

def filter_spiketrain(data):
    mask = np.logical_and(data[:, 0] >= TMIN, data[:, 0] < TMAX)
    data = data[mask, :]
    data[:, 0] -= TMIN
    data[:, 0] *= conf.speedup
    return data

def activity(data, bins):
    return np.histogram(data[:, 0], bins=bins)[0]


if __name__ == "__main__":
    conf = data.Standard()

    rates = np.zeros((conf.Nfreqs, conf.Nseeds))
    for f, freq in enumerate(conf.freqs):
        for s in conf.seeds:
            data = np.load(conf.network_spike_path.format(freq, s))
            rates[f, s] = rate(data, conf)
    rates = hp.median_ci(rates)

    freqs = np.array([conf.freqs[2], conf.freqs[-1]])

    low = np.load(conf.network_spike_path.format(freqs[0], 0))
    high = np.load(conf.network_spike_path.format(freqs[1], 0))

    bins = np.arange(TMIN, TMAX + 1e-6, 1e-6)

    np.savez("data/raster.npz",
             rates=rates, freqs=freqs, bins=bins,
             raster_low=filter_spiketrain(low),
             raster_high=filter_spiketrain(high),
             activity_low=activity(low, bins),
             activity_high=activity(high, bins))

import data
import numpy as np
import mrestimator as mre
import matplotlib.pyplot as plt


def estimate_activity(freq, conf, path):
    print(freq)
    activity = np.full((conf.Nseeds, conf.bins.size - 1), 0.0)
    for s in range(conf.Nseeds):
        try:
            spikes = np.load(path.format(freq, s))
            activity[s, :] = np.histogram(spikes[:, 0], bins=conf.bins)[0]
        except IOError:
            print("File not found")
    return activity


if __name__ == "__main__":
    conf = data.Standard()

    paths = [conf.network_spike_path, conf.network_spike_path_uncalib]

    taus = np.zeros((len(paths), 3, conf.Nfreqs))
    rks = np.zeros((len(paths), 3, conf.Nfreqs,
                    conf.steps[1] - conf.steps[0] + 1))
    
    for p, path in enumerate(paths):
        for f, freq in enumerate(conf.freqs):
            activity = estimate_activity(freq, conf, path)

            # different methods:
            # 'ts': trial separated
            # 'sm': stationary mean
            rk = mre.coefficients(activity, 
                                  steps=conf.steps, dtunit='step', method='sm')

            # different fit functions:
            # 'exponential': exponential function without offset
            # 'offset': exponential function with offset
            m = mre.fit(rk, fitfunc='offset', numboot=100, quantiles=[0.25, 0.75])

            # plot correlation coefficients and fit
            # ores = mre.OutputHandler()
            # ores.add_coefficients(rk)
            # ores.add_fit(m)
            # plt.show()

            # get correlation coefficients
            rks[p, 0, f, :] = rk.coefficients
            rks[p, 1, f, :] = rk.coefficients + rk.stderrs
            rks[p, 2, f, :] = rk.coefficients - rk.stderrs

            # get correlation time constant
            taus[p, 0, f] = m.tau * conf.binwidth * conf.speedup
            taus[p, 1:, f] = m.tauquantiles * conf.binwidth * conf.speedup
    
    dts = rk.steps * conf.binwidth * conf.speedup

    np.savez("data/ac.npz", rks=rks, taus=taus, dts=dts)

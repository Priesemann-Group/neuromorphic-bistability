from brian2 import Hz, ms, mV, defaultclock, seed, prefs, randn, NeuronGroup, Synapses, PoissonGroup, rand, run, SpikeMonitor, PopulationRateMonitor, PoissonInput, StateMonitor
from brian2 import *
import brian2.only as br2
import brian2.numpy_ as np
import argparse
from tqdm import tqdm
import os
import shutil


# assign directory for compiled code to later explicitly remove it again
cache_dir = os.path.expanduser(f"~/.cython/brian-pid-{os.getpid()}")
prefs.codegen.runtime.cython.cache_dir = cache_dir
prefs.codegen.runtime.cython.multiprocess_safe = False
# enforce simulation with c
prefs.codegen.target = "cython"


def relu(x):
    x[x < 0] = 0
    return x


def integer_weights(x):
    """ cast to integer weights between 0 and 62 """
    x = np.clip(np.floor(x), 0, 63)
    return x


parser = argparse.ArgumentParser(description='Script for simmulation of experiment with Brian2 '
    'Either run a homeostatically adapted experiment, or one with fixed weights')
parser.add_argument('-o', type=str, required=True, help='output path')
parser.add_argument('-s', type=int, required=True, help='seed')
parser.add_argument('-g', type=float, required=False, help='g', default=1.0)
parser.add_argument('-Kext', type=int, required=False, help='number of external connections per neurons', default=90)
parser.add_argument('-N', type=int, required=False, help='number of neurons', default=512)
parser.add_argument('-n', type=int, required=False, help='number iterations', default=500)
parser.add_argument('-p', type=float, required=False, help='probability to update synapses in homeostatic step', default=0.025)
parser.add_argument('--learning_rate', type=float, required=False, default=0.5 * 0.75)
parser.add_argument('--no_int_weights', action='store_true', help='turn off integer weights')
parser.add_argument('--no_paranoise', action='store_true', help='turn off parameter noise')
parser.add_argument('--no_tempnoise', action='store_true', help='turn off temporal noise on membrane potential')
parser.add_argument('--no_offset', action='store_true', help='turn off synaptic scale offset')
parser.add_argument('--no_inh', action='store_true', help='no external inhibitory synapses')
parser.add_argument('--no_shared', action='store_true', help='unique external Poisson groups for each neuron')
parser.add_argument('--adapt', action='store_true', help='run the homeostatic adaptation experiment, else just run static experiment with fixed weights')
parser.add_argument('--adapt_ext', action='store_true', help='also homeostatically adapt external weights')
parser.add_argument('--trace', action='store_true', help='recorde v and I traces for subpopulation')
parser.add_argument('--RESUME', action='store_true', help='load saved state and resume with more updates')
parser.add_argument('--SAVE_STATE', action='store_true', help='save a state after homeostasis for later resuming')
args = parser.parse_args()

# set seed for stochastic values
seed(args.s)

defaultclock.dt = 50e-3 * ms

# fixed weights
fixed_w_ext_exc = 17
fixed_w_ext_inh = 17
fixed_w_rec_exc = 10  # can be determined from mean of distribution when adapting recurrent weights
# Alternative parameters for stronger recurrence and input,  compare to: 2mV rec weights at 20mV dyn range, 0.1 mV ext
# fixed_w_ext_exc = 10
# fixed_w_ext_inh = 10
# fixed_w_rec_exc= 50
# nu_ext = 150
fixed_w_rec_inh = fixed_w_rec_exc * args.g

# homeostatic parameters
nu_target = 10 * Hz
# external input rate per neuron is considered to be constant. This is
# biologically plausible (consistent with the homeostatic target rate) and
# relevant for the neuromorphic chip to maintain stable synaptic strengths
learning_rate = args.learning_rate
p_update = args.p
num_iterations = args.n
time_iteration = 1000 * ms
# equilibration time between recordings that approximates the asynchronous
# weight updates on the neuromorphic chip which here happen "instantaneously"
time_equilibration = 1000 * ms

time_meas = 100 * 1000 * ms

K_ext = args.Kext
nu_ext = nu_target

# network
N = args.N
N_inh = round(N * 0.2)
N_exc = round(N * 0.8)
N_ext = round(N / 2)
N_ext_inh = round(N_ext * 0.2)
N_ext_exc = round(N_ext * 0.8)
# connection neurons similar to neuromorphic chip implementation
K_rec = 102

filename = f'{args.o}/data_brian'
if not args.adapt:
    filename += "_phase-diagram"
if args.no_shared:
    filename += "-noshared"
if args.no_inh:
    filename += "-noinh"
if args.no_tempnoise:
    filename += '_no-tempnoise'
filename += f'_N{N}_g{args.g}_Kext{K_ext}_seed{args.s}.npz'
print(f'will write to {filename}')

###############################################################################

# neuron and synapse parameters
params = """\
I0_exc: 0.003993204955005641/0.0006482469399926281
I0_inh: -0.0047411477195369485/0.0006801625964422302
I0_exc_offset: -0.004627577302205717/0.008171815023154099
I0_inh_offset: 0.00493467674924135/0.007902658553102578
tau_m: 21.480/1.448
tau_syn_exc: 5.105/0.247
tau_syn_inh: 5.182/0.208
v_leak: 0.455/0.029
v_thres: 0.740/0.006
v_reset: 0.323/0.006
"""
parameters = dict()
for line in params.splitlines():
    split = line.split(": ")
    values = split[1].split("/")
    unit = ms if "tau" in split[0] else 1000 * mV
    parameters[split[0]] = float(values[0]) * unit
    parameters["sigma_" + split[0]] = float(values[1]) * unit
parameters["tau_ref"] = 2 * ms
parameters["d_syn"] = 1 * ms
parameters["I0_inh"] *= -1
parameters["I0_inh_offset"] *= -1

if args.no_paranoise:
    parameters["sigma_v_thres"] = parameters["sigma_v_leak"] = parameters["sigma_v_reset"] = 0 * mV
    parameters["sigma_tau_m"] = parameters["sigma_tau_syn_exc"] = parameters["sigma_tau_syn_inh"] = 0 * ms
    parameters["sigma_I0_exc"] = parameters["sigma_I0_inh"] = parameters["sigma_I0_exc_offset"] = parameters["sigma_I0_inh_offset"] = 0 * mV

# temporal white noise on membrane potential
parameters["sigma"] = 2.0 * mV
if args.no_tempnoise:
    parameters["sigma"] = 0.0 * mV

###############################################################################

# differential equations
eqs = '''
du/dt = -(u-u_leak_i)/tau_mem_i  + (I_exc - I_inh)/tau_mem_i + sigma*sqrt(2/tau_mem_i)*xi : volt (unless refractory)
dI_exc/dt = -I_exc/tau_exc_i : volt
dI_inh/dt = -I_inh/tau_inh_i : volt
#additional parameters
tau_mem_i : second
tau_exc_i : second
tau_inh_i : second
u_leak_i  : volt
u_reset_i : volt
u_thres_i : volt
'''
# define group of Neurons
# use explicit namespace: https://brian2.readthedocs.io/en/stable/advanced/namespaces.html
G = NeuronGroup(N, eqs, threshold='u>u_thres_i', reset='u=u_reset_i', refractory='tau_ref', method='euler', namespace=parameters)
G.u = 'v_leak'
# parameter noise (Gaussian)
G.u_thres_i = 'v_thres     + randn()*sigma_v_thres'
G.u_leak_i = 'v_leak      + randn()*sigma_v_leak'
G.u_reset_i = 'v_reset     + randn()*sigma_v_reset'
G.tau_mem_i = 'tau_m       + randn()*sigma_tau_m'
G.tau_exc_i = 'tau_syn_exc + randn()*sigma_tau_syn_exc'
G.tau_inh_i = 'tau_syn_inh + randn()*sigma_tau_syn_inh'


# synapses
model_synapse = """
w : 1
scale_i : volt
offset_i : volt
"""
G_exc = G[:N_exc]
G_inh = G[N_exc:]
S_rec_exc = Synapses(G_exc, G, model=model_synapse, on_pre='I_exc += scale_i*w + offset_i', delay=parameters["d_syn"], namespace={})
S_rec_inh = Synapses(G_inh, G, model=model_synapse, on_pre='I_inh += scale_i*w + offset_i', delay=parameters["d_syn"], namespace={})
# self connections are possible
S_rec_exc.connect(p=K_rec / N)
S_rec_inh.connect(p=K_rec / N)


# set fixed recurrent delta weights
if not args.adapt:
    S_rec_exc.w = fixed_w_rec_exc
    S_rec_inh.w = fixed_w_rec_inh


# shared input channels (like on chip)
if not args.no_shared:
    # external input is implemented by 256 Poisson neurons that are randomly connected to network
    P_ext = PoissonGroup(N_ext, nu_ext)
    S_ext_exc = Synapses(P_ext[:N_ext_exc], G, model=model_synapse, on_pre='I_exc += scale_i*w + offset_i', delay=parameters["d_syn"], namespace={})
    S_ext_exc.connect(p=K_ext / N_ext)
    if not args.no_inh:
        S_ext_inh = Synapses(P_ext[N_ext_exc:], G, model=model_synapse, on_pre='I_inh += scale_i*w + offset_i', delay=parameters["d_syn"], namespace={})
        S_ext_inh.connect(p=K_ext / N_ext)

    # set fixed external delta weights
    if not args.adapt_ext:
        S_ext_exc.w = fixed_w_ext_exc
        if not args.no_inh:
            S_ext_inh.w = fixed_w_ext_inh

    for j in range(len(G)):
        # excitatory current
        scale = parameters["I0_exc"] + randn() * parameters["sigma_I0_exc"]
        offset = parameters["I0_exc_offset"] + randn() * parameters["sigma_I0_exc_offset"]
        for S in [S_rec_exc, S_ext_exc]:
            S.scale_i[:, j] = scale
            S.offset_i[:, j] = offset
        # inhibitory current
        scale = parameters["I0_inh"] + randn() * parameters["sigma_I0_inh"]
        offset = parameters["I0_inh_offset"] + randn() * parameters["sigma_I0_inh_offset"]
        S_inhs = [S_rec_inh]
        if not args.no_inh:
            S_inhs.append(S_ext_inh)
        for S in S_inhs:
            S.scale_i[:, j] = scale
            S.offset_i[:, j] = offset

# unique Poisson input channels for each neuron
else:
    n_exc_ext = int(N_exc / N * K_ext)
    G.namespace['scale_exc'] = parameters["I0_exc"]  # + randn() * parameters["sigma_I0_exc"]  # TODO No Noise!
    G.namespace['offset_exc'] = 0 * mV  # parameters["I0_exc_offset"] + randn() * parameters["sigma_I0_exc_offset"]
    G.namespace['fixed_w_ext_exc'] = fixed_w_ext_exc
    input_exc = PoissonInput(target=G, target_var="I_exc", N=n_exc_ext, rate=nu_ext, weight="scale_exc * fixed_w_ext_exc + offset_exc")
    if not args.no_inh:
        n_inh_ext = int(N_inh / N * K_ext)
        G.namespace['scale_inh'] = parameters["I0_inh"] + randn() * parameters["sigma_I0_inh"]
        G.namespace['offset_inh'] = parameters["I0_inh_offset"] + randn() * parameters["sigma_I0_inh_offset"]
        G.namespace['fixed_w_ext_inh'] = fixed_w_ext_inh
        input_inh = PoissonInput(target=G, target_var="I_inh", N=n_inh_ext, rate=nu_ext, weight="scale_inh * fixed_w_ext_inh + offset_inh")

    # recurrent synapses
    for j in range(len(G)):
        S_rec_exc.scale_i[:, j] = parameters["I0_exc"] + randn() * parameters["sigma_I0_exc"]
        S_rec_exc.offset_i[:, j] = parameters["I0_exc_offset"] + randn() * parameters["sigma_I0_exc_offset"]
        S_rec_inh.scale_i[:, j] = parameters["I0_inh"] + randn() * parameters["sigma_I0_inh"]
        S_rec_inh.offset_i[:, j] = parameters["I0_inh_offset"] + randn() * parameters["sigma_I0_inh_offset"]


if args.adapt:
    # check number of synapses per neuron
    print('mean number of external synapses per neuron: ', np.mean([(len(S_ext_exc.w[:, j]) + len(S_ext_inh.w[:, j])) for j in range(1, len(G))]))
    print('mean number of recurrent synapses per neuron: ', np.mean([(len(S_rec_exc.w[:, j]) + len(S_rec_inh.w[:, j])) for j in range(1, len(G))]))
    print('mean offset of excitatory synapses: ', (np.mean(S_rec_exc.offset_i[:]) + np.mean(S_ext_exc.offset_i[:])) / 2)
    print('mean offset of inhibitory synapses: ', (np.mean(S_rec_inh.offset_i[:]) + np.mean(S_ext_inh.offset_i[:])) / 2)


###############################################################################

if args.adapt:

    SYNAPSES = [S_rec_exc, S_rec_inh]
    if args.adapt_ext:
        SYNAPSES.append(S_ext_exc)
        SYNAPSES.append(S_ext_inh)

    print('### homeostatic iteration')
    spike = SpikeMonitor(G, record=False, name="myspikemonitor")

    if args.RESUME:
        nu_log = list(np.load(filename + "_tmp.npy"))
        br2.restore(filename=filename + "_tmp")
        print(len(nu_log))
    else:
        nu_log = []

    for iteration in tqdm(range(num_iterations)):
        run(time_equilibration, namespace={})
        spike = SpikeMonitor(G, record=False, name="myspikemonitor")
        run(time_iteration, namespace={})
        # analyze past firing rate and update synapses
        nus = spike.count / time_iteration
        delta_weight = learning_rate * (nu_target - nus) * ms * 1000
        for j in range(len(G)):
            for S in SYNAPSES:
                if len(S.w[:, j]) > 0:
                    if args.no_int_weights:
                        S.w[:, j] = relu(S.w[:, j] + delta_weight[j] * (rand(len(S.w[:, j])) < p_update))
                    else:  # this is default with integer weight arithmetic
                        S.w[:, j] = integer_weights(S.w[:, j] + delta_weight[j] * (rand(len(S.w[:, j])) < p_update))
        nu_log.append(spike.num_spikes / len(G) / time_iteration)

    if args.SAVE_STATE:
        np.save(filename + "_tmp", nu_log)
        br2.store(filename=filename + "_tmp")

else:
    # kick the network with strong external input to an active state and let it equilibrate
    P_kick = PoissonGroup(N / 2, 10 * nu_ext)
    S_kick = Synapses(P_kick, G, model=model_synapse, on_pre='I_exc += scale_i*w + offset_i', delay=parameters["d_syn"], namespace={})
    S_kick.connect(p=1)
    S_kick.scale_i[:] = parameters["I0_exc"]
    S_kick.offset_i[:] = 0 * mV
    S_kick.w = fixed_w_ext_exc
    run(5000 * ms, profile=True, namespace={})
    S_kick.w = 0
    print("### static equilibration")
    run(5000 * ms, profile=True, namespace={})

###############################################################################

# final measurements (fix weights for measurement)
print("### final measurement without homeostasis")
M = SpikeMonitor(G)
R = PopulationRateMonitor(G)
if args.trace:
    STATE = StateMonitor(G, ('u', "I_exc", "I_inh"), record=[0, 1])
run(time_meas, profile=True, namespace={})

###############################################################################

# convert to experimental format
spike_time_array = np.zeros((len(M.i), 2))
spike_time_array[:, 0] = np.array(M.t / ms)
spike_time_array[:, 1] = M.i

data = dict(
    time=R.t,
    rate=R.rate / Hz,
    spikes=spike_time_array,
)
if args.trace:
    data = {**data, **dict(
        t=STATE.t / ms,
        v=STATE.u / mV,
        I_exc=STATE.I_exc / mV,
        I_inh=STATE.I_inh / mV,
    )}

if args.adapt:
    data = {**data, **dict(
        nu=nu_log,
        w_rec_exc=S_rec_exc.w[:],
        w_rec_inh=S_rec_inh.w[:],
        w_ext_exc=S_ext_exc.w[:],
        w_ext_inh=S_ext_inh.w[:],
        scale_rec_exc=S_rec_exc.scale_i,
        scale_rec_inh=S_rec_inh.scale_i,
        scale_ext_exc=S_ext_exc.scale_i,
        scale_ext_inh=S_ext_inh.scale_i,
        offset_rec_exc=S_rec_exc.offset_i,
        offset_rec_inh=S_rec_inh.offset_i,
        offset_ext_exc=S_ext_exc.offset_i,
        offset_ext_inh=S_ext_inh.offset_i
    )}
np.savez_compressed(filename, **data)

print(f'### wrote data to {filename}')

# remove compiled code again
try:
    shutil.rmtree(cache_dir, ignore_errors=True)
except Exception as e:
    print(f"Unable to remove cached files: {repr(e)}")

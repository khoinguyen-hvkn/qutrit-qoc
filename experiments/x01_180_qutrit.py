import os
import sys
import warnings
warnings.filterwarnings('ignore')

path = sys.path[0]
parent_dir = os.path.abspath(os.path.join(path, os.pardir))

for root, dirs, files in os.walk(parent_dir):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))

sys.path.append(parent_dir)
sys.path.pop(0)
# print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import inspect

from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from utils import *

data_path = '../output/x01_π/'

# from qiskit_ibm_provider import IBMProvider

# IBMProvider.save_account('030e0f3b6562aeef01f326920c2b5c95160de76135f801f67cce23704c77a274aa5241b45c7777d8a2bcfb4c71803c8796f4d6896241914aba81fc7281bae508', overwrite=True)

# provider = IBMProvider()
# backend = provider.get_backend('ibm_lagos')

# config = backend.configuration()
# dt = config.dt
# acquire_alignment = config.timing_constraints['acquire_alignment']
# granularity = config.timing_constraints['granularity']
# pulse_alignment = config.timing_constraints['pulse_alignment']
# lcm = np.lcm(acquire_alignment, pulse_alignment)

# def get_closest_multiple_of(value, base_number):
#     return int(value + base_number/2) - (int(value + base_number/2) % base_number)

# def get_closest_multiple_of_16(num):
#     return get_closest_multiple_of(num, granularity)


ns = 1.0e9
dt = 2/9*1e-9


ns = 1.0e9
dt = 2/9/ns
f01 = 5235359100.418953
anhar = -339867138.55915606

#Defining time scales

steps = 144 #1-2: 160
# steps = get_closest_multiple_of_16(steps)
total_time = steps * dt * ns #ns dt = 0.2222222222

# Choose optimizing State transfer or Unitary gate
state_transfer = True

#Defining H0
qubit_state_num = 3 #change
qubit_num = 1 
freq_ge = f01/ns #GHz #change
alpha = anhar/ns
g_ops = 0.073036776 #GHz #change

# ens = np.array([ 2*np.pi*ii*(freq_ge - 0.5*(ii-1)*alpha) for ii in np.arange(qubit_state_num)])
ens = np.array([0, 0, alpha])
Q_x   = np.diag(np.sqrt(np.arange(1,qubit_state_num)),1)+np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1)
Q_y   = (0+1j) *(np.diag(np.sqrt(np.arange(1,qubit_state_num)),1)-np.diag(np.sqrt(np.arange(1,qubit_state_num)),-1))
Q_z   = np.diag(np.arange(0,qubit_state_num))
Q_I   = np.identity(qubit_state_num)
H_q = np.diag(ens)

H0 = H_q

#Defining Concerned states (starting states)

psi0 = [0, 1, 2]

#Defining states to include in the drawing of occupation
states_draw_list = [0, 1, 2]
states_draw_names = ['0','1','2']

#Defining U (Target)
U = np.array([ 
    [np.sqrt(2)/2, -1j*np.sqrt(2)/2, 0],
    [-1j*np.sqrt(2)/2, np.sqrt(2)/2, 0],
    [0, 0, 1]
])

#Defining U0 (Initial)
q_identity = np.identity(qubit_state_num)
U0 = q_identity

#Defining control Hs
XI = Q_x
YI = Q_y
ZI = Q_z
Hops = [XI, YI]
g_ops = 0.073036776 
ops_max_amp = [2 * np.pi * g_ops, 2 * np.pi * g_ops] # = Omega_{d, 0}
Hnames = ['x', 'y']

print(Hnames)

#Defining convergence parameters
max_iterations = 1000
decay = max_iterations/2
convergence = {'rate': 0.1, 'update_step': 1, 'max_iterations': max_iterations, 'conv_target': 1e-4, 'learning_rate_decay': decay}

# guassian envelope pulse
reg_coeffs = {'envelope': 0.1, 'dwdt': 0.001}

uks, U_final = Grape(H0, Hops, Hnames, U, total_time, steps, psi0,
                    convergence = convergence, 
                    draw = [states_draw_list, states_draw_names],
                    state_transfer = False,
                    use_gpu = True,
                    sparse_H = False,
                    show_plots = True, 
                    unitary_error = 1e-6, 
                    method = 'L-BFGS-B', 
                    maxA = ops_max_amp,
                    Taylor_terms = [20,0],
                    reg_coeffs = reg_coeffs,
                    save = True,
                    file_name = 'x01_lagos',
                    data_path = data_path)

plt.plot(uks)
plt.savefig('./figs/x01_180_qutrit_e4')

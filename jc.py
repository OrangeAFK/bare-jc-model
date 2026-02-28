import numpy as np
from scipy.linalg import eigh # diag

# units are gigarads/s
omega_r = 5.0 * 2*np.pi # resonator freq (gigarads/s)
omega_q = 6.0 * 2*np.pi # qubit freq (gigarads/s)
g = 0.1 * 2*np.pi

N = 5 # photon states in resonator --> truncated Fock space 

# resonator annihilation operator
a = np.zeros((N,N))
for n in range(1,N):
    a[n-1,n] = np.sqrt(n)
a_dag = a.T

sigma_z = np.array([[1, 0], [0, -1]]) # pauli z
sigma_plus = np.array([[0, 1], [0, 0]]) # raising operator
sigma_minus = np.array([[0, 0], [1, 0]]) # lowering operator

# resonator and qubit identity matrices
I_r = np.eye(N) 
I_q = np.eye(2)

# resonator and qubit operators in full Hilbert space

a_full = np.kron(a, I_q)
a_dag_full = np.kron(a_dag, I_q)

sigma_z_full = np.kron(I_r, sigma_z)
sigma_plus_full = np.kron(I_r, sigma_plus)
sigma_minus_full = np.kron(I_r, sigma_minus)

# JC Hamiltonian
# we factor and cancel out h-bar because the relative magnitude matters instead of absolute
H = (omega_r * a_dag_full @ a_full + 
    0.5 * omega_q * sigma_z_full + 
    g * (a_dag_full @ sigma_minus_full + a_full @ sigma_plus_full))

eigvals, eigvecs = eigh(H)

print("Eigenvalues (GHz): ")
print(eigvals / (2*np.pi)) # gigarad to gigahz

'''
[-3.          1.99009805  3.00990195  6.98038476  8.01961524 11.97084974
 13.02915026 16.96148352 18.03851648 23.        ]

 -3 is the ground state (-omega_q / 2).

 The next two are 1-excitation manifolds (0<->1 photons, |1,g> +- |0,e> ) with .01 GHz vacuum Rabi splitting
 The next two are the 2-excitation manifolds, then three, then four, and then the truncated 5-excitation state
 
 Note the increasing sqrt(n) vacuum Rabi splitting

'''
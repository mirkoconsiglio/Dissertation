import numpy as np
from qiskit.quantum_info import partial_trace

def theoretical_tangle(phi):  # Theoretical Three-tangle function
    d1 = phi[0] ** 2 * phi[7] ** 2 + phi[1] ** 2 * phi[6] ** 2 + \
         phi[2] ** 2 * phi[5] ** 2 + phi[4] ** 2 * phi[3] ** 2
    d2 = phi[0] * phi[7] * phi[3] * phi[4] + phi[0] * phi[7] * phi[5] * phi[2] + \
         phi[0] * phi[7] * phi[6] * phi[1] + phi[3] * phi[4] * phi[5] * phi[2] + \
         phi[3] * phi[4] * phi[6] * phi[1] + phi[5] * phi[2] * phi[6] * phi[1]
    d3 = phi[0] * phi[6] * phi[5] * phi[3] + phi[7] * phi[1] * phi[2] * phi[4]
    tau = 4 * np.abs(d1 - 2 * d2 + 4 * d3)

    return tau

def concurrence(rho): # Concurrence Function
    Y = np.array([[0, -1j],
                  [1j, 0]])
    spin_flip = np.kron(Y, Y)
    rho_tilde = np.matmul(np.matmul(spin_flip, np.matrix.conjugate(rho)), spin_flip)
    l = np.sort(np.nan_to_num(np.sqrt(np.real(np.linalg.eigvals(np.matmul(rho, rho_tilde))))))[::-1]

    return np.max([0, l[0] - l[1] - l[2] - l[3]])

psi = [0, 1, 1, 0, 1, 0, 0, 2]
psi /= np.linalg.norm(psi)
rho = np.outer(psi, psi)
rho = partial_trace(rho, [2]).data
print(concurrence(rho)**2)
print(theoretical_tangle(psi))
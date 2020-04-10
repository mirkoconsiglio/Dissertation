import itertools
import os
import pickle
import csv
import time
import numpy as np
import qiskit.visualization as visual
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.ignis.verification.tomography import StateTomographyFitter, state_tomography_circuits
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import thermal_relaxation_error
from qiskit.quantum_info.states.measures import state_fidelity
#from qiskit.quantum_info import partial_trace
from qiskit.tools.qi.qi import partial_trace
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.special import comb

# IBMQ.save_account('584e07bcdb11290b401e6a79ba00887070738a17576998b2535b42ed9c52d49413ac8b5ce3febf58165a2e2ce41dda1529bca8fe091c8dc9629b2e1e62b27a03')

# np.random.seed(0)
# random.seed(0)

# rcParams.update({'figure.max_open_warning': 0})

I = np.array([[1, 0],
              [0, 1]])
X = np.array([[0, 1],
              [1, 0]])
Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0, -1]])
spin_flip = np.kron(Y, Y)


class Circuit(object):

    def __init__(self, state=('GHZ', 3)):

        if state == 'h':
            self.state = 'h'
            self.n = 1
        elif state == 'GHZ_teleport':
            self.state = 'GHZ_teleport'
            self.n = 4
        else:
            self.state = state[0]
            self.n = state[1]
            if len(state) == 3:
                self.k = state[2]
            else:
                self.k = 1

        self.q = QuantumRegister(self.n, 'q')
        self.qc = QuantumCircuit(self.q)

    def collisional_model(self,
                          backend='qasm_simulator',
                          live=False,
                          noise_model=None,
                          channel='amplitude_damping',
                          shots=1024,
                          measured_qubits=(),
                          max_collisions=5,
                          theta=np.pi/2,
                          concurrence=False,
                          tangle=False,
                          witness=False,
                          tangle_witness=True,
                          markov=True,
                          print_results=True,
                          save_results=False,
                          directory=None,
                          full=True,
                          initial_statevector=np.array([0.5, 0.5])):

        self.channel = channel
        self.markov = markov

        if self.state == 'h':
            self.qc.h(0)

        elif self.state == 'GHZ_teleport':
            initial_state = initial_statevector
            initial_state /= np.linalg.norm(initial_state)
            self.qc.initialize(initial_state, [0])
            self.theoretical_rho = np.outer(initial_state, initial_state)
            self.GHZ([1, 2, 3])

        elif self.state == 'D':
            self.D()
            if full:
                self.theoretical_rho = self.full_theoretical_D(self.n, self.k)
            else:
                self.theoretical_rho = self.theoretical_D(self.n, self.k)

        elif self.state == 'W':
            self.W()
            if full:
                self.theoretical_rho = self.full_theoretical_D(self.n, self.k)
            else:
                self.theoretical_rho = self.theoretical_D(self.n, self.k)

        elif self.state == 'GHZ':
            self.GHZ(range(self.n))
            if full:
                self.theoretical_rho = self.full_theoretical_GHZ(self.n)
            else:
                self.theoretical_rho = self.theoretical_GHZ(self.n)

        self.histogram_data = []
        self.directory = directory

        if backend == 'qasm_simulator':
            self.backend = Aer.get_backend('qasm_simulator')
            self.coupling_map = None
            if noise_model:
                self.noise_model = noise_model
                self.basis_gates = self.noise_model.basis_gates
            else:
                self.noise_model = None
                self.basis_gates = None
        else:
            provider = IBMQ.get_provider(group='open')
            self.device = provider.get_backend(backend)
            self.properties = self.device.properties()
            self.coupling_map = self.device.configuration().coupling_map
            self.noise_model = noise.device.basic_device_noise_model(self.properties)
            self.basis_gates = self.noise_model.basis_gates

            if save_results:
                visual.plot_error_map(self.device).savefig('{}/error_map.png'.format(self.directory))

            if live:
                check = input("Are you sure you want to run the circuit live? [y/n]: ")
                if check == 'y' or check == 'yes':
                    self.backend = self.device
                else:
                    self.backend = Aer.get_backend('qasm_simulator')
            else:
                self.backend = Aer.get_backend('qasm_simulator')

        self.unitary_backend = Aer.get_backend('unitary_simulator')

        self.shots = shots

        if self.state == 'GHZ_teleport':
            self.a = 3
            self.b = 3
        elif measured_qubits == ():
            self.a = 0
            self.b = self.n - 1
        else:
            self.a = measured_qubits[0]
            self.b = measured_qubits[1]

        self.max_collisions = max_collisions
        self.theta = theta

        if self.max_collisions > 0:
            if self.channel == 'phase_damping_one_qubit':
                self.ancilla = QuantumRegister(1, 'a')
                self.qc.add_register(self.ancilla)
            elif not markov:
                self.ancilla = QuantumRegister(3, 'a')
                self.qc.add_register(self.ancilla)
                self.qc.h(self.ancilla[2])
                self.qc.cx(self.ancilla[2], self.ancilla[1])
                self.qc.cx(self.ancilla[1], self.ancilla[0])
            else:
                self.ancilla = QuantumRegister(self.max_collisions, 'a')
                self.qc.add_register(self.ancilla)

        if self.channel == 'amplitude_damping':
            self.U = self.amplitude_damping_operator(full)
        elif self.channel == 'phase_damping' or 'phase_damping_one_qubit':
            self.U = self.phase_damping_operator(full)
        elif self.channel == 'heisenberg':
            print('Not yet programmed')
            quit()
            self.U = self.heisenberg_operator()

        if tangle_witness:
            self.tangle_witness_GHZ = 3 / 4 * np.kron(np.kron(I, I), I) - self.full_theoretical_GHZ(3)
            self.tangle_witness_tri = 1 / 2 * np.kron(np.kron(I, I), I) - self.full_theoretical_GHZ(3)

        counts_list = []
        rho_list = []
        theoretical_rho_list = []
        concurrence_list = []
        theoretical_concurrence_list = []
        tangle_ub_list = []
        tangle_lb_list = []
        theoretical_tangle_list = []
        fidelity_list = []
        witness_list = []
        tangle_witness_GHZ_list = []
        tangle_witness_tri_list = []
        trace_squared_list = []
        for k in range(self.max_collisions + 1):
            if k > 0:
                self.collision(k)
                if witness:
                    counts = self.measure(witness)
                else:
                    rho = self.tomography(full)
                    self.evolve_theoretical_rho(full)
            else:
                if witness:
                    counts = 1 / 2
                else:
                    rho = self.tomography(full)

            if save_results:
                visual.plot_state_city(rho).savefig('{}/state_city_{}.png'.format(self.directory, k))
                visual.plot_state_paulivec(rho).savefig('{}/paulivec_{}.png'.format(self.directory, k))

            if witness:
                counts_list.append(counts)
            else:
                rho_list.append(rho)
                theoretical_rho_list.append(self.theoretical_rho)
                if not full and self.state != 'GHZ_teleport':
                    C = self.concurrence(rho)
                    concurrence_list.append(C)
                    TC = self.concurrence(self.theoretical_rho)
                    theoretical_concurrence_list.append(TC)
                if tangle:
                    T_ub = self.tangle_ub(rho)
                    tangle_ub_list.append(T_ub)
                    #T_lb = self.tangle_lb(rho)
                    #tangle_lb_list.append(T_lb)
                    if self.state == 'GHZ':
                        TT = np.exp(np.log(np.cos(self.theta)) * k)
                        theoretical_tangle_list.append(TT)
                F = state_fidelity(theoretical_rho_list[0], rho)
                fidelity_list.append(F)
                if witness:
                    W = np.real(np.trace(np.matmul(self.full_theoretical_GHZ(self.n), rho)))
                    witness_list.append(W)
                if tangle_witness and full:
                    TWGHZ = np.real(np.trace(np.matmul(self.tangle_witness_GHZ, rho)))
                    tangle_witness_GHZ_list.append(TWGHZ)
                    TWtri = np.real(np.trace(np.matmul(self.tangle_witness_tri, rho)))
                    tangle_witness_tri_list.append(TWtri)
                Tr2 = np.real(np.trace(np.matmul(rho, rho)))
                trace_squared_list.append(Tr2)

                if print_results:
                    print("Collision Number:", k)
                    print("Original Density Matrix:")
                    print(theoretical_rho_list[0])
                    print("Theoretical Density Matrix:")
                    print(self.theoretical_rho)
                    print("Measured Density Matrix:")
                    print(rho)
                    print("Trace:", np.real(np.trace(rho)))
                    print("Trace Squared:", Tr2)
                    if concurrence:
                        print("Concurrence:", C)
                        print("Theoretical Concurrence:", TC)
                    if tangle:
                        print("Tangle Upper Bound:", T_ub)
                        #print("Tangle Lower Bound:", T_lb)
                        print("Theoretical Tangle:", TT)
                    print("Fidelity:", F)
                    if witness:
                        print("Witness:", W)
                    if tangle_witness and full:
                        print("GHZ Tangle Witness:", TWGHZ)
                        print("Tripartite Tangle Witness:", TWtri)
                    print("Eigenvalues:", np.sort(np.real(np.linalg.eigvals(rho)))[::-1])
                    print("\n-----------------------------------------------------------------------------------\n")

        if print_results:
            if save_results:
                visual.circuit_drawer(self.qc, filename='{}/constructed_circuit.png'.format(self.directory), output='mpl')
            else:
                print(self.qc)
            print("Constructed Circuit Depth: ", self.qc.depth())
            try:
                transpiled_qc = transpile(self.qc, self.device, optimization_level=3)
                if save_results:
                    visual.circuit_drawer(transpiled_qc, filename='{}/transpiled_circuit.png'.format(self.directory), output='mpl')
                print("Transpiled Circuit Depth: ", self.qc.depth())
            except:
                pass
            print()

        if save_results:
            visual.plot_histogram(self.histogram_data, title=self.state).savefig('{}/histogram.png'.format(self.directory))

        return counts_list, theoretical_rho_list, rho_list, theoretical_concurrence_list, concurrence_list, \
               theoretical_tangle_list, tangle_ub_list, tangle_lb_list, fidelity_list, witness_list, \
               tangle_witness_GHZ_list, tangle_witness_tri_list, trace_squared_list

    def collision(self, k):
        if self.channel == 'amplitude_damping':
            self.amplitude_damping(k)
        elif self.channel == 'phase_damping':
            self.phase_damping(k, self.markov)
        elif self.channel == 'phase_damping_one_qubit':
            self.phase_damping_one_qubit(k)
        elif self.channel == 'heisenberg':
            self.heisenberg(k)

    def amplitude_damping(self, k):
        self.qc.cu3(self.theta, 0, 0, self.q[self.n - 1], self.ancilla[k - 1])
        self.qc.cx(self.ancilla[k - 1], self.q[self.n - 1])

    def phase_damping(self, k, markov):
        if markov:
            self.qc.h(self.ancilla[k - 1])
            self.qc.cx(self.ancilla[k - 1], self.q[self.n - 1])
            self.qc.rz(self.theta, self.q[self.n - 1])
            self.qc.cx(self.ancilla[k - 1], self.q[self.n - 1])
        else:
            self.qc.cx(self.ancilla[k % 2], self.q[self.n - 1])
            self.qc.rz(self.theta, self.q[self.n - 1])
            self.qc.cx(self.ancilla[k % 2], self.q[self.n - 1])

    def phase_damping_one_qubit(self, k):
        if k == 1:
            self.qc.h(self.ancilla[0])
        else:
            self.qc.reset(self.ancilla[0])
            self.qc.h(self.ancilla[0])
        self.qc.cx(self.ancilla[0], self.q[self.n - 1])
        self.qc.rz(self.theta, self.q[self.n - 1])
        self.qc.cx(self.ancilla[0], self.q[self.n - 1])

    def heisenberg(self, k):
        # self.qc.h(self.ancilla[k - 1])
        self.qc.cx(self.q[self.n - 1], self.ancilla[k - 1])
        self.qc.cu3(self.theta, 0, self.theta, self.ancilla[k - 1], self.q[self.n - 1])
        # self.qc.cx(self.q[self.n - 1], self.ancilla[k - 1])

    def D(self):
        def SCS(n, k):
            for i in range(n - 1, n - k - 1, -1):
                if i == n - 1:
                    if self.k != 1 and self.k != self.n - 1:
                        self.qc.cx(self.q[i - 1], self.q[n - 1])
                    self.qc.cu3(2 * np.arccos(np.sqrt((n - i) / n)), 0, 0, self.q[n - 1], self.q[i - 1])
                    if self.k == 1 or k == 1:
                        self.qc.cx(self.q[i - 1], self.q[n - 1])
                    else:
                        self.qc.cx(self.q[i - 1], self.q[i - 2])
                elif i > 0:
                    self.qc.cx(self.q[i - 1], self.q[n - 1])

                    if self.k == 1:
                        self.qc.cx(self.q[i], self.q[i - 1])
                    self.qc.ry(-np.arccos(np.sqrt((n - i) / n)) / 2, self.q[i - 1])
                    self.qc.cx(self.q[n - 1], self.q[i - 1])
                    self.qc.ry(np.arccos(np.sqrt((n - i) / n)) / 2, self.q[i - 1])
                    self.qc.cx(self.q[i], self.q[i - 1])
                    self.qc.ry(-np.arccos(np.sqrt((n - i) / n)) / 2, self.q[i - 1])
                    self.qc.cx(self.q[n - 1], self.q[i - 1])
                    self.qc.ry(np.arccos(np.sqrt((n - i) / n)) / 2, self.q[i - 1])

                    self.qc.cx(self.q[i - 1], self.q[n - 1])
                self.qc.barrier()

        if self.n < 2:
            raise ValueError("Number of state qubits must be greater than 1")
        elif self.k < 0:
            raise ValueError("Number of flipped qubits must not be negative")
        elif self.k > self.n:
            raise ValueError("Number of flipped qubits must not be greater than", self.n)
        elif self.k == 0:
            pass
        elif self.k == self.n:
            self.qc.x(self.q)
        else:
            n = self.n
            k = self.k

            flip = False
            if k > n / 2:
                k = n - k
                flip = True

            for i in range(n - k, n):
                self.qc.x(self.q[i])

            while n > k:
                SCS(n, k)
                if n == 2 and k == 1:
                    break
                if n == k + 1:
                    k -= 1
                n -= 1

            if flip == True:
                self.qc.x(self.q)

    def W(self):
        if self.n < 2:
            raise ValueError("Number of state qubits must be greater than 1")
        else:
            self.k = 1
            self.D()

    def GHZ(self, q):

        if len(q) < 2:
            raise ValueError("Number of state qubits must be greater than 1")
        self.qc.h(q[0])
        for i in q:
            if i > q[0]:
                i -= q[0]
                self.qc.cx(q[0], q[i])

    @staticmethod
    def theoretical_D(n, k):
        m = n / 2 - k
        v1 = (n + 2 * m) * (n - 2 + 2 * m) / (4 * n * (n - 1))
        v2 = (n - 2 * m) * (n - 2 - 2 * m) / (4 * n * (n - 1))
        w = (n ** 2 - 4 * m ** 2) / (4 * n * (n - 1))
        return np.array([[v1, 0, 0, 0],
                         [0, w, w, 0],
                         [0, w, w, 0],
                         [0, 0, 0, v2]], dtype=complex)

    @staticmethod
    def theoretical_GHZ(n):
        if n == 2:
            return np.array([[1 / 2, 0, 0, 1 / 2],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [1 / 2, 0, 0, 1 / 2]], dtype=complex)
        else:
            return np.array([[1 / 2, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1 / 2]], dtype=complex)

    @staticmethod
    def full_theoretical_D(n, k):
        nck = []
        for bits in itertools.combinations(range(n), k):
            s = ['0'] * n
            for bit in bits:
                s[bit] = '1'
            nck.append(''.join(s))

        f = 1 / np.sqrt(comb(n, k, exact=True))
        psi = np.zeros(2 ** n)
        for i in nck:
            psi[int(i, 2)] = f

        rho = np.outer(psi, psi)

        return rho

    @staticmethod
    def full_theoretical_GHZ(n):
        psi = np.zeros(2 ** n)
        psi[0] = 1 / np.sqrt(2)
        psi[2 ** n - 1] = 1 / np.sqrt(2)

        rho = np.outer(psi, psi)

        return rho

    def amplitude_damping_operator(self, full):
        if full:
            q = QuantumCircuit(self.n + 1)
            q.cu3(self.theta, 0, 0, self.n - 1, self.n)
            q.cx(self.n, self.n - 1)
        else:
            q = QuantumCircuit(3)
            q.cu3(self.theta, 0, 0, 1, 2)
            q.cx(2, 1)

        job = execute(q, self.unitary_backend)
        result = job.result()

        return result.get_unitary(q)

    def phase_damping_operator(self, full):
        if self.state == 'GHZ_teleport':
            q = QuantumCircuit(2)
            q.h(1)
            q.cx(1, 0)
            q.rz(self.theta, 0)
            q.cx(1, 0)
        elif full:
            q = QuantumCircuit(self.n + 1)
            q.h(self.n)
            q.cx(self.n, self.n - 1)
            q.rz(self.theta, self.n - 1)
            q.cx(self.n, self.n - 1)
        else:
            q = QuantumCircuit(3)
            q.h(2)
            q.cx(2, 1)
            q.rz(self.theta, 1)
            q.cx(2, 1)

        job = execute(q, self.unitary_backend)
        result = job.result()

        U = result.get_unitary()

        return U

    def heisenberg_operator(self):
        q = QuantumCircuit(3)
        # q.h(2)
        q.cx(1, 2)
        q.cu3(self.theta, 0, self.theta, 2, 1)
        # q.cx(1, 2)

        job = execute(q, self.unitary_backend)
        result = job.result()

        return result.get_unitary(q)

    def evolve_theoretical_rho(self, full):
        if full:
            evolving_rho = np.kron(np.array([[1, 0], [0, 0]]), self.theoretical_rho)
            evolved_rho = np.matmul(np.matmul(self.U, evolving_rho), self.U.conj().T)
            self.theoretical_rho = partial_trace(evolved_rho, self.n)
        else:
            evolving_rho = np.kron(np.array([[1, 0], [0, 0]]), self.theoretical_rho)
            evolved_rho = np.matmul(np.matmul(self.U, evolving_rho), self.U.conj().T)

            if self.state == 'GHZ_teleport':
                self.theoretical_rho = partial_trace(evolved_rho, 1)
            else:
                self.theoretical_rho = partial_trace(evolved_rho, 2)

    def tomography(self, full):
        if self.state == 'GHZ_teleport':
            self.qc.barrier()

            self.qc.cx(0, 1)
            self.qc.h(0)
            self.qc.h(2)

            c0 = ClassicalRegister(1)
            c1 = ClassicalRegister(1)
            c2 = ClassicalRegister(1)

            self.qc.add_register(c0, c1, c2)

            self.qc.measure(0, c0)
            self.qc.measure(1, c1)
            self.qc.measure(2, c2)

            self.qc.z(3).c_if(c0, 1)
            self.qc.x(3).c_if(c1, 1)
            self.qc.z(3).c_if(c2, 1)

            tomography_circuits = state_tomography_circuits(self.qc, self.q[self.a])
        else:
            tomography_circuits = state_tomography_circuits(self.qc, self.q)

        job = execute(tomography_circuits,
                      backend=self.backend,
                      coupling_map=self.coupling_map,
                      noise_model=self.noise_model,
                      basis_gates=self.basis_gates,
                      shots=self.shots,
                      optimization_level=3)
        tomography_results = job.result()

        rho = StateTomographyFitter(tomography_results, tomography_circuits).fit()

        if self.state == 'GHZ_teleport':
            self.histogram_data.append(tomography_results.get_counts(tomography_circuits[2]))
        else:
            self.histogram_data.append(tomography_results.get_counts(tomography_circuits[3 ** self.n - 1]))

            if not full:
                rho = partial_trace(rho, np.delete(np.arange(self.n), [self.a, self.b]))

        if self.state == 'GHZ_teleport':
            for i in range(3):
                self.qc.data.pop(-1)
            self.qc.remove_final_measurements()
            for i in range(3):
                self.qc.data.pop(-1)

        return rho

    def measure(self, witness):
        if witness == 'x':
            self.qc.h(self.q[self.b])
        elif witness == 'y':
            self.qc.sdg(self.q[self.b])
            self.qc.h(self.q[self.b])
        self.qc.add_register(ClassicalRegister(1))
        self.qc.measure(self.b, 0)

        job = execute(self.qc,
                      backend=self.backend,
                      coupling_map=self.coupling_map,
                      noise_model=self.noise_model,
                      basis_gates=self.basis_gates,
                      shots=self.shots,
                      optimization_level=3)
        results = job.result()
        counts = results.get_counts()

        counts_0 = counts.get('0')
        counts_1 = counts.get('1')

        if counts_0 and counts_1:
            counts = (counts_0 - counts_1) / (2 * self.shots)
        elif counts_0:
            counts = 1 / 2
        else:
            counts = -1 / 2

        self.qc.remove_final_measurements()
        if witness == 'x':
            self.qc.data.pop()
        if witness == 'y':
            self.qc.data.pop()
            self.qc.data.pop()

        return counts

    @staticmethod
    def concurrence(rho):
        rho_tilde = np.matmul(np.matmul(spin_flip, np.matrix.conjugate(rho)), spin_flip)
        l = np.sort(np.real(np.sqrt(np.linalg.eigvals(np.matmul(rho, rho_tilde)))))[::-1]
        C = np.max([0, l[0] - l[1] - l[2] - l[3]])

        return C

    @staticmethod
    def tangle_ub(rho):
        N = 1
        error = 1e-30
        print('error: ', error)

        Nc = 8
        Np = 4

        def real_to_complex(z):  # real vector of length 2n -> complex of length n
            return z[:len(z) // 2] + 1j * z[len(z) // 2:]

        def complex_to_real(z):  # complex vector of length n -> real of length 2n
            return np.concatenate((np.real(z), np.imag(z)))

        def theoretical_tangle(phi):
            d1 = phi[0] ** 2 * phi[7] ** 2 + phi[1] ** 2 * phi[6] ** 2 + phi[2] ** 2 * phi[5] ** 2 + phi[4] ** 2 * phi[3] ** 2
            d2 = phi[0] * phi[7] * phi[3] * phi[4] + phi[0] * phi[7] * phi[5] * phi[2] + \
                 phi[0] * phi[7] * phi[6] * phi[1] + phi[3] * phi[4] * phi[5] * phi[2] + \
                 phi[3] * phi[4] * phi[6] * phi[1] + phi[5] * phi[2] * phi[6] * phi[1]
            d3 = phi[0] * phi[6] * phi[5] * phi[3] + phi[7] * phi[1] * phi[2] * phi[4]
            tau = 4 * np.abs(d1 - 2 * d2 + 4 * d3)

            return tau

        def objective(x, Nc, Np):
            p = x[:Np]
            result = 0
            for i in range(Np):
                j = Np + i * 2 * Nc
                c = real_to_complex(x[j:j + 2 * Nc])

                result += p[i] * theoretical_tangle(c)

            return result

        constraints = []

        def cons1(x):
            return 1 - np.sum(x[:Np])

        constraints.append(NonlinearConstraint(cons1, -error, error))

        def cons2(x):
            result = np.zeros(Np)
            for i in range(Np):
                j = Np + i * 2 * Nc
                d = real_to_complex(x[j:j + 2 * Nc])
                result[i] = 1 - np.linalg.norm(d) ** 2

            return result

        lower_constraints = []
        upper_constraints = []
        for j in range(Np):
            lower_constraints.append(-error)
            upper_constraints.append(error)

        constraints.append(NonlinearConstraint(cons2, lower_constraints, upper_constraints))

        def cons3(x):
            p = x[:Np]
            c = np.zeros((Nc, Nc), dtype=complex)
            for i in range(Np):
                j = Np + i * 2 * Nc
                c[i] = real_to_complex(x[j:j + 2 * Nc])

            reals = np.zeros(Nc ** 2)
            imags = np.zeros(Nc ** 2)
            for i in range(Nc):
                for j in range(Nc):
                    res = 0
                    for k in range(Np):
                        res += p[k] * c[k, i] * np.conj(c[k, j])
                    res -= rho[i, j]

                    reals[i * Nc + j] = np.real(res)
                    imags[i * Nc + j] = np.imag(res)

            result = np.concatenate((reals, imags))

            return result

        lower_constraints = []
        upper_constraints = []
        for j in range(2 * Nc ** 2):
            lower_constraints.append(-error)
            upper_constraints.append(error)

        constraints.append(NonlinearConstraint(cons3, lower_constraints, upper_constraints))

        lower_bounds = []
        upper_bounds = []
        for j in range(Np + 2 * Nc * Np):
            if j < Np:
                lower_bounds.append(0)
                upper_bounds.append(1)
            else:
                lower_bounds.append(-1)
                upper_bounds.append(1)
        bounds = Bounds(lower_bounds, upper_bounds)

        tangles = np.zeros(N)
        for j in range(N):
            p = np.random.random(size=Np)
            p /= np.sum(p)
            x0 = np.array(p)
            for k in range(Np):
                a = np.random.random(size=Nc) * 2 - 1
                b = np.random.random(size=Nc) * 2 - 1
                c = a + 1j * b
                c /= np.linalg.norm(c)
                c = complex_to_real(c)
                x0 = np.append(x0, c)

            tangles[j] = minimize(objective, x0, args=(Nc, Np), method='trust-constr', constraints=constraints,
                                  bounds=bounds,
                                  options={'gtol': 1e-8,
                                           'xtol': 1e-8,
                                           'barrier_tol': 1e-8,
                                           'maxiter': 100,
                                           'sparse_jacobian': False,
                                           'factorization_method': 'SVDFactorization',
                                           'initial_tr_radius': 1,
                                           'initial_constr_penalty': 1,
                                           'initial_barrier_parameter': 1e-8,
                                           'initial_barrier_tolerance': 1,
                                           'verbose': 2})['fun']

            if tangles[j] < 0:
                tangles[j] = np.inf

        return np.min(tangles)

    @staticmethod
    def tangle_lb(rho):
        eigs = np.sort(np.real(np.linalg.eigvals(rho)))[::-1]
        rank = sum(i > 1e-16 for i in eigs)

        return eng.tangle(matlab.double(np.real(rho).tolist()), matlab.double(np.imag(rho).tolist()), int(rank))


def noise_model(n):
    noise = NoiseModel()

    # T1 and T2 values for qubits 0-3
    T1s = np.random.normal(50e3, 10e3, n)  # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 10e3, n)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(n)])

    # Instruction times (in nanoseconds)
    time_u1 = 0  # virtual gate
    time_u2 = 50  # (single X90 pulse)
    time_u3 = 100  # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000  # 1 microsecond

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)]
    errors_u1 = [thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)]
    errors_u2 = [thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)]
    errors_u3 = [thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
        thermal_relaxation_error(t1b, t2b, time_cx))
        for t1a, t2a in zip(T1s, T2s)]
        for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    for j in range(n):
        noise.add_quantum_error(errors_reset[j], "reset", [j])
        noise.add_quantum_error(errors_measure[j], "measure", [j])
        noise.add_quantum_error(errors_u1[j], "u1", [j])
        noise.add_quantum_error(errors_u2[j], "u2", [j])
        noise.add_quantum_error(errors_u3[j], "u3", [j])
        for k in range(n):
            noise.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    return noise


def plot_results(theoretical_concurrence_list, concurrence_list, theoretical_tangle_list,
                 tangle_ub_list, tangle_lb_list, fidelity_list, witness_list, tangle_witness_GHZ_list,
                 tangle_witness_tri_list, trace_squared_list, max_collisions, directory):
    fig, ax = subplots()
    ax.axhline(0, c='k', alpha=0.5, ls='--')
    ax.axvline(0, c='k', alpha=0.5, ls='--')
    if concurrence:
        ax.plot(range(max_collisions + 1), theoretical_concurrence_list, label='Theoretical Concurrence', marker='o')
        ax.plot(range(max_collisions + 1), concurrence_list, label='Concurrence', marker='o')
    if tangle:
        ax.plot(range(max_collisions + 1), theoretical_tangle_list, label='Theoretical Tangle', marker='o')
        ax.plot(range(max_collisions + 1), tangle_ub_list, label='Numerical Tangle', marker='o')
        #ax.plot(range(max_collisions + 1), tangle_lb_list, label='Tangle Lower Bound', marker='o')
    ax.plot(range(max_collisions + 1), fidelity_list, label='Fidelity', marker='o')
    if witness:
        ax.plot(range(max_collisions + 1), witness_list, label='Witness', marker='o')
    if tangle_witness and full:
        ax.plot(range(max_collisions + 1), tangle_witness_GHZ_list, label='GHZ Tangle Witness', marker='o')
        ax.plot(range(max_collisions + 1), tangle_witness_tri_list, label='Tripartite Tangle Witness', marker='o')
    # ax.plot(range(max_collisions + 1), trace_squared_list, label='Trace Squared')
    ax.legend()
    if save_results:
        fig.savefig('{}/plot.png'.format(directory))


def save_data(theoretical_rho_list, rho_list, theoretical_concurrence_list, concurrence_list,
              theoretical_tangle_list, tangle_ub_list, tangle_lb_list, fidelity_list, witness_list,
              tangle_witness_GHZ_list, tangle_witness_tri_list, trace_squared_list, max_collisions, directory):
    print('Saving data - do not quit.\n')

    data = {'theoretical_rho_list': theoretical_rho_list,
            'simulated_rho_list': rho_list,
            'theoretical_concurrence_list': theoretical_concurrence_list,
            'concurrence_list': concurrence_list,
            'theoretical_tangle_list': theoretical_tangle_list,
            'tangle_ub_list': tangle_ub_list,
            'tangle_lb_list': tangle_lb_list,
            'fidelity_list': fidelity_list,
            'witness_list': witness_list,
            'tangle_witness_GHZ_list': tangle_witness_GHZ_list,
            'tangle_witness_tri_list': tangle_witness_tri_list,
            'trace_squared_list': trace_squared_list,
            'max_collisions': max_collisions
            }

    f = open('{}/data.json'.format(directory), 'wb')
    pickle.dump(data, f, protocol=0)
    f.close()

    print('Data saved.\n')

if __name__ == '__main__':

    backends = ['qasm_simulator', 'ibmq_16_melbourne', 'ibmq_london', 'ibmq_burlington', 'ibmq_essex', 'ibmq_ourense', 'ibmq_vigo', 'ibmq_5_yorktown']

    channel = 'phase_damping'
    backend = 'qasm_simulator'

    if backend != 'qasm_simulator':
        IBMQ.load_account()

    state = 'GHZ_teleport'

    circuit = Circuit(state=state)

    shots = 1024
    measured_qubits = ()
    max_collisions = 3
    theta = np.pi / 4
    concurrence = False
    tangle = False
    witness = None
    tangle_witness = False
    rand = np.random.random()
    initial_statevector = np.array([0.5, 0.5])
    markov = True
    noise_model = None  # noise_model(state[1] + max_collisions)
    full = False
    print_results = True
    save_results = True
    live = False  # DO NOT TURN ON

    state_label = ''
    for j in state:
        state_label += str(j)

    #if tangle:
    #    import matlab.engine
    #    eng = matlab.engine.start_matlab()
    #    eng.addpath(eng.genpath(r'D:\Google Drive\University\Physics\Fourth Year\Physics Project\CoRoNa'))
    #    eng.addpath(eng.genpath(r'C:\Users\mirko\Google Drive (mirko.consiglio.16@um.edu.mt)\University\Physics\Fourth Year\Physics Project\CoRoNA'))

    if save_results:
        directory = '{}/{}/{}'.format(channel, backend, state_label)

        try:
            os.mkdir('{}'.format(channel))
        except FileExistsError:
            pass
        try:
            os.mkdir('{}/{}'.format(channel, backend))
        except FileExistsError:
            pass
        try:
            os.mkdir('{}/{}/{}'.format(channel, backend, state_label))
        except FileExistsError:
            pass
    else:
        directory = None

    print('Collisional model for state {} on {}:'.format(state_label, backend))
    print('Channel: {}, max number of collisions: {}\n'.format(channel, max_collisions))

    t = time.perf_counter()

    counts_list, theoretical_rho_list, rho_list, theoretical_concurrence_list, concurrence_list, \
    theoretical_tangle_list, tangle_ub_list, tangle_lb_list, fidelity_list, witness_list, \
    tangle_witness_GHZ_list, tangle_witness_tri_list, trace_squared_list = \
        circuit.collisional_model(backend=backend,
                                  live=live,
                                  noise_model=noise_model,
                                  channel=channel,
                                  shots=shots,
                                  measured_qubits=measured_qubits,
                                  max_collisions=max_collisions,
                                  concurrence=concurrence,
                                  tangle=tangle,
                                  witness=witness,
                                  tangle_witness=tangle_witness,
                                  markov=markov,
                                  theta=theta,
                                  print_results=print_results,
                                  save_results=save_results,
                                  directory=directory,
                                  full=full,
                                  initial_statevector=initial_statevector)

    elapsed_time = time.perf_counter() - t
    print('Total time elapsed for simulation: {:.2f}\n'.format(elapsed_time))

    if max_collisions > 0:
        if witness:
            fig, ax = subplots()
            ax.plot(range(0, max_collisions + 1), counts_list)
            if markov:
                x = np.linspace(0, max_collisions, 2000)
                y = (np.cos(theta) ** x) / 2
                ax.plot(x, y)
            else:
                x = np.linspace(0, max_collisions, 2000)
                y = (np.cos(x * theta / 2) ** 2 - np.sin(x * theta / 2) ** 2) / 2
                ax.plot(x, y)

        else:
            plot_results(theoretical_concurrence_list, concurrence_list, theoretical_tangle_list,
                         tangle_ub_list, tangle_lb_list, fidelity_list, witness_list, tangle_witness_GHZ_list,
                         tangle_witness_tri_list, trace_squared_list, max_collisions, directory)


            if save_results:
                save_data(theoretical_rho_list, rho_list, theoretical_concurrence_list, concurrence_list,
                          theoretical_tangle_list, tangle_ub_list, tangle_lb_list, fidelity_list, witness_list,
                          tangle_witness_GHZ_list, tangle_witness_tri_list, trace_squared_list,
                          max_collisions, directory)

plt.show()

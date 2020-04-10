import os
import time
import numpy as np
from qiskit.visualization import plot_error_map
from datetime import datetime
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile
from qiskit.ignis.verification.tomography import StateTomographyFitter, state_tomography_circuits
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import thermal_relaxation_error
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.quantum_info import partial_trace
from qiskit.tools.monitor import job_monitor

# Requires IBMQ.save_account(TOKEN)

class Teleportation_Protocol(object):
    def __init__(self, protocol='Bell_teleport', device='qasm_simulator', live=False, qasm_sim=False,
                 noise_model=None, shots=1024, save_results=True, directory=None): # QIP_Task initialisation function

        # Stores protocol information
        self.protocol = protocol
        self.live = live
        self.shots = shots
        self.save_results = save_results
        self.directory = directory

        if device == 'qasm_simulator': # Defines settings for qasm_simulator use only
            self.qasm_sim = False
            self.backend = Aer.get_backend('qasm_simulator')
            self.device = self.backend
            self.coupling_map = None
            if noise_model:
                self.noise_model = noise_model
                self.basis_gates = self.noise_model.basis_gates
            else:
                self.noise_model = None
                self.basis_gates = None
        else: # Defines settings for processor use
            self.qasm_sim = qasm_sim
            provider = IBMQ.get_provider(group='open')
            self.device = provider.get_backend(device)
            if save_results:
                plot_error_map(self.device).savefig('{}/error_map.svg'.format(self.directory), format='svg')
            if self.live: # Defines settings for live simulations
                self.backend = self.device
                self.coupling_map = None
                self.noise_model = None
                self.basis_gates = None
            else: # Defines settings for artificial simulations
                from qiskit.providers.aer.noise import NoiseModel

                self.backend = Aer.get_backend('qasm_simulator')
                self.properties = self.device.properties()
                self.coupling_map = self.device.configuration().coupling_map
                self.noise_model = NoiseModel.from_backend(self.properties)
                self.basis_gates = self.noise_model.basis_gates

    def tomography(self, qc, backend, device_settings, meas_qubits): # State tomography function
        tomography_circuits = state_tomography_circuits(qc, meas_qubits)
        if backend == Aer.get_backend('qasm_simulator'): # Job execution on qasm_simulator
            if device_settings:
                job = execute(tomography_circuits, backend=backend, coupling_map=self.coupling_map,
                              noise_model=self.noise_model, basis_gates=self.basis_gates, shots=self.shots,
                              optimization_level=3)
            else:
                job = execute(tomography_circuits, backend=backend, shots=self.shots, optimization_level=3)
            job_id = None
        else: # Job execution on live quantum processor
            job = execute(tomography_circuits, backend=backend, shots=self.shots, optimization_level=3)
            job_monitor(job) # hold and monitor job until it is completed
            job_id = job.job_id()
        tomography_results = job.result()

        rho = StateTomographyFitter(tomography_results, tomography_circuits).fit() # Fit results to circuits

        return rho, job_id

    @staticmethod
    def general_GHZ(qc, q, n, i): # create general GHZ state in circuit
        qc.h(q[i])
        for j in range(1, n):
            qc.cx(q[i], q[i + j])

    @staticmethod
    def general_GHZ_matrix(n): # Define general GHZ state
        psi = np.zeros(2 ** n)
        psi[0] = 1 / np.sqrt(2)
        psi[-1] = 1 / np.sqrt(2)

        return np.outer(psi, psi)

    @staticmethod
    def concurrence(rho): # Concurrence Function
        Y = np.array([[0, -1j],
                      [1j, 0]])
        spin_flip = np.kron(Y, Y)
        rho_tilde = np.matmul(np.matmul(spin_flip, np.matrix.conjugate(rho)), spin_flip)
        l = np.sort(np.nan_to_num(np.sqrt(np.real(np.linalg.eigvals(np.matmul(rho, rho_tilde))))))[::-1]

        return np.max([0, l[0] - l[1] - l[2] - l[3]])

    @staticmethod
    def three_tangle(rho, N=10, error=1e-30, Nc=8, Np=4, solver_options=None): # Three-tangle function
        from scipy.optimize import minimize, NonlinearConstraint, Bounds

        if solver_options is None: # General parameters for the solver
            solver_options = {'gtol': 1e-8,
                              'xtol': 1e-8,
                              'barrier_tol': 1e-8,
                              'maxiter': 200,
                              'sparse_jacobian': False,
                              'factorization_method': 'SVDFactorization',
                              'initial_tr_radius': 1,
                              'initial_constr_penalty': 1,
                              'initial_barrier_parameter': 1e-8,
                              'initial_barrier_tolerance': 1,
                              'verbose': 3}

        def real_to_complex(z):  # real vector of length 2n to complex of length n function
            return z[:len(z) // 2] + 1j * z[len(z) // 2:]

        def complex_to_real(z):  # complex vector of length n to real vector of length 2n function
            return np.concatenate((np.real(z), np.imag(z)))

        def theoretical_tangle(phi): # Theoretical Three-tangle function
            d1 = phi[0] ** 2 * phi[7] ** 2 + phi[1] ** 2 * phi[6] ** 2 + \
                 phi[2] ** 2 * phi[5] ** 2 + phi[4] ** 2 * phi[3] ** 2
            d2 = phi[0] * phi[7] * phi[3] * phi[4] + phi[0] * phi[7] * phi[5] * phi[2] + \
                 phi[0] * phi[7] * phi[6] * phi[1] + phi[3] * phi[4] * phi[5] * phi[2] + \
                 phi[3] * phi[4] * phi[6] * phi[1] + phi[5] * phi[2] * phi[6] * phi[1]
            d3 = phi[0] * phi[6] * phi[5] * phi[3] + phi[7] * phi[1] * phi[2] * phi[4]
            tau = 4 * np.abs(d1 - 2 * d2 + 4 * d3)

            return tau

        def objective(x, Nc, Np): # Objective function to be minimised
            p = x[:Np]
            result = 0
            for i in range(Np):
                j = Np + i * 2 * Nc
                c = real_to_complex(x[j:j + 2 * Nc])

                result += p[i] * theoretical_tangle(c)

            return result

        def cons1(x): # sum of probability constraint
            return 1 - np.sum(x[:Np])

        def cons2(x): # sum of coefficients constraint
            result = np.zeros(Np)
            for i in range(Np):
                j = Np + i * 2 * Nc
                d = real_to_complex(x[j:j + 2 * Nc])
                result[i] = 1 - np.linalg.norm(d) ** 2

            return result

        def cons3(x): # contraint on sum of mixed states to be equal to inputted rho
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

        # Append constraints to list
        constraints = []
        constraints.append(NonlinearConstraint(cons1, -error, error))
        lower_constraints = []
        upper_constraints = []
        for j in range(Np):
            lower_constraints.append(-error)
            upper_constraints.append(error)
        constraints.append(NonlinearConstraint(cons2, lower_constraints, upper_constraints))
        lower_constraints = []
        upper_constraints = []
        for j in range(2 * Nc ** 2):
            lower_constraints.append(-error)
            upper_constraints.append(error)
        constraints.append(NonlinearConstraint(cons3, lower_constraints, upper_constraints))

        # Append bounds to list
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

        # Take N random starting positions
        tangles = np.zeros(N)
        for j in range(N):
            # generate random probabilities
            p = np.random.random(size=Np)
            p /= np.sum(p)
            x0 = np.array(p)

            # generate random coefficients
            for k in range(Np):
                a = np.random.random(size=Nc) * 2 - 1
                b = np.random.random(size=Nc) * 2 - 1
                c = a + 1j * b
                c /= np.linalg.norm(c)
                c = complex_to_real(c)
                x0 = np.append(x0, c)

            # solve objective function using trust-contrained algorithm
            try:
                tangles[j] = minimize(objective, x0, args=(Nc, Np), method='trust-constr', constraints=constraints,
                                      bounds=bounds, options=solver_options)['fun']
            except:
                pass

            # crude way of discarding negative results (possibly due to errors in numerical calculations)
            if tangles[j] < 0:
                tangles[j] = np.inf

        return np.min(tangles)

    @staticmethod
    def save_data(directory, **kwargs): # function to save kwargs as json file
        import pickle

        print('Saving data - do not quit.\n')

        data = {}
        for i in kwargs:
            data[i] = kwargs[i]

        f = open('{}/data.json'.format(directory), 'wb')
        pickle.dump(data, f, protocol=0)
        f.close()

        print('Data saved.\n')

    def collisional_model(self,
                          channel='phase_damping',
                          target_qubit=3,
                          collision_number=5,
                          theta=np.pi / 4,
                          initial_statevector=np.array([1, 1] / np.sqrt(2)),
                          print_results=True,
                          **kwargs): # collisional model function

        def collision(channel, **kwargs): # introduces one collision into the circuit
            if channel == 'phase_damping':
                phase_damping(**kwargs)

        def phase_damping(qc, q, a, k, theta): # sets up phase damping collision
            qc.h(a[k - 1])
            qc.cx(a[k - 1], q[-1])
            qc.rz(theta, q[-1])
            qc.cx(a[k - 1], q[-1])

        # obtains unitary function for the phase damping operator
        def phase_damping_operator(n, theta):
            if n: # unitary function when measuring entangled state
                _qc = QuantumCircuit(n)
                _qc.h(-1)
                _qc.cx(-1, -2)
                _qc.rz(theta, -2)
                _qc.cx(-1, -2)
            else: # unitary function when measuring teleported state
                _qc = QuantumCircuit(2)
                _qc.h(1)
                _qc.cx(1, 0)
                _qc.rz(theta, 0)
                _qc.cx(1, 0)

            job = execute(_qc, Aer.get_backend('unitary_simulator'))
            result = job.result()

            return result.get_unitary()

        def evolve_theoretical_rho(rho, U, traced_out_qubit): # evolution function of state after every collision
            evolving_rho = np.kron(np.array([[1, 0], [0, 0]]), rho)
            evolved_rho = np.matmul(np.matmul(U, evolving_rho), U.conj().T)

            return partial_trace(evolved_rho, [traced_out_qubit]).data

        def apply_protocol(qc, state): # apply teleportation protocol
            qc.barrier()  # barrier to separate components

            qc.cx(0, 1)
            qc.h(0)
            if state == 'GHZ':  # protocol for secret sharing
                qc.h(2)
                qc.cz(2, 3)
                qc.cx(1, 3)
                qc.cz(0, 3)
            elif state == 'Bell':  # protocol for standard teleportation
                qc.cx(1, 2)
                qc.cz(0, 2)

        def revert(qc, d): # Function to revert back before protocol was applied
            for _ in range(d):
                qc.data.pop(-1)

        print('Collisional model for protocol {} on {}:'.format(self.protocol, self.device))
        print('Channel: {}, number of collisions: {}\n'.format(channel, collision_number))

        if self.protocol == 'GHZ_teleport': # Settings for secret sharing protocol
            state = 'GHZ'
            n = 3
            d = 6
            theoretical_rho_S = self.general_GHZ_matrix(n) # Create theoretical GHZ state
        elif self.protocol == 'Bell_teleport': # Settings for standard teleportation protocol
            state = 'Bell'
            n = 2
            d = 4
            theoretical_rho_S = self.general_GHZ_matrix(n) # Create theoretical Bell state

        # Create quantum circuit
        q = QuantumRegister(n+1, 'q')
        qc = QuantumCircuit(q)

        # Initialise random qubit state to be teleported
        qc.initialize(initial_statevector, q[0])
        theoretical_rho_T = np.outer(initial_statevector, np.conj(initial_statevector))

        self.general_GHZ(qc, q, n, 1) # Create general GHZ state in circuit

        # Generate ancilla states
        if collision_number > 0:
            a = QuantumRegister(collision_number, 'a')
            qc.add_register(a)

        if channel == 'phase_damping': # Create phase damping operators
            U_S = phase_damping_operator(n+1, theta)
            U_T = phase_damping_operator(None, theta)

        # create lists used to store data for simulation
        theoretical_rho_S_list = []
        rho_S_list = []
        theoretical_rho_T_list = []
        rho_T_list = []
        numerical_tangle_list = []
        concurrence_list = []
        fidelity_S_list = []
        fidelity_T_list = []
        rho_S_list_sim = []
        rho_T_list_sim = []
        numerical_tangle_list_sim = []
        concurrence_list_sim = []
        fidelity_S_list_sim = []
        fidelity_T_list_sim = []
        job_ids_list = []

        # Start simulation and collision process
        for k in range(collision_number + 1):
            if k > 0: # Apply collision to circuit
                collision(channel, qc=qc, q=q, a=a, k=k, theta=theta)
                theoretical_rho_S = evolve_theoretical_rho(theoretical_rho_S, U_S, n)
                theoretical_rho_T = evolve_theoretical_rho(theoretical_rho_T, U_T, 1)

            if self.protocol == 'GHZ_teleport': # State tomography for GHZ state
                rho_S, S_id = self.tomography(qc, self.backend, True, [q[1], q[2], q[3]])
            elif self.protocol == 'Bell_teleport': # State tomography for Bell state
                rho_S, S_id = self.tomography(qc, self.backend, True, [q[1], q[2]])

            apply_protocol(qc, state) # applies teleportation protocol in circuit

            # State tomography for teleported state
            rho_T, T_id = self.tomography(qc, self.backend, True, [q[target_qubit]])

            revert(qc, d) # Revert back before protocol was applied

            # Append results to respective lists, calculated concurrence/three-tangle and fidelity
            rho_S_list.append(rho_S)
            theoretical_rho_S_list.append(theoretical_rho_S)
            rho_T_list.append(rho_T)
            theoretical_rho_T_list.append(theoretical_rho_T)
            job_ids_list.append([S_id, T_id])
            if self.protocol == 'GHZ_teleport':
                NT = self.three_tangle(rho_S, **kwargs)
                numerical_tangle_list.append(NT)
            if self.protocol == 'Bell_teleport':
                C = self.concurrence(rho_S)
                concurrence_list.append(C)
            F_S = state_fidelity(theoretical_rho_S_list[0], rho_S)
            fidelity_S_list.append(F_S)
            F_T = state_fidelity(theoretical_rho_T_list[0], rho_T)
            fidelity_T_list.append(F_T)

            # simulate artificial quantum computer
            if self.qasm_sim:
                if self.protocol == 'GHZ_teleport':  # State tomography for GHZ state
                    rho_S_sim, _ = self.tomography(qc, Aer.get_backend('qasm_simulator'), False, [q[1], q[2], q[3]])
                elif self.protocol == 'Bell_teleport':  # State tomography for Bell state
                    rho_S_sim, _= self.tomography(qc, Aer.get_backend('qasm_simulator'), False, [q[1], q[2]])

                apply_protocol(qc, state)  # applies teleportation protocol in circuit

                rho_T_sim, _ = self.tomography(qc, Aer.get_backend('qasm_simulator'), False, [q[target_qubit]])

                revert(qc, d)  # Revert back before protocol was applied

                # Append results to respective lists, calculated concurrence/three-tangle and fidelity
                rho_S_list_sim.append(rho_S_sim)
                rho_T_list_sim.append(rho_T_sim)
                if self.protocol == 'GHZ_teleport':
                    NT_sim = self.three_tangle(rho_S_sim, **kwargs)
                    numerical_tangle_list_sim.append(NT_sim)
                if self.protocol == 'Bell_teleport':
                    C_sim = self.concurrence(rho_S_sim)
                    concurrence_list_sim.append(C_sim)
                F_S_sim = state_fidelity(theoretical_rho_S_list[0], rho_S_sim)
                fidelity_S_list_sim.append(F_S_sim)
                F_T_sim = state_fidelity(theoretical_rho_T_list[0], rho_T_sim)
                fidelity_T_list_sim.append(F_T_sim)

            # Print results and other quantities
            if print_results:
                print("Collision Number:", k)
                print("Original {} Density Matrix:".format(state))
                print(theoretical_rho_S_list[0])
                print("Theoretical {} Density Matrix:".format(state))
                print(theoretical_rho_S)
                print("Measured {} Density Matrix:".format(state))
                print(rho_S)
                print("Original Teleported Density Matrix:")
                print(theoretical_rho_T_list[0])
                print("Theoretical Teleported Density Matrix:")
                print(theoretical_rho_T)
                print("Measured Teleported Density Matrix:")
                print(rho_T)
                print("Trace:", np.real(np.trace(rho_S)))
                if self.protocol == 'GHZ_teleport':
                    print("Numerical Three-Tangle:", NT)
                if self.protocol == 'Bell_teleport':
                    print("Concurrence:", C)
                print("{} State Fidelity:".format(state), F_S)
                print("Teleported state Fidelity:", F_T)
                print("{} state Eigenvalues:".format(state), np.sort(np.real(np.linalg.eigvals(rho_S)))[::-1])
                print("Teleported state Eigenvalues:", np.sort(np.real(np.linalg.eigvals(rho_T)))[::-1])
                print('\n')

        if self.save_results: # save results
            self.save_data(self.directory, collision_number=collision_number,
                           theoretical_rho_S_list=theoretical_rho_S_list, rho_S_list=rho_S_list,
                           theoretical_rho_T_list=theoretical_rho_T_list, rho_T_list=rho_T_list,
                           numerical_tangle_list=numerical_tangle_list, concurrence_list=concurrence_list,
                           fidelity_S_list=fidelity_S_list, fidelity_T_list=fidelity_T_list,
                           rho_S_list_sim=rho_S_list_sim, rho_T_list_sim=rho_T_list_sim,
                           numerical_tangle_list_sim=numerical_tangle_list_sim,
                           concurrence_list_sim=concurrence_list_sim, fidelity_S_list_sim=fidelity_S_list_sim,
                           fidelity_T_list_sim=fidelity_T_list_sim, theta=theta,
                           initial_statevector=initial_statevector, job_ids_list=job_ids_list, **kwargs)

if __name__ == '__main__':
    devices = ['qasm_simulator', 'ibmq_16_melbourne', 'ibmq_london', 'ibmq_burlington', 'ibmq_essex', 'ibmq_ourense',
                'ibmq_vigo', 'ibmq_5_yorktown']

    channel = 'phase_damping'
    device = 'ibmq_5_yorktown'
    if device != 'qasm_simulator':
        IBMQ.load_account()
    protocol = 'Bell_teleport'
    shots = 8192
    target_qubit = 2 # 1: Alice, 2: Bob, 3: Charlie
    collision_number = 2
    theta = np.pi / 3
    th = np.pi / 2 # np.random.random() * np.pi
    phi = 0 # np.random.random() * 2 * np.pi
    initial_statevector = np.array([np.cos(th / 2), np.exp(1j * phi) * np.sin(th / 2)])
    markov = True
    noise_model = None  # noise_model(n + collision_number)
    print_results = True
    save_results = True
    qasm_sim = True
    live = True

    if save_results:
        degrees = round(theta * 180 / np.pi, 2)

        date_time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        directory = 'data/{}/{}_{}_{}_{}_{}'.format(device, date_time, shots, degrees,
                                                    round(initial_statevector[0], 3), round(initial_statevector[1], 3))

        try:
            os.mkdir('data')
        except FileExistsError:
            pass
        try:
            os.mkdir('data/{}'.format(device))
        except FileExistsError:
            pass
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
    else:
        directory = None

    task = Teleportation_Protocol(protocol=protocol, device=device, live=live, qasm_sim=qasm_sim,
                    noise_model=noise_model, shots=shots, save_results=save_results, directory=directory)

    t = time.perf_counter()

    task.collisional_model(channel=channel,
                           target_qubit=target_qubit,
                           collision_number=collision_number,
                           theta=theta,
                           markov=markov,
                           initial_statevector=initial_statevector,
                           print_results=print_results)

    elapsed_time = time.perf_counter() - t
    print('Total time elapsed for simulation: {:.2f}\n'.format(elapsed_time))

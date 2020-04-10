import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds


def tangle2():
    Z = 1
    error = 1e-8

    rho = np.array([[1/3, 0, 0, 0],
                    [0, 1/3, 1/3, 0],
                    [0, 1/3, 1/3, 0],
                    [0, 0, 0, 0]])

    print(rho)

    Nc = 4
    Np = 4
    C = np.zeros(Z)

    def real_to_complex(z):  # real vector of length 2n -> complex of length n
        return z[:len(z) // 2] + 1j * z[len(z) // 2:]

    def complex_to_real(z):  # complex vector of length n -> real of length 2n
        return np.concatenate((np.real(z), np.imag(z)))

    def theoretical_concurrence(phi):
        C = 2 * abs(phi[0] * phi[3] - phi[1] * phi[2])

        return C

    def objective(x, Nc, Np):
        p = x[:Np]
        result = 0
        for i in range(Np):
            j = Np + i * 2 * Nc
            c = real_to_complex(x[j:j + 2 * Nc])

            result += p[i] * theoretical_concurrence(c)

        return result

    for i in range(Z):
        p = np.random.sample(Np)
        p /= np.sum(p)
        x0 = np.array(p)
        for j in range(Np):
            a = np.random.sample(Nc) * 2 - 1
            b = np.random.sample(Nc) * 2 - 1
            c = a + 1j * b
            c /= np.linalg.norm(c)
            c = complex_to_real(c)
            x0 = np.append(x0, c)

        constraints = []

        def cons1(x):
            result = []
            for i in range(Np):
                result.append(x[i])

            return result

        lower_constraints = []
        upper_constraints = []
        for j in range(Np):
            lower_constraints.append(0)
            upper_constraints.append(1)

        constraints.append(NonlinearConstraint(cons1, lower_constraints, upper_constraints))

        def cons2(x):

            return 1 - np.sum(x[:Np])

        constraints.append(NonlinearConstraint(cons2, -error, error))

        def cons3(x):
            c = np.zeros(Np)
            for i in range(Np):
                j = Np + i * 2 * Nc
                d = real_to_complex(x[j:j + 2 * Nc])
                c[i] = 1 - np.linalg.norm(d) ** 2

            return c

        lower_constraints = []
        upper_constraints = []
        for j in range(Np):
            lower_constraints.append(-error)
            upper_constraints.append(error)

        constraints.append(NonlinearConstraint(cons3, lower_constraints, upper_constraints))

        def cons4(x):
            p = x[:Np]
            c = np.zeros((Nc, Nc), dtype=complex)
            for i in range(Np):
                j = i * 2 * Nc
                c[i] = real_to_complex(x[Np + j:Np + 2 * Nc + j])

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

        constraints.append(NonlinearConstraint(cons4, lower_constraints, upper_constraints))

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

        for i in range(Z):
            p = np.random.normal(size=Np)
            p /= np.sum(p)
            x0 = np.array(p)
            for j in range(Np):
                a = np.random.normal(size=Nc) * 2 - 1
                b = np.random.normal(size=Nc) * 2 - 1
                c = a + 1j * b
                c /= np.linalg.norm(c)
                c = complex_to_real(c)
                x0 = np.append(x0, c)

            C[i] = minimize(objective, x0, args=(Nc, Np), method='trust-constr', constraints=constraints, bounds=bounds,
                     options={'gtol': 1e-8,
                              'xtol': 1e-8,
                              'barrier_tol': 1e-8,
                              'maxiter': 1000,
                              'sparse_jacobian': False,
                              'factorization_method': 'SVDFactorization',
                              'initial_tr_radius': 100,
                              'initial_constr_penalty': 100,
                              'initial_barrier_parameter': 0.1,
                              'initial_barrier_tolerance': 0.1,
                              'verbose': 3})['fun']

    return np.min(C)

print(tangle2())

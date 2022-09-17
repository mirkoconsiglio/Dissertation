import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize


def three_tangle(rho, N=10, kappa=1e6, Nc=8, Np=4, solver_options=None):  # Three-tangle function
	'''
	:param rho: 8 x 8 density matrix input to calculate tangle
	:param N: Number of random starting points for the minimisation problem
	:param error: Used to introduce a numerical leeway for constraints to avoid diverging
	solutions
	:param Nc: Dimension of the density matrix used for the generation of random complex
	coefficients of states
	:param Np: Number of partitions in the decomposition. Caratheodory’s theorem states Np = 4
	is enough to obtain the optimised three-tangle of mixed states of rank 2
	:param solver_options: Options for the trust-constrained solver, such as; tolerance,
	maximum number of iterations, factorisation method, inital trust radius and barrier
	parameters
	:return: The minimum value of the three-tangle over all found N local minima
	'''
	
	if solver_options is None:  # General parameters for the solver
		solver_options = {'disp': True}
	
	def real_to_complex(z):  # real vector of length 2n to complex of length n function
		return z[:len(z) // 2] + 1j * z[len(z) // 2:]
	
	def complex_to_real(z):  # complex vector of length n to real vector of length 2n function
		return np.concatenate((np.real(z), np.imag(z)))
	
	def theoretical_tangle(phi):  # Theoretical Three-tangle function
		d1 = phi[0] ** 2 * phi[7] ** 2 + phi[1] ** 2 * phi[6] ** 2 + \
		     phi[2] ** 2 * phi[5] ** 2 + phi[4] ** 2 * phi[3] ** 2
		d2 = phi[0] * phi[7] * phi[3] * phi[4] + phi[0] * phi[7] * phi[5] * phi[2] + \
		     phi[0] * phi[7] * phi[6] * phi[1] + phi[3] * phi[4] * phi[5] * phi[2] + \
		     phi[3] * phi[4] * phi[6] * phi[1] + phi[5] * phi[2] * phi[6] * phi[1]
		d3 = phi[0] * phi[6] * phi[5] * phi[3] + phi[7] * phi[1] * phi[2] * phi[4]
		tau = 4 * np.abs(d1 - 2 * d2 + 4 * d3)
		
		return np.sqrt(tau)
	
	def objective(x, Nc, Np):  # Objective function to be minimised
		p = x[:Np]
		result = 0
		for i in range(Np):
			j = Np + i * 2 * Nc
			c = real_to_complex(x[j:j + 2 * Nc])
			
			result += p[i] * theoretical_tangle(c)
		
		c = np.zeros((Np, Nc), dtype=complex)
		for i in range(Np):
			j = Np + i * 2 * Nc
			c[i] = real_to_complex(x[j:j + 2 * Nc])
		
		residual = 0
		for i in range(Nc):
			for j in range(Nc):
				res = 0
				for k in range(Np):
					res += p[k] * c[k, i] * np.conj(c[k, j])
				res -= rho[i, j]
				residual += np.abs(res) ** 2
		
		result += kappa * residual
		
		return result
	
	def cons0(x):
		result = np.zeros(2 * (Np - 1))
		for i in range(Np - 1):
			result[2 * i] = x[i]
			result[2 * i + 1] = 1 - x[i]
		
		return result
	
	def cons1(x):  # sum of probability constraint
		return 1 - np.sum(x[:Np - 1])
	
	def cons2(x):  # sum of coefficients constraint
		result = np.zeros(Np)
		for i in range(Np):
			j = Np + i * 2 * Nc
			d = real_to_complex(x[j:j + 2 * Nc])
			result[i] = 1 - np.linalg.norm(d) ** 2
		
		return result
	
	# Append constraints to list
	constraints = []
	lower_constraints = []
	upper_constraints = []
	for j in range(2 * (Np - 1)):
		lower_constraints.append(0)
		upper_constraints.append(np.infty)
	constraints.append(NonlinearConstraint(cons0, lower_constraints, upper_constraints))
	constraints.append(NonlinearConstraint(cons1, 0, 1))
	lower_constraints = []
	upper_constraints = []
	for j in range(Np):
		lower_constraints.append(0)
		upper_constraints.append(1)
	constraints.append(NonlinearConstraint(cons2, lower_constraints, upper_constraints))
	
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
		
		# solve objective function using trust-constrained algorithm
		try:
			output = minimize(objective, x0, args=(Nc, Np), constraints=constraints,
			                  options=solver_options)
			print(output['fun'])
			tangles[j] = output['fun']
		except:
			pass
		
		# crude way of discarding negative results (possibly due to errors in numerical
		# calculations)
		if tangles[j] < 0:
			tangles[j] = np.inf
	
	return np.min(tangles)


rho = np.array([[0.42875, 0, 0, 0, 0, 0, 0, 0.36450000000000005],
                [0, 0.02375, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.02375, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.02375, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.02375, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.02375, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.02375, 0],
                [0.36450000000000005, 0, 0, 0, 0, 0, 0, 0.42875]])

rho = np.array([[1 / 2, 0, 0, 0, 0, 0, 0, 1 / 2],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, 0, 0, 0, 1 / 2]])

print(three_tangle(rho, N=100))

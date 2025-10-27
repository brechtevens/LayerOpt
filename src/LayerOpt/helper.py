import numpy as np
from numba import jit, njit, prange

from scipy.optimize import basinhopping
import multiprocessing as mp
import itertools

import matplotlib.pyplot as plt
from matplotlib import interactive

import time
from typing import Union
from .Permittivity import PermittivityModel
from .Experiment import ExperimentData


class MultilayerStructure:
	"""
		Class for defining multilayer dielectric slab structures.

		Attributes
		----------
		t : float
			Thickness of each layer in meters.
		n_lay : int
			Number of layers.
		impedance_out : complex
			Characteristic impedance of the last (outgoing) medium.
			Typically equal to one for free space.
	"""
	def __init__(self, t=5e-3, n_lay=5, impedance_out=1):
		"""
			Initialize the multilayer dielectric slab structure with the given parameters or defaults.

			Parameters
			----------
			t : float, optional
				Thickness of each layer in meters. Default is 5e-3.
			n_lay : int, optional
				Number of layers. Default is 5.
			impedance_out : complex or float, optional
				Characteristic impedance of the outgoing medium. Default is 1 (free space).
			"""
		self.t = t
		self.n_lay = n_lay
		self.impedance_out = impedance_out

	@property
	def total_t(self):
		"""The total thickness of the multilayer dielectric slab structure."""
		return float(self.t*self.n_lay)

	@staticmethod
	@njit(cache=True)
	def get_k0(f):
		"""
			Computes the free-space wavenumber of a wave with a given frequency.
						
			Parameters
			----------
			f : float
				Frequency, given in [GHz].

			Returns
			-------
			float 
				The corresponding free-space wavenumber.
		"""
		return 2 * np.pi * f * 10**9 / 299792458

	@staticmethod
	@njit(cache=True)
	def _get_waves_serial(x, freq_vec, n_lay, t, get_k0, p, p_extra, impedance_out, real_permittivity, imaginary_permittivity):
		"""
			Compute the normalized forward and backward waves of a multilayer dielectric slab structure in a serial manner.

			This function calculates the electromagnetic wave reflection properties for a multilayer dielectric slab structure
			based on the permittivity properties of the layers. The method follows Section 6.1 from Orfanidis' book 
			"Electromagnetic Waves and Antennas", specifically equations (6.1.2) and (6.1.3) to compute the forward and 
			backward waves at each interface.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the normalized forward and backward waves in GHz.
			n_lay : int
				Number of layers in the multilayer structure.
			t : float
				Thickness of each layer, used to compute phase shifts.
			get_k0 : callable
				Function to compute the free-space wavenumber `k0` for a given frequency.
			p : array-like
				Parameters of the permittivity model.
			p_extra : array-like
				Additional fixed parameters of the permittivity model.
			impedance_out : complex
				Characteristic impedance of the last, outgoing medium.
			real_permittivity : callable
				Function to compute the real part of the permittivity for each layer.
			imaginary_permittivity : callable
				Function to compute the imaginary part of the permittivity for each layer.

			Returns
			-------
			E : np.ndarray, shape (L, 2)
				Complex array containing the electric field amplitudes for each frequency, where `L` is the number of frequencies
				in `freq_vec`. `E[:, 0]` represents the forward-traveling wave and `E[:, 1]` represents the backward-traveling wave.
		"""
		L = len(freq_vec)

		# Preallocate E, RHO and TRANS matrices
		E = np.zeros((L, 2), dtype=np.complex128)

		eta = np.ones((L,n_lay+2), dtype=np.complex128)			# characteristic impedance
		eps_list = np.zeros((L, n_lay), dtype=np.complex128)
		n = np.zeros((L, n_lay+1), dtype=np.complex128)			# refractive index
		rho = np.zeros((L, n_lay + 1), dtype=np.complex128)		# reflection coefficient

		RHO = np.zeros((2, 2), dtype=np.complex128)
		TRANS = np.zeros((2, 2), dtype=np.complex128)

		k0 = get_k0(freq_vec)

		eta[:,-1] = impedance_out

		for k in range(L):
			for i in range(n_lay):
				eps_list[k, i] = (real_permittivity(x[i], freq_vec[k], p, p_extra) + 
							1j * imaginary_permittivity(x[i], freq_vec[k], p, p_extra))
				eta[k, i+1] = np.sqrt(1/eps_list[k, i])
				n[k, i+1] = np.sqrt(eps_list[k, i])

			rho[k,:] = (eta[k,1:] - eta[k,:-1]) / (eta[k,1:] + eta[k,:-1])

			# Base case for the last layer
			if rho[k,n_lay] == -1:
				E[k, 0] = 1
				E[k, 1] = -1
			else:
				E[k, 0] = 1 / (1 + rho[k,n_lay])
				E[k, 1] = rho[k,n_lay] / (1 + rho[k,n_lay])

			# Loop for layers
			for m in range(n_lay):
				l = n_lay - 1 - m

				# Update RHO matrix
				RHO[0, 0] = 1
				RHO[0, 1] = rho[k,l]
				RHO[1, 0] = rho[k,l]
				RHO[1, 1] = 1
				
				# Update TRANS matrix
				TRANS[0, 0] = np.exp(1j * k0[k] * t * n[k,l + 1])
				TRANS[0, 1] = 0
				TRANS[1, 0] = 0
				TRANS[1, 1] = np.exp(-1j * k0[k] * t * n[k,l + 1])

				# Update E with matrix multiplication
				E[k, :] = 1 / (1 + rho[k,l]) * np.dot(RHO, np.dot(TRANS, E[k, :]))

		return E

	@staticmethod
	@njit(cache=False, parallel=True)
	def _get_waves_parallel(x, freq_vec, n_lay, t, get_k0, p, p_extra, impedance_out, real_permittivity, imaginary_permittivity):
		"""
			Compute the normalized forward and backward waves of a multilayer dielectric slab structure in a parallel manner.

			This function calculates the electromagnetic wave reflection properties for a multilayer dielectric slab structure
			based on the permittivity properties of the layers. The method follows Section 6.1 from Orfanidis' book 
			"Electromagnetic Waves and Antennas", specifically equations (6.1.2) and (6.1.3) to compute the forward and 
			backward waves at each interface.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the normalized forward and backward waves in GHz.
			n_lay : int
				Number of layers in the multilayer structure.
			t : float
				Thickness of each layer, used to compute phase shifts.
			get_k0 : callable
				Function to compute the free-space wavenumber `k0` for a given frequency.
			p : array-like
				Parameters of the permittivity model.
			p_extra : array-like
				Additional fixed parameters of the permittivity model.
			impedance_out : complex
				Characteristic impedance of the last, outgoing medium.
			real_permittivity : callable
				Function to compute the real part of the permittivity for each layer.
			imaginary_permittivity : callable
				Function to compute the imaginary part of the permittivity for each layer.

			Returns
			-------
			E : np.ndarray, shape (L, 2)
				Complex array containing the electric field amplitudes for each frequency, where `L` is the number of frequencies
				in `freq_vec`. `E[:, 0]` represents the forward-traveling wave and `E[:, 1]` represents the backward-traveling wave.
		"""
		L = len(freq_vec)

		# Preallocate E, RHO and TRANS matrices
		E = np.zeros((L, 2), dtype=np.complex128)

		eta = np.ones((L,n_lay+2), dtype=np.complex128)
		eps_list = np.zeros((L, n_lay), dtype=np.complex128)
		n = np.zeros((L, n_lay+1), dtype=np.complex128)
		rho = np.zeros((L, n_lay + 1), dtype=np.complex128)

		k0 = get_k0(freq_vec)

		RHO = np.zeros((L, 2, 2), dtype=np.complex128)
		TRANS = np.zeros((L, 2, 2), dtype=np.complex128)

		eta[:,-1] = impedance_out

		for k in prange(L):
			for i in range(n_lay):
				eps_list[k, i] = (real_permittivity(x[i], freq_vec[k], p, p_extra) + 
							1j * imaginary_permittivity(x[i], freq_vec[k], p, p_extra))
				eta[k, i+1] = np.sqrt(1/eps_list[k, i])
				n[k, i+1] = np.sqrt(eps_list[k, i])

			rho[k,:] = (eta[k,1:] - eta[k,:-1]) / (eta[k,1:] + eta[k,:-1])

			# Base case for the last layer
			if rho[k,n_lay] == -1:
				E[k, 0] = 1
				E[k, 1] = -1
			else:
				E[k, 0] = 1 / (1 + rho[k,n_lay])
				E[k, 1] = rho[k,n_lay] / (1 + rho[k,n_lay])

			# Loop for layers
			for m in range(n_lay):
				l = n_lay - 1 - m

				# Update RHO matrix
				RHO[k, 0, 0] = 1
				RHO[k, 0, 1] = rho[k,l]
				RHO[k, 1, 0] = rho[k,l]
				RHO[k, 1, 1] = 1
				
				# Update TRANS matrix
				TRANS[k, 0, 0] = np.exp(1j * k0[k] * t * n[k,l + 1])
				TRANS[k, 0, 1] = 0
				TRANS[k, 1, 0] = 0
				TRANS[k, 1, 1] = np.exp(-1j * k0[k] * t * n[k,l + 1])

				# Update E with matrix multiplication
				E[k, :] = 1 / (1 + rho[k,l]) * np.dot(RHO[k, :, :], np.dot(TRANS[k, :, :], E[k, :]))

		return E

	def get_waves(self, x : np.ndarray, freq_vec : np.ndarray, eps_model : PermittivityModel):
		"""
			Compute the normalized forward and backward waves.

			This function calculates the electromagnetic wave reflection properties for a multilayer dielectric slab structure
			based on the permittivity properties of the layers. The method follows Section 6.1 from Orfanidis' book 
			"Electromagnetic Waves and Antennas", specifically equations (6.1.2) and (6.1.3) to compute the forward and 
			backward waves at each interface. The computations are performed either in series or in parallel depending on 
			the number of considered frequencies.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the normalized forward and backward waves in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.

			Returns
			-------
			E : np.ndarray, shape (L, 2)
				Complex array containing the electric field amplitudes for each frequency, where `L` is the number of frequencies
				in `freq_vec`. `E[:, 0]` represents the forward-traveling wave and `E[:, 1]` represents the backward-traveling wave.
		"""
		if len(freq_vec) < 50:
			return self._get_waves_serial(
				x, freq_vec, self.n_lay, self.t, self.get_k0, eps_model.p, eps_model.p_extra, self.impedance_out, 
				eps_model._compute_real_permittivity, eps_model._compute_imaginary_permittivity
			)
		else:
			return self._get_waves_parallel(
				x, freq_vec, self.n_lay, self.t, self.get_k0, eps_model.p, eps_model.p_extra, self.impedance_out,
				eps_model._compute_real_permittivity, eps_model._compute_imaginary_permittivity
			)

	def get_S_parameters(self, x, freq_vec, eps_model : PermittivityModel):
		"""
			Compute the S-parameters (S11 and S21).

			This function calculates two scattering parameters (S11 and S21) for a multilayer slab structure based on the 
			given frequency vector, permittivity model, and Material parameters for each layer. See also Orfanidis' equation (14.1.3).

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the S-parameters in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.

			Returns
			-------
			S11 : np.ndarray, shape (L, 1)
				Reflection coefficient for each frequency, where `L` is the number of frequencies in `freq_vec`.
			S21 : np.ndarray, shape (L, 1)
				Transmission coefficient for each frequency, where `L` is the number of frequencies in `freq_vec`.
		"""
		E = self.get_waves(x, freq_vec, eps_model)

		S11 = E[:, 1]/E[:, 0]
		if self.impedance_out == 0:
			S21 = np.zeros_like(E[:, 0])
		else:
			S21 = E[:, 0]**(-1)
		return S11, S21

	def get_wave_properties(self, x, freq_vec, eps_model : PermittivityModel):
		"""
			Compute the reflectance (R), transmittance (T), and absorption (A). The computations are based on 
			the S-parameters (S11 and S21) obtained from `get_S_parameters`.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the wave properties in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.

			Returns
			-------
			R : np.ndarray, shape (L, 1)
				Reflectance for each frequency, where `L` is the number of frequencies in `freq_vec`.
			T : np.ndarray, shape (L, 1)
				Transmittance for each frequency, where `L` is the number of frequencies in `freq_vec`.
			A : np.ndarray, shape (L, 1)
				Absorption for each frequency, where `L` is the number of frequencies in `freq_vec`.
		"""
		S11, S21 = self.get_S_parameters(x, freq_vec, eps_model)
		R = abs(S11)**2
		T = abs(S21)**2
		A = 1 - R - T
		return R, T, A

	def get_AR_SET(self, x, freq_vec, eps_model : PermittivityModel):
		"""
			Compute the absorption-to-reflectance ratio (A/R) and the shielding effectiveness (SET). The computations are based on 
			reflectance (R), transmittance (T), and absorption (A) obtained from `get_wave_properties`.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute the metrics in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.

			Returns
			-------
			AR : np.ndarray
				Absorption-to-reflectance ratio for each frequency.
			SET : np.ndarray
				Shielding effectiveness for each frequency in decibels.
		"""
		R, T, A = self.get_wave_properties(x, freq_vec, eps_model)
		return A/R, 10*np.log10(1/T)

	def get_worst_case_AR_SET(self, x, freq_vec, eps_model : PermittivityModel):
		"""
			Compute the worst-case (minimum) absorption-to-reflectance ratio (A/R) and shielding effectiveness (SET).

			This function evaluates the absorption-to-reflectance ratio and shielding effectiveness across all frequencies
			and returns the minimum values, representing the worst-case performance of the multilayer slab.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to evaluate the metrics in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.

			Returns
			-------
			min_AR : float
				Minimum absorption-to-reflectance ratio across all given frequencies.
			min_SET : float
				Minimum shielding effectiveness in decibels across all given frequencies.
		"""
		AR, SET = self.get_AR_SET(x, freq_vec, eps_model)
		return np.min(AR), np.min(SET)

	def _visualize_behavior(self, x, freq_vec, eps_model : PermittivityModel, label=''):
		"""
			Plot the absorption-to-reflectance ratio (A/R) versus shielding effectiveness (SET) for a multilayer slab,
			computing A/R and SET using `get_AR_SET` across the specified frequencies. If the maximum A/R exceeds 250,
			the y-axis is limited to a maximum of 250.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.
			freq_vec : array-like
				Frequencies at which to compute and plot the metrics in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.
			label : str, optional
				Label for the plot legend. Default is an empty string.
		"""
		AR, SET = self.get_AR_SET(x, freq_vec, eps_model)

		## plot response
		plt.plot(SET, AR, marker='d', linestyle="", label=label)
		plt.ylabel("A/R")
		plt.xlabel("Shielding effectiveness [dB]")
		
		current_ylim = plt.gca().get_ylim()
		if current_ylim[1] > 250:
			plt.ylim(bottom=-5, top=250)

	def visualize_behavior(self, arg : Union[np.ndarray, tuple, 'MultilayerProblemSolution'], freq_vec, eps_model : PermittivityModel, p=None, only_discrete=False, label=''):
		"""
			Visualize the absorption-to-reflectance ratio (A/R) versus shielding effectiveness (SET).

			Parameters
			----------
			arg : np.ndarray, tuple, or MultilayerProblemSolution
				Solution(s) to visualize. Can be a vector, a tuple, or a `MultilayerProblemSolution` object.
				If `arg` is a NumPy array or tuple, it is treated as a single solution.
				If `arg` is a `MultilayerProblemSolution`, the discrete solution (`x_discrete`) is always plotted,
				and the continuous solution (`x_cont`) is plotted unless `only_discrete` is True.
			freq_vec : array-like
				Frequencies at which to compute and plot the metrics in GHz.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.
			p : optional
				Optional additional parameters passed to `_visualize_behavior`.
			only_discrete : bool, default False
				If True, only the discrete solution is plotted when `arg` is a `MultilayerProblemSolution`.
			label : str, default ''
				Label for the plot legend.
		"""
		if isinstance(arg, np.ndarray) or isinstance(arg, tuple):
			self._visualize_behavior(arg, freq_vec, eps_model, p, label=label)
		else:
			self._visualize_behavior(arg.x_discrete, freq_vec, eps_model, p, label='discrete solution ' + label)
			if only_discrete:
				plt.legend()
			else:
				self._visualize_behavior(arg.x_cont, freq_vec, eps_model, p, label='continuous solution ' + label)
				plt.legend()

def closest_entries(array, x, m):
	"""
		Find the `m` entries in an array closest to a given value.

		This function returns the `m` values from `array` that are nearest to `x`. 
		If `m` equals the length of `array`, the full array is returned. The returned values
		are sorted according to their original order in the array.

		Parameters
		----------
		array : np.ndarray
			Input array to search.
		x : float
			Target value to compare against.
		m : int
			Number of closest entries to return. Must be less than or equal to the length of `array`.

		Returns
		-------
		tuple
			A tuple containing the `m` entries from `array` closest to `x`, sorted in the same order
			as they appear in `array`.

		Raises
		------
		AssertionError
			If `m` is greater than the length of `array`.
    """
	assert(m <= len(array))

	if m == len(array):
		return array
	else:
		# Compute the absolute differences between each entry and x
		differences = np.abs(array - x)
		
		# Get the indices of the m smallest differences
		indices = np.argpartition(differences, m)[:m]
		
		# Extract the entries corresponding to these indices
		closest_entries = array[indices]
		
		# Sort the closest entries by their original order in the array if needed
		closest_entries_sorted = closest_entries[np.argsort(array[indices])]
		
		return tuple(closest_entries_sorted)

class MultilayerOptimizer:
	"""
		Class for optimizing multilayer dielectric slab structures.

		Attributes
		----------
		structure : MultilayerStructure
			The multilayer slab structure to optimize.
		eps_model : PermittivityModel
			Model for the permittivity (dielectric constant) of the material.
		freq_optimize : np.ndarray
			Frequencies [GHz] at which to evaluate the optimization objectives.
		SE_min : float
			Minimum required shielding effectiveness in decibels.
		lb : float
			Lower bound for material parameters.
		ub : float
			Upper bound for material parameters.
		material_list : np.ndarray
			List of material parameters that can be manufactured.
		n_discr : int
			Number of material parameters to check during optimization at each layer.
	"""
	def __init__(self, structure: MultilayerStructure, eps_model: PermittivityModel, freq_optimize: np.ndarray = np.array([8, 12]), SE_min: float = 30, lb: float = 0.0, ub: float = 1.0, material_list: np.ndarray = [0, 0.5, 1], n_discr: int = 2):
		"""
			Initialize the optimizer with a structure, permittivity model, and optional settings.

			Parameters
			----------
			structure : MultilayerStructure
				The multilayer slab structure to optimize.
			eps_model : PermittivityModel
				Model for the permittivity (dielectric constant) of the material.
			freq_optimize : np.ndarray, optional
				Frequencies [GHz] at which to evaluate the optimization objectives.
				Default is np.array([8, 12]).
			SE_min : float, optional
				Minimum required shielding effectiveness in decibels. Default is 30.
			lb : float, optional
				Lower bound for material parameters. Default is 0.0.
			ub : float, optional
				Upper bound for material parameters. Default is 1.0.
			material_list : np.ndarray, optional
				List of material parameters that can be manufactured. Default is
				[lb, 0.5, ub].
			n_discr : int, optional
				Number of material parameters to check during optimization at each layer.
				Default is 2.
		"""
		self.structure = structure
		self.eps_model = eps_model
		self.freq_optimize = freq_optimize if freq_optimize is not None else np.array([8, 12])
		self.SE_min = SE_min
		self.lb = lb
		self.ub = ub
		self.material_list = material_list if material_list is not None else np.array(
			[self.lb, 0.5, 1, 2.5, 3.5, 5, 6, 7.5, 8.5, self.ub]
		)
		self.n_discr = n_discr
	@property
	def total_t(self):
		"""The total thickness of the multilayer dielectric slab structure."""
		return self.structure.total_t
	
	@property
	def n_lay(self):
		"""The number of layers of the multilayer dielectric slab structure."""
		return self.structure.n_lay

	def eval_cost(self, x):
		"""
			Evaluate the cost function for a given layer composition. The cost is defined as the 
			maximum ratio of reflectance to absorption (R/A) across the optimization frequencies.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.

			Returns
			-------
			float
				Maximum R/A ratio across the optimization frequencies.
		"""
		R, T, A = self.structure.get_wave_properties(x, self.freq_optimize, self.eps_model)
		return np.max(R/A)

	def eval_constraint(self, x):
		"""
			Evaluate the shielding effectiveness constraint for a given layer composition.

			The constraint ensures that the minimum shielding effectiveness across the 
			optimization frequencies meets the target SE_min. If `impedance_out` is zero, 
			the constraint is automatically satisfied.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.

			Returns
			-------
			float
				Constraint value in dB; positive or zero  if the design satisfies the SE_min requirement, 
				strictly negative if it does not.
		"""
		if self.structure.impedance_out == 0:
			return 0
		else:
			R, T, A = self.structure.get_wave_properties(x, self.freq_optimize, self.eps_model)
			return 10*np.log10(1/max(T)) - self.SE_min

	def eval_brute_cost(self, x):
		"""
			Evaluate a cost function for brute-force optimization with constraint handling.

			If the shielding effectiveness constraint is violated, a large penalty is returned; 
			otherwise, the normal cost (maximum R/A) is computed.

			Parameters
			----------
			x : array-like
				Material parameters for each layer in the multilayer structure.

			Returns
			-------
			float
				Penalized cost: either a large number if the constraint is violated or the maximum R/A ratio.
		"""
		if self.eval_constraint(x) < 0:
			return 10**8
		else:
			return self.eval_cost(x)

	def optimize(self, x_0):
		"""
			Optimize the material parameters for each layer in  multilayer dielectric slab structure to minimize R/A while satisfying shielding constraints.

			This method performs a two-step optimization:
			
			1. Continuous optimization using a basin-hopping algorithm with bounds on  and SE constraints.
			2. Discrete optimization over manufacturable Material parameters based on the continuous solution.

			The optimization can iterate multiple times around the discrete solution to search for nearby minima.

			Parameters
			----------
			x_0 : array-like
				Initial guess for the layer Material parameters.

			Returns
			-------
			MultilayerProblemSolution
				MultilayerProblemSolution object containing the initial, continuous, and discrete solutions, their corresponding cost values,
				the ranges used for discrete optimization, and timing information for precompilation and both optimization steps.
		"""
		# Optimizer options
		options = {"maxiter": 1e4, "disp": False}
		bounds = [(self.lb,self.ub) for _ in range(len(x_0))]

		# Precompile cost function using JIT
		tic = time.perf_counter()
		self.eval_cost(x_0)
		toc =  time.perf_counter()
		precompile_time = toc - tic


		def optimization_step(x_0):
			tic = time.perf_counter()
			opt = basinhopping(
				self.eval_cost, 
				x_0, 
				minimizer_kwargs={
					# "method":"SLSQP", 
					"bounds":bounds, 
					"constraints":{"type": "ineq", "fun": self.eval_constraint}, 
					"options":options
				}, 
				niter=100,
				T=0.1
			)
			toc =  time.perf_counter()
			sol_time_cont = toc - tic

			if opt['lowest_optimization_result']['status'] != 0:
				print(opt['lowest_optimization_result']['message'])

			# Returns final x value of the continuous optimizer
			x_cont = opt.x
			f_cont = opt.fun
			
			# Apply the function to each element in the input array
			x_ranges = np.array([closest_entries(self.material_list, val, self.n_discr) for val in x_cont])

			# Discrete optimization
			tic = time.perf_counter()
			x_points = list(itertools.product(*x_ranges))
			with mp.Pool(mp.cpu_count()) as pool:
				x_results = pool.map(self.eval_brute_cost, x_points)
			x_discrete = np.array(x_points[np.argmin(x_results)])
			toc = time.perf_counter()
			sol_time_discrete = toc - tic
			f_discrete = self.eval_cost(x_discrete)

			return x_cont, f_cont, x_discrete, f_discrete, x_ranges, sol_time_cont, sol_time_discrete

		print('Starting optimization')
		x_cont, f_cont, x_discrete, f_discrete, x_ranges, sol_time_cont, sol_time_discrete = optimization_step(x_0)

		print('Finished initial optimization, checking nearby region for other minima')
		for k in range(25):
			x_cont_new, f_cont_new, x_discrete_new, f_discrete_new, x_ranges_new, delta_sol_time_cont, delta_sol_time_discrete = optimization_step(x_discrete)

			sol_time_cont += delta_sol_time_cont
			sol_time_discrete += delta_sol_time_discrete

			if f_discrete_new >= f_discrete - 1e-12:
				print('No better local minima found, optimization completed.')
				break
			else:
				x_cont, f_cont, x_discrete, f_discrete, x_ranges = x_cont_new, f_cont_new, x_discrete_new, f_discrete_new, x_ranges_new
				print('Iteration ' + str(k+1) + ': reduced A/R by ' + str(f_discrete_new - f_discrete))
		
		return MultilayerProblemSolution(x_0, x_cont, f_cont, x_discrete, f_discrete, x_ranges, precompile_time, sol_time_cont, sol_time_discrete)

	def visualize_optimization(self, MP_sol, x_0=None, only_discrete=False):
		"""
			Plot the initial, continuous, and discrete optimization solutions for the material parameters across the multilayer dielectric slab structure.
			Grid lines are added to indicate which discrete material parameters were considered during the brute-force step of the optimization procedure.

			Parameters
			----------
			MP_sol : MultilayerProblemSolution
				MultilayerProblemSolution object containing the initial, continuous, and discrete solutions, along with the ranges 
				used for discrete optimization.
			x_0 : array-like, optional
				Initial guess for the material properties across layers. If None, the initial guess stored in `MP_sol` is used.
			only_discrete : bool, default False
				If True, only the discrete solution is plotted. Otherwise, both continuous and discrete solutions are shown.
		"""
		x_mins = [min(MP_sol.x_ranges[0])]
		x_maxs = [max(MP_sol.x_ranges[0])]
		for i in range(1,self.n_lay+1):
			x_mins.append(np.min(MP_sol.x_ranges[i-1:i+1]))
			x_maxs.append(np.max(MP_sol.x_ranges[i-1:i+1]))

		plt.figure(1)
		for i in range(self.n_discr):
			plt.hlines(MP_sol.x_ranges[:,i],np.linspace(0,self.total_t,self.n_lay+1)[:-1],np.linspace(0,self.total_t,self.n_lay+1)[1:], color='0.5', linewidth=0.5, linestyle='dashed',zorder=0)
		
		for i, x in enumerate(np.linspace(0,self.total_t,self.n_lay+1)):
			plt.vlines(x, x_mins[i], x_maxs[i], color='0.5', linewidth=0.5, linestyle='dashed',zorder=0)

		plt.stairs(MP_sol.x_0, np.linspace(0,self.total_t,self.n_lay+1), color='r', linewidth=2.0, label='initial guess')
		if not only_discrete:
			plt.stairs(MP_sol.x_cont, np.linspace(0,self.total_t,self.n_lay+1), color='b', linewidth=2.0, label='continuous solution',zorder=1)
		plt.stairs(MP_sol.x_discrete, np.linspace(0,self.total_t,self.n_lay+1), color='0', linewidth=2.0, label='discrete solution',zorder=2)
		plt.ylim([self.lb - 0.1*(self.ub-self.lb),self.ub + 0.1*(self.ub-self.lb)])
		plt.xlim([-0.1*self.total_t,1.1*self.total_t])
		plt.legend()
		plt.xlabel('Thickness [m]')
		plt.ylabel('CNT concentration [wt%]')
		interactive(True)


class MultilayerProblemSolution:
	"""
		Class for storing the optimization results of a multilayer dielectric slab structure.

		Attributes
		----------
		x_0 : array-like
			Initial guess for the material parameters across layers.
		x_cont : array-like
			Material parameters obtained from the continuous optimization step.
		f_cont : float
			Cost value corresponding to the continuous solution (e.g., maximum R/A ratio).
		x_discrete : array-like
			Material parameters obtained from the discrete optimization step.
		f_discrete : float
			Cost value corresponding to the discrete solution.
		x_ranges : np.ndarray
			Discrete material parameter ranges considered for each layer during brute-force optimization.
		precompile_time : float
			Time taken to precompile the cost function or perform initial computations (seconds).
		sol_time_cont : float
			Time taken for the continuous optimization step (seconds).
		sol_time_discrete : float
			Time taken for the discrete optimization step (seconds).
    """
	def __init__(self, x_0, x_cont, f_cont, x_discrete, f_discrete, x_ranges, precompile_time, sol_time_cont, sol_time_discrete):
		"""
			Initialize an MultilayerProblemSolution instance with all relevant optimization outputs.

			Parameters
			----------
			x_0 : array-like
				Initial guess for the material parameters across layers.
			x_cont : array-like
				Material parameters obtained from the continuous optimization step.
			f_cont : float
				Cost value corresponding to the continuous solution.
			x_discrete : array-like
				Material parameters obtained from the discrete optimization step.
			f_discrete : float
				Cost value corresponding to the discrete solution.
			x_ranges : np.ndarray
				Discrete material parameter ranges considered for each layer.
			precompile_time : float
				Time taken to precompile the cost function or perform initial computations (seconds).
			sol_time_cont : float
				Time taken for the continuous optimization step (seconds).
			sol_time_discrete : float
				Time taken for the discrete optimization step (seconds).
        """
		self.x_0 = x_0
		self.x_cont = x_cont
		self.f_cont = f_cont
		self.x_discrete = x_discrete
		self.f_discrete = f_discrete
		self.x_ranges = x_ranges
		self.precompile_time = precompile_time
		self.sol_time_cont = sol_time_cont
		self.sol_time_discrete = sol_time_discrete

	def __str__(self):
		"""
			Return a human-readable string representation of the optimization results, 
			displayed when using `print()` on the object.
		"""
		np.set_printoptions(precision=3, floatmode='fixed', suppress=True, linewidth=10e5)
		x_0_str = np.array2string(self.x_0, separator=', ')
		x_cont_str = np.array2string(self.x_cont, separator=', ')
		x_discrete_str = np.array2string(self.x_discrete, separator=', ')
		
		return ("-------------------------------\n"
			f"MultilayerProblemSolution:\n"
			"-------------------------------\n"
			"                cost max(R/A)   Material parameters\n"
			f"  x_0:          -               {x_0_str}\n"
			f"  x_cont:       {self.f_cont:.6f}        {x_cont_str}\n"
			f"  x_discrete:   {self.f_discrete:.6f}        {x_discrete_str}\n"
			"-------------------------------\n"
			f"  precompile_time:    {self.precompile_time:.3f}s\n"
			f"  sol_time_cont:      {self.sol_time_cont:.3f}s\n"
			f"  sol_time_discrete:  {self.sol_time_discrete:.3f}s\n"
			"------------------------------\n")

	def __repr__(self):
		"""
			Return a detailed string representation of the object for debugging.
		"""
		np.set_printoptions(precision=6, floatmode='fixed', suppress=True, linewidth=10e5)
		return (f"MultilayerProblemSolution(\n"
			f"  x_0 =        {self.x_0},\n"
			f"  x_cont =     {self.x_cont},\n"
			f"  f_cont =     {self.f_cont},\n"
			f"  x_discrete = {self.x_discrete},\n"
			f"  f_discrete = {self.f_discrete},\n"
			f"  x_ranges =  \n"
			f"{self.x_ranges},\n"
			f"  precompile_time =   {self.precompile_time},\n"
			f"  sol_time_cont =     {self.sol_time_cont},\n"
			f"  sol_time_discrete = {self.sol_time_discrete})")

class MultilayerLearner:
	"""
		Class for learning the permittivity model parameters from experimental measurements.

		This class provides methods to evaluate the error between measured and predicted 
		quantities (transmittance, reflectance, absorption, S-parameters) and to optimize 
		the permittivity model parameters by minimizing these errors.

		Attributes
		----------
		measurements : ExperimentData
			Object containing experimental measurements of the multilayer structures.
		eps_model : PermittivityModel
			Model for the permittivity (dielectric constant) of the material, whose parameters 
			will be optimized.
	"""
	def __init__(self, measurements: ExperimentData, eps_model : PermittivityModel):
		self.measurements = measurements
		self.eps_model = eps_model

	@property
	def n(self):
		"""Total number of measurements."""
		return 501*len(self.measurements.get_names())

	def eval_relative_errors_SET_AR(self, p):
		"""
			Compute the squared relative errors for shielding effectiveness (SET) and 
			absorption-to-reflectance ratio (A/R) across all experiments.

			Parameters
			----------
			p : array-like
				Current permittivity model parameters.

			Returns
			-------
			errors_SET : float
				Sum of squared relative errors for SET across all experiments.
			errors_AR : float
				Sum of squared relative errors for A/R across all experiments.
		"""
		errors_SET, errors_AR = 0, 0

		for experiment in self.measurements.get_names():
			R, T, A = self.measurements.structure[experiment].get_wave_properties(self.measurements.get_material_parameters(experiment), self.measurements.get_X(experiment), self.eps_model, p)
			T_meas, R_meas, A_meas = self.measurements.get_T(experiment), self.measurements.get_R(experiment), self.measurements.get_A(experiment)
			SET_meas = 10*np.log10(1/T_meas)
			AR_meas = A_meas/R_meas

			errors_SET += np.linalg.norm(
					(10*np.log10(1/T) - SET_meas)/SET_meas
				)**2
			errors_AR += np.linalg.norm(
					(A/R - AR_meas)/AR_meas
				)**2

		return errors_SET, errors_AR

	def eval_errors_coefficients(self, p):
		"""
			Compute the squared errors between predicted and measured transmittance (T), 
			reflectance (R), and absorption (A) coefficients across all experiments.

			Parameters
			----------
			p : array-like
				Current permittivity model parameters.

			Returns
			-------
			errors_T : float
				Sum of squared errors for transmittance.
			errors_R : float
				Sum of squared errors for reflectance.
			errors_A : float
				Sum of squared errors for absorption.
		"""
		errors_T, errors_R, errors_A = 0, 0, 0

		for experiment in self.measurements.get_names():
			R, T, A = self.measurements.structure[experiment].get_wave_properties(self.measurements.get_material_parameters(experiment), self.measurements.get_X(experiment), self.eps_model, p)
			errors_T += np.linalg.norm(T - self.measurements.get_T(experiment))**2
			errors_R += np.linalg.norm(R - self.measurements.get_R(experiment))**2
			errors_A += np.linalg.norm(A - self.measurements.get_A(experiment))**2

		return errors_T, errors_R, errors_A

	def eval_cost(self, p, cost_fun='relative-SET-AR'):
		"""
			Evaluate the cost function for the learning problem.

			Parameters
			----------
			p : array-like
				Current permittivity model parameters.
			cost_fun : str, default 'relative-SET-AR'
				Choice of cost function. If 'coeffs', the cost is computed using 
				squared errors on transmittance, reflectance, and absorption. 
				If 'relative-SET-AR', the cost is computed using squared relative 
				errors on shielding effectiveness and absorption-to-reflectance ratio.

			Returns
			-------
			float
				Normalized cost value based on the selected cost function.

			Raises
			------
			NotImplementedError
				If an unsupported cost_fun value is provided.
		"""
		if cost_fun == 'coeffs':
			return sum(self.eval_errors_coefficients(p))/(3*self.n)
		elif cost_fun == 'relative-SET-AR':
			return sum(self.eval_relative_errors_SET_AR(p))/(2*self.n)
		else:
			raise NotImplementedError()

	def optimize(self, x_0, cost_fun='relative-SET-AR'):
		"""
			Optimizes the permittivity model parameters by minimizing the chosen cost function using a basin-hopping algorithm.

			Parameters
			----------
			x_0 : array-like
				Initial guess for the permittivity model parameters.
			cost_fun : str, default 'relative-SET-AR'
				Choice of cost function. If 'coeffs', the cost is based on squared errors
				on transmittance, reflectance, and absorption. If 'relative-SET-AR', the cost
				is based on squared relative errors on shielding effectiveness and absorption-to-reflectance ratio.

			Returns
			-------
			optimal_params : np.ndarray
				Optimized permittivity model parameters that minimize the cost function.
			optimal_cost : float
				Value of the cost function at the optimal parameters.
		"""
		def evaluate_cost_fun(x):
			return self.eval_cost(x, cost_fun)

		print('starting least-squares')
		tic = time.perf_counter()
		opt = basinhopping(
			evaluate_cost_fun, 
			x_0, 
			minimizer_kwargs={
				"bounds":self.eps_model.bounds,
				"options":{"maxiter": 1e4, "disp": False}
			}, 
			niter=10,
			T=0.01
		)
		toc =  time.perf_counter()
		sol_time_cont = toc - tic

		print(opt['lowest_optimization_result']['message'])

		# Returns final x value of the continuous optimizer
		optimal_params = opt.x
		optimal_cost = opt.fun

		return optimal_params, optimal_cost
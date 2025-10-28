import numpy as np
from numba import njit

tolerance = 10**-8

class PermittivityModel:
	p = np.array([], dtype=np.float64)
	p_extra = np.array([], dtype=np.float64)

	"""
		Base class for defining parametric forms of the permittivity (ε) of a material. We use the convention that

		    ε = ε' + iε″, with ε″ < 0 for lossy materials.

		Attributes
		----------
		p : np.ndarray
			Model parameters of the permittivity model.
		p_extra : np.ndarray
			Extra model parameters that may be used in specific models.
		bounds : list of tuple
			Bounds on the model parameters, typically used to ensure physically meaningful values.
    """
	def __init__(self, params=None, bounds=None):
		"""
			Initialize the permittivity model with the given parameters or defaults.

			Parameters
			----------
			params : array-like, optional
				Parameters for the permittivity model. If None, defaults are used from `get_default_params`.
			bounds : array-like of tuple, optional
				Bounds on the model parameters. If None, defaults are used from `get_default_bounds`.
        """
		if params is None:
			params = self.get_default_params()
		self.p = np.array(params)
		
		if bounds is None:
			bounds = self.get_default_bounds()
		self.bounds = bounds

	def get_default_params(self):
		"""Returns the default parameters for the permittivity model."""
		raise NotImplementedError("This method should be implemented by subclasses.")

	def get_default_bounds(self):
		"""Returns the default bounds on the parameters of the permittivity model."""
		return [*[(-np.inf, np.inf)]*len(self.p)]

	@staticmethod
	def _compute_real_permittivity(x, f, p, p_extra):
		"""
			Real part of the permittivity (ε').
			
			Parameters:
			x : float - Material parameter.
			f : float - Frequency of the incoming wave [GHz]
			p : array-like - Model parameters.
			p_extra : array-like - Additional, fixed model parameters
			
			Returns:
			float - Real permittivity (ε')
		"""
		raise NotImplementedError("This method should be implemented by subclasses.")
	
	@staticmethod
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		"""
			Imaginary part of the permittivity (ε'').
			
			Parameters:
			x : float - Material parameter.
			f : float - Frequency of the incoming wave [GHz]
			p : array-like - Model parameters
			p_extra : array-like - Additional, fixed model parameters
			
			Returns:
			float - Imaginary permittivity (ε'')
		"""
		raise NotImplementedError("This method should be implemented by subclasses.")
	
	@staticmethod
	@njit(cache=True)
	def _compute_permittivity(x, f, p, p_extra, real_permittivity, imaginary_permittivity):
		"""
			Compute complex permittivity (ε = ε' + iε'').
			
			Parameters:
			x : array-like - Array of material parameters.
			f : float - Frequency of the incoming wave [GHz]
			p : array-like - Model parameters
			p_extra : array-like - Additional, fixed model parameters
			real_permittivity : callable - Function to compute ε'
			imaginary_permittivity : callable - Function to compute ε''
			
			Returns:
			np.ndarray - Complex permittivity (ε = ε' + iε'')
		"""
		permittivity_list = np.zeros(len(x), dtype=np.complex128)
		for i in range(len(x)):
			real_part = real_permittivity(x[i], f, p, p_extra)
			imag_part = imaginary_permittivity(x[i], f, p, p_extra)
			permittivity_list[i] = real_part + imag_part * 1j
		return permittivity_list
	
	def evaluate_permittivity(self, x, f, p=None, p_extra=None):
		"""
			Evaluate complex permittivity for given inputs.
			
			Parameters:
			x : array-like - Array of material parameters.
			f : float - Frequency of the incoming wave [GHz]
			p : array-like - Model parameters, optional (defaults to self.p)
			p_extra : array-like - Additional, fixed parameters, optional (defaults to self.p_extra)
			
			Returns:
			np.ndarray - Complex permittivity (ε = ε' + iε'')
		"""
		if p is None:
			p = self.p
		else:
			p = np.array(p)
			if len(p) != len(self.p):
				raise ValueError(f"Custom parameter vector must have length {len(self.p)}.")
		if p_extra is None:
			p_extra = self.p_extra
		return self._compute_permittivity(x, f, p, p_extra, self._compute_real_permittivity, self._compute_imaginary_permittivity)


class SimpleLinearPermittivityModel(PermittivityModel):
	"""
		Linear permittivity model.
	"""

	def get_default_params(self):
		return np.array([1.19759538e-15, 3.97364581e-16, -2.64069767e-15, -8.59883062e+00])

	def get_default_bounds(self):
		return [*[(tolerance, np.inf)]*2, *[(-np.inf, -tolerance)]*2]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		return p[0] + p[1]*x

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		return p[2] + p[3]*x

class LinearPermittivityModel(PermittivityModel):
	"""
		Linear permittivity model with frequency dependence.
	"""

	def get_default_params(self):
		return np.array([1.19759538e-15, 3.97364581e-16, 2.71758059e-01, -2.64069767e-15, -8.59883062e+00, -9.66581826e-16])

	def get_default_bounds(self):
		return [*[(tolerance, np.inf)]*3, *[(-np.inf, -tolerance)]*3]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		return p[0] + p[1]*x + p[2]*f

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		return p[3] + p[4]*x + p[5]*f

class SimpleQuadraticPermittivityModel(PermittivityModel):
	"""
		Quadratic permittivity model.
	"""

	def get_default_params(self):
		return np.array([ 2.09924225,  1.50079415,  5.35061839, -0.13599982, -4.30072558, -1.21634402])

	def get_default_bounds(self):
		return [*[(tolerance, np.inf)]*2, *[(-np.inf, np.inf)]*1, *[(-np.inf, -tolerance)]*2, *[(-np.inf, np.inf)]*1]

	# def get_default_bounds(self):
	# 	return [*[(-np.inf, np.inf)]*4, *[(-np.inf, np.inf)]*4]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		return p[0] + p[1]*x + 1/2*p[2]*x**2

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		return p[3] + p[4]*x + 1/2*p[5]*x**2

class QuadraticPermittivityModel(PermittivityModel):
	"""
		Quadratic permittivity model.
	"""

	def get_default_params(self):
		return np.array([ 2.09924225,  1.50079415,  0.17569648,  5.35061839, -0.13599982, -4.30072558, -0.10727507, -1.21634402])

	def get_default_bounds(self):
		return [*[(tolerance, np.inf)]*3, *[(-np.inf, np.inf)]*1, *[(-np.inf, -tolerance)]*3, *[(-np.inf, np.inf)]*1]

	# def get_default_bounds(self):
	# 	return [*[(-np.inf, np.inf)]*4, *[(-np.inf, np.inf)]*4]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		return p[0] + p[1]*x + 1/2*p[3]*x**2 + p[2]*f

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		return p[4] + p[5]*x + 1/2*p[7]*x**2 + p[6]*f


class PiecewisePermittivityModel(PermittivityModel):
	"""
		Piecewise linear permittivity model with linear frequency dependence.
	"""

	@classmethod
	def get_default_params(cls):
		return np.array([5.51007617e-18, 3.47751481e+00, 6.91864652e+01, 2.23849603e-01, -1.18885276e-01, -1.02113460e+01, -1.10159188e+00, -6.25000000e-20])

	def get_default_bounds(self):
		return [*[(tolerance, np.inf)]*4, *[(-np.inf, -tolerance)]*4]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		x_switch = 6
		if x <= x_switch:
			return p[0] + p[1]*x + p[3]*f
		else:
			return p[0] + p[1]*x_switch + p[2]*(x-x_switch) + p[3]*f

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		x_switch = 6
		if x <= x_switch:
			return p[4] + p[5]*x + p[7]*f
		else:
			return p[4] + p[5]*x_switch + p[6]*(x-x_switch) + p[7]*f


class LinearSplinePermittivityModel(PermittivityModel):
	"""
		Linear spline permittivity model.
	"""

	def __init__(self, nodes: np.ndarray, params=None, bounds=None):
		"""Initialize the permittivity model with the given parameters or defaults."""
		self.p_extra = nodes
		super().__init__(params, bounds)

	def get_default_params(self):
		return np.array([1,*[20]*(len(self.p_extra)+1),0,-1,*[-10]*(len(self.p_extra)+1),0])

	def get_default_bounds(self):
		return [
			*[(tolerance, np.inf)]*(len(self.p_extra)+3),
			*[(-np.inf, -tolerance)]*(len(self.p_extra)+3)
			]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		a, b, c = p[0], p[1:len(p_extra)+2], p[len(p_extra)+2]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			a = a + (b[i] - b[i+1])*p_extra[i]
			i += 1
		return a + b[i]*x + c*f

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		delta = len(p_extra)+3
		a, b, c = p[delta], p[delta+1:delta+len(p_extra)+2], p[delta+len(p_extra)+2]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			a = a + (b[i] - b[i+1])*p_extra[i]
			i += 1
		return a + b[i]*x + c*f


class SimpleQuadraticSplinePermittivityModel(PermittivityModel):
	"""
		Quadratic spline permittivity model.
	"""

	def __init__(self, nodes: np.ndarray, params=None, bounds=None):
		"""Initialize the permittivity model with the given parameters or defaults."""
		self.p_extra = nodes
		super().__init__(params, bounds)

	def get_default_params(self):
		return np.array([
			1,10,*[2]*(len(self.p_extra)+1),
			-1,-5,*[-1]*(len(self.p_extra)+1)])

	def get_default_bounds(self):
		return [
			*[(tolerance, np.inf)]*(2),
			*[(-np.inf, np.inf)]*(len(self.p_extra)+1),
			*[(-np.inf, -tolerance)]*(2),
			*[(-np.inf, np.inf)]*(len(self.p_extra)+1),
			]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		a, b, d = p[0], p[1], p[2:len(p_extra)+3]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			grad = b + d[i]*p_extra[i]
			a = a + b*p_extra[i] + 1/2*d[i]*p_extra[i]**2 - grad*p_extra[i] + 1/2*d[i+1]*p_extra[i]**2
			b = grad - d[i+1]*p_extra[i]
			i += 1
		return a + b*x + 1/2*d[i]*x**2

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		delta = len(p_extra)+3
		a, b, d = p[delta], p[delta+1], p[delta+2:delta+len(p_extra)+3]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			grad = b + d[i]*p_extra[i]
			a = a + b*p_extra[i] + 1/2*d[i]*p_extra[i]**2 - grad*p_extra[i] + 1/2*d[i+1]*p_extra[i]**2
			b = grad - d[i+1]*p_extra[i]
			i += 1
		return a + b*x + 1/2*d[i]*x**2

class QuadraticSplinePermittivityModel(PermittivityModel):
	"""
		Quadratic spline permittivity model.
	"""

	def __init__(self, nodes: np.ndarray, params=None, bounds=None):
		"""Initialize the permittivity model with the given parameters or defaults."""
		self.p_extra = nodes
		super().__init__(params, bounds)

	def get_default_params(self):
		return np.array([
			1,10,0,*[2]*(len(self.p_extra)+1),
			-1,-5,0,*[-1]*(len(self.p_extra)+1)])

	def get_default_bounds(self):
		return [
			*[(tolerance, np.inf)]*(3),
			*[(-np.inf, np.inf)]*(len(self.p_extra)+1),
			*[(-np.inf, -tolerance)]*(3),
			*[(-np.inf, np.inf)]*(len(self.p_extra)+1),
			]

	@staticmethod
	@njit(cache=True)
	def _compute_real_permittivity(x, f, p, p_extra):
		a, b, c, d = p[0], p[1], p[2], p[3:len(p_extra)+4]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			grad = b + d[i]*p_extra[i]
			a = a + b*p_extra[i] + 1/2*d[i]*p_extra[i]**2 - grad*p_extra[i] + 1/2*d[i+1]*p_extra[i]**2
			b = grad - d[i+1]*p_extra[i]
			i += 1
		return a + b*x + 1/2*d[i]*x**2 + c*f

	@staticmethod
	@njit(cache=True)
	def _compute_imaginary_permittivity(x, f, p, p_extra):
		delta = len(p_extra)+4
		a, b, c, d = p[delta], p[delta+1], p[delta+2], p[delta+3:delta+len(p_extra)+4]
		i = 0
		while i < len(p_extra) and x > p_extra[i]:
			grad = b + d[i]*p_extra[i]
			a = a + b*p_extra[i] + 1/2*d[i]*p_extra[i]**2 - grad*p_extra[i] + 1/2*d[i+1]*p_extra[i]**2
			b = grad - d[i+1]*p_extra[i]
			i += 1
		return a + b*x + 1/2*d[i]*x**2 + c*f

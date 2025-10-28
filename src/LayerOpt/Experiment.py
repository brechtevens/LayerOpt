import os
import errno
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class ExperimentData:
	"""
		Dataclass for managing and loading experimental multilayer measurement data.

		This class loads experiment metadata (material parameters, multilayer structure, characteristic impedance of the outgoing medium)
		and S-parameter data (S11, S21, S12, S22) from a given directory. It also provides core methods to 
		compute transmission, reflection, and absorption coefficients, as well as derived metrics 
		such as A/R and shielding effectiveness (SET).

		Attributes
		----------
		directory : str
			Path to the directory containing experiment CSV files and the `overview.csv` file.
		reverse : bool
			If True, also considers 'reversed' versions of the experiments by considering not only the forward
			S11 and S21 measurements but also the reverse S22 and S21 measurements. Note that this is only allowed
			when all experiments have free air om both sides of the multilayer stack, i.e., the characteristic
			impedance of the last, outgoing medium is equal to one. 
		wt : dict of {str: np.ndarray}
			Dictionary mapping experiment names to the corresponding list of material parameters of each layer.
		layer_thickness : dict of {str: float}
			Dictionary mapping experiment names to the corresponding layer thickness.
		impedance_out : dict of {str: float}
			Dictionary mapping experiment names to the corresponding characteristic impedance of the final, outgoing medium.
		structure : dict of {str: MultilayerStructure}
			Dictionary mapping experiment names to the corresponding MultilayerStructure.
		X : dict of {str: np.ndarray}
			Dictionary mapping experiment names to the corresponding frequencies.
			Each entry is a 1D NumPy array of frequency points (in GHz) extracted from the corresponding experiment CSV file.
		Y : dict of {str: np.ndarray}
			Dictionary mapping experiment names to the corresponding complex-valued S-parameters, stored as a 2xL array where 
			the first row contains S11 and the second row contains S21.
			Magnitude and phase data from the CSV file (e.g., `db:Trc1_S11`, `ang:Trc1_S11`, etc.) are combined into these complex quantities.
		n:  int
			Number of VNA measurements in the dataset.

		Notes
		-----
		The class expects a file named `overview.csv` in the given directory, which defines the material parameters 
		of each experiment, as well as the layer thickness and optionally the characteristic impedance of the last, outgoing medium. 
		This is typically equal to one (free space) but can, for instance, be set to zero to represent a reflective material.

		Each experiment listed in `overview.csv` should be accompanied by a corresponding CSV file containing S-parameter 
		data with columns named `db:Trc...` and `ang:Trc...` measured by a vector network analyzer. The files are automatically parsed, converted 
		into complex S-parameters, and stored for analysis.

		Examples
		--------
		Load all experiments from a directory:

		>>> data = ExperimentData(directory='../measurements/')
		>>> print(data.get_names())
		dict_keys(['sample1', 'sample2', 'sample3'])

		Retrieve and plot A/R versus shielding effectiveness for one experiment:

		>>> data.plot_behavior('sample1')

		Get computed coefficients:

		>>> T = data.get_T('sample1')
		>>> R = data.get_R('sample1')
		>>> A = data.get_A('sample1')
	"""
	directory: str
	reverse: bool = False  # Add reverse flag to control reverse data loading
	wt: dict = field(default_factory=dict)
	layer_thickness: dict = field(default_factory=dict)
	impedance_in: dict = field(default_factory=dict)
	impedance_out: dict = field(default_factory=dict)
	structure: dict = field(default_factory=dict)
	X: dict = field(default_factory=dict)
	Y: dict = field(default_factory=dict)
	n: int = 0

	def __post_init__(self):
		"""Automatically load all experiment data upon initialization."""
		self.load_experiment_data()

	def load_overview(self):
		"""
			Load the material parameter data and experiment setup metadata from the file `overview.csv`.

			This method reads a CSV file that defines the material parameters, layer thickness,
			and optionally the outgoing medium impedance for each experiment. It also 
			instantiates a `MultilayerStructure` object for each experiment.

			Raises
			------
			FileNotFoundError
				If `overview.csv` cannot be found in the specified directory.
			NotImplementedError
				If `reverse = True` but one of the experiments in `overview.csv` has a characteristic impedance of the last, outgoing medium which is not equal to one.
			TODO: warning
		"""
		from .core import MultilayerStructure

		if os.path.exists(os.path.normpath(os.path.join(self.directory, 'overview.csv'))):
			df = pd.read_csv(os.path.normpath(os.path.join(self.directory, 'overview.csv')))

			# Convert weight strings to numpy arrays and store them
			for index, row in df.iterrows():
				self.wt[row['experiment_name']] = np.array(row['parameters'].strip('[]').split(','), dtype=float)

				optional_fields = {
					'layer_thickness': 0.005,
					'impedance_out': 1
				}

				for field_name, default in optional_fields.items():
					value = row.get(field_name, np.nan)
					value = default if np.isnan(value) else value
					attr_dict = getattr(self, field_name)
					attr_dict[row['experiment_name']] = value

				if self.reverse and self.impedance_out[row['experiment_name']] != 1.0:
					raise NotImplementedError("The option `reverse = True` is only allowed when the experiments have free space on both sides of the madium, i.e., when the characteristic impedance of the last, outgoing medium is equal to one")
				
				self.structure[row['experiment_name']] = MultilayerStructure(t = self.layer_thickness[row['experiment_name']], n_lay = len(self.wt[row['experiment_name']]), impedance_out = self.impedance_out[row['experiment_name']])
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.normpath(os.path.join(self.directory, 'overview.csv')))

	def load_experiment(self, filename):
		"""
			Load S-parameter data for a single experiment.

			This method parses a CSV file containing frequency-dependent S-parameter
			magnitude (in dB) and phase (in degrees) data measured by a vector network analyzer.
			The complex-valued coefficients (S11, S21, S12, S22) are reconstructed and stored.
			If the `reverse` flag is set, reversed measurements are also added (S22, S12), with the string
			"_reverse" added to the original experiment name.

			Parameters
			----------
			filename : str
				Name of the CSV file containing the experiment data.
			
			Raises
			------
			NotImplementedError
				If the CSV file contains unsupported column names.
		"""
		file_path = os.path.normpath(os.path.join(self.directory, filename))
		experiment_name = filename.split('.')[0]

		# Load the CSV file
		data = pd.read_csv(file_path, skiprows=2, delimiter='\t', usecols=range(9))

		filtered_magnitudes = {}
		filtered_angles = {}
		for key in data.keys():
			if key == 'freq[Hz]':
				X_freq = data['freq[Hz]'].values * 1e-9  # Convert frequency to GHz
				self.n += len(data['freq[Hz]'])
			elif key[:6] == 'db:Trc':
				filtered_magnitudes[key[-3:]] = data[key].values
			elif key[:7] == 'ang:Trc':
				filtered_angles[key[-3:]] = data[key].values
			else:
				raise NotImplementedError("The provided CSV file contains keys which are not compatible. Only the key 'freq[Hz]' and keys ending in 'db:Trc' and 'and:Trc' are supported.")
			
		def get_S(db, angle):
			return (10**(db/20)) * np.exp(1j*angle*np.pi/180)

		S11, S12, S21, S22 = [get_S(filtered_magnitudes[key], filtered_angles[key]) for key in ['S11', 'S12', 'S21', 'S22']]

		# Store the frequency and coefficient data in the class dictionaries
		self.X[experiment_name] = X_freq
		self.Y[experiment_name] = np.array([S11, S21])

		# Store reverse data if reverse flag is enabled
		if self.reverse:
			reverse_experiment_name = experiment_name + "_reverse"
			self.wt[reverse_experiment_name] = np.flip(self.wt[experiment_name])
			self.X[reverse_experiment_name] = X_freq
			self.Y[reverse_experiment_name] = np.array([S22, S12])

	def load_experiment_data(self):
		"""
			Load all experiment data from the given directory.

			This method first loads the material parameter data and experiment setup metadata from `overview.csv`,
			then iterates over all other CSV files in the directory to load S-parameter data for each experiment.
		"""
		self.load_overview()

		# Loop through all files in the directory and load experiment data (excluding the weights file)
		for filename in os.listdir(self.directory):
			if filename.endswith('.csv') and filename != 'overview.csv':
				self.load_experiment(filename)

	def get_names(self):
		"""
			Returns the names of all loaded experiments.

			Returns
			-------
			list of str
				Names of all loaded experiments.
		"""
		return self.wt.keys()

	def get_material_parameters(self, experiment_name):
		"""
			Returns the material parameters for a specific experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Array of material parameters associated with the experiment.
		"""
		return self.wt.get(experiment_name)

	def get_X(self, experiment_name):
		"""
			Returns the frequency vector for a specific experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				1D array of frequency values in GHz.
		"""
		return self.X.get(experiment_name)

	def get_Y(self, experiment_name):
		"""
			Returns the measured S11- and S21-parameters for a specific experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				2xL array of complex S-parameters where the first row corresponds to S11 
				and the second row corresponds to S21.

			Notes
			-----
				For '_reverse' experiments, the first row corresponds to the S22- and the second
				to the S21-parameters of the original (non-reversed) experiment.
		"""
		return self.Y.get(experiment_name)

	def get_S11(self, experiment_name):
		"""
			Returns the measured S11-parameters for a specific experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Array of complex S11-parameters.
		"""
		return self.get_Y(experiment_name)[0, :]

	def get_S21(self, experiment_name):
		"""
			Returns the measured S21-parameters for a specific experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Array of complex S21-parameters.
		"""
		return self.get_Y(experiment_name)[1, :]

	def get_T(self, experiment_name):
		"""
			Compute the transmittance coefficient for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Transmittance values calculated as :math:`|S_{21}|^2`.
		"""
		return abs(self.get_S21(experiment_name))**2

	def get_R(self, experiment_name):
		"""
			Compute the reflectance coefficient for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Reflectance values calculated as :math:`|S_{11}|^2`.
		"""
		return abs(self.get_S11(experiment_name))**2

	def get_A(self, experiment_name):
		"""
			Compute the absorptance coefficient for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			np.ndarray
				Absorptance values calculated as 1 - T - R.
		"""
		return 1 - self.get_T(experiment_name) - self.get_R(experiment_name)

	def plot_behavior(self, experiment_name):
		"""
			Plot the measured A/R ratio versus shielding effectiveness (SET) for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment to plot.
		"""
		AR, SET = self.get_AR_SET(experiment_name)

		## plot response
		dir = self.directory.split('..')[1]
		plt.plot(SET, AR, marker='.', linestyle="", label=dir + experiment_name)
		plt.ylabel("A/R")
		plt.xlabel("Shielding effectiveness [dB]")
		plt.legend()

		current_ylim = plt.gca().get_ylim()
		if current_ylim[1] > 250:
			plt.ylim(bottom=-5, top=250)
		
	def plot_coefficients(self, experiment_name):
		"""
			Plot the measured reflectance (R) versus transmittance (T) for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment to plot.
		"""
		T = self.get_T(experiment_name)
		R = self.get_R(experiment_name)

		## plot response
		dir = self.directory.split('..')[1]
		plt.plot(T, R, marker='d', linestyle="", label=dir + experiment_name)
		plt.ylabel("R")
		plt.xlabel("T")
		plt.legend()

	def get_unique_experiments(self):
		"""
			Group experiments by their material parameters.

			Returns
			-------
			dict
				Dictionary mapping tuples of material parameters to lists of experiment names
				that share those material parameters.
		"""
		reverse_dict = defaultdict(list)
		for key, value in self.wt.items():
			reverse_dict[tuple(value)].append(key)
		return reverse_dict

	def get_duplicate_experiments(self):
		"""
			Find experiments that occur more than once.

			Returns
			-------
			dict
				Dictionary mapping tuples of material parameters to lists of experiment names
				that have identical material parameters (i.e., appear more than once).

			Notes
			-----
				This function only considers the material parameters. An experiment is 
				considered a `real duplicate` if the layer thickness and the characteristic 
				impedance of the outgoing medium are also identical.
		"""
		return {value: keys for value, keys in self.get_unique_experiments().items() if len(keys) > 1}

	def get_AR_SET(self, experiment_name):
		"""
			Compute the A/R ratio and shielding effectiveness (SET) for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			tuple of np.ndarray
				A/R ratio and shielding effectiveness (in dB), respectively.
		"""
		T = self.get_T(experiment_name)
		R = self.get_R(experiment_name)
		A = self.get_A(experiment_name)
		return A/R, 10*np.log10(1/T)

	def get_worst_case_AR_SET(self, experiment_name):
		"""
			Compute the worst-case (minimum) values of A/R and SET for a given experiment.

			Parameters
			----------
			experiment_name : str
				Name of the experiment.

			Returns
			-------
			tuple of float
				Minimum A/R ratio and minimum shielding effectiveness (in dB) across all measured frequencies.
		"""
		AR, SET = self.get_AR_SET(experiment_name)
		return np.min(AR), np.min(SET)
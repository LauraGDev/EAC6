"""
@ IOC - CE IABD
"""
import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score

class TestGenerarDataset(unittest.TestCase):
	"""
	classe TestGenerarDataset
	"""
	global mu_p_be
	global mu_p_me
	global mu_b_bb
	global mu_b_mb
	global sigma
	global dicc

	mu_p_be = 3240 # mitjana temps pujada bons escaladors
	mu_p_me = 4268 # mitjana temps pujada mals escaladors
	mu_b_bb = 1440 # mitjana temps baixada bons baixadors
	mu_b_mb = 2160 # mitjana temps baixada mals baixadors
	sigma = 240 # 240 s = 4 min

	dicc = [
		{"name":"BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
		{"name":"MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
		{"name":"MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
	]

	def test_longituddataset(self):
		"""
		Test la longitud de l'array
		"""
		arr = generar_dataset(200, 1, dicc[0])
		self.assertEqual(len(arr), 200)

	def test_valorsmitjatp(self):
		"""
		Test del valor mitjà del tp
		"""

		arr = generar_dataset(100, 1, dicc[0])
		#assertLess(a, b)
		#assertGreater(a, b)
		#arr.append(generar_dataset(100, 1, dicc[3]))
		arr_tp = [row[1] for row in arr] # la columna tp és la segona
		tp_mig = sum(arr_tp)/len(arr_tp)
		self.assertLess(tp_mig, 3400)

	def test_valorsmitjatb(self):
		"""
		Test del valor mitjà del tp
		"""
		arr = generar_dataset(100, 1, dicc[1])
		arr_tb = [row[1] for row in arr] # la columna tp és la segona
		tb_mig = sum(arr_tb)/len(arr_tb)
		self.assertGreater(tb_mig, 2000)

class TestClustersCiclistes(unittest.TestCase):
	"""
	classe TestClustersCiclistes
	"""
	global ciclistes_data_clean
	global data_labels

	path_dataset = './data/ciclistes.csv'
	ciclistes_data = load_dataset(path_dataset)
	ciclistes_data_clean = clean(ciclistes_data)
	true_labels = extract_true_labels(ciclistes_data_clean)
	ciclistes_data_clean = ciclistes_data_clean.drop('tipus', axis=1) # eliminem el tipus, ja no interessa

	clustering_model = clustering_kmeans(ciclistes_data_clean)
	with open('model/clustering_model.pkl', 'wb') as f:
		pickle.dump(clustering_model, f)
	data_labels = clustering_model.labels_

	def test_check_column(self):
		"""
		Comprovem que una columna existeix
		"""

		self.assertIn('tp', ciclistes_data_clean.columns)

	def test_data_labels(self):
		"""
		Comprovem que data_labels té la mateixa longitud que ciclistes
		"""

		self.assertEqual(len(data_labels), len(ciclistes_data_clean))

	def test_model_saved(self):
		"""
		Comprovem que a la carpeta model/ hi ha els fitxer clustering_model.pkl
		"""
		check_file = os.path.isfile('./model/clustering_model.pkl')
		self.assertTrue(check_file)

if __name__ == '__main__':
	unittest.main()

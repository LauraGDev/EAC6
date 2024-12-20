""" @ IOC - Joan Quintana - 2024 - CE IABD """

import sys
import logging
import shutil
import mlflow

from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score


if __name__ == "__main__":

	# TODO
	print('s\'han generat els runs')

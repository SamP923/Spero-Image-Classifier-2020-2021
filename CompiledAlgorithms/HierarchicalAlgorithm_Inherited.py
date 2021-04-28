import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
import webcolors
import os

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from collections import Counter
from datetime import datetime

from .ClusteringAlgorithm import ClusteringAlgorithm
from .ENVI_Files import Envi
from .elbowMethod import Elbow

import time
import sys

# def time_difference(last_time):
#     return str(round(time.time() - last_time))

class DominantColorsHier(ClusteringAlgorithm):

    def cluster(self, img):
        # Run the Scikit-Learn algorithm with the determined number of clusters
        hierarchical = AgglomerativeClustering(n_clusters=self.CLUSTERS)
        # hierarchical = AgglomerativeClustering(n_clusters=self.CLUSTERS, linkage='ward', connectivity=knn_graph)
        hierarchical.fit(img)
        self.LABELS = hierarchical.labels_
        self.make_avg_labels()


    def findDominant(self):
        img = super().findDominant()
        try:
            self.cluster(img)
        except NameError:
            print("not found")
        self.plot()
        return self.RESULT_PATH
    
    def log(self, message):
        sys.stdout.write(message + "\n")

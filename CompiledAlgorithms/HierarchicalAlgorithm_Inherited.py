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
        self.CENTROIDS = self.find_centers()


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

    def make_avg_labels(self):

        st = time.time()
        self.log("Calculating cluster averages...")

        # Get amount of wavelengths per pixel
        wavelengths = len(self.IMAGE[0])

        # Create list to store the avg values for each cluster, for each wavelength
        avg_labels = np.zeros((self.CLUSTERS, wavelengths))

        index = 0  # Store current position in image pixels
        # Loop through the label list
        for label in self.LABELS:
            for x in range(wavelengths):
                # Add the corresponding IMAGE pixel values to the sum
                avg_labels[label][x] += self.IMAGE[index][x]
            index += 1

        # Count the occurrences of each label, and store them in a list
        x = Counter(sorted(self.LABELS))
        values = np.array(list(x.values()))

        # Divide each sum by its label total
        for label in range(self.CLUSTERS):
            avg_labels[label] /= values[label]

        # Round to 2 decimals
        avg_labels = avg_labels.round(2)
        self.CENTROIDS = avg_labels
        self.log("Cluster averages calculated")

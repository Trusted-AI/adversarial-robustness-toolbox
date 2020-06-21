# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.detector.poison import ClusteringAnalyzer

logger = logging.getLogger(__name__)


class TestActivationDefence(unittest.TestCase):
    def test_size_analyzer(self):
        nb_clusters = 2
        nb_classes = 3
        clusters_by_class = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        clusters_by_class[0] = [0, 1, 1, 1, 1]  # Class 0
        clusters_by_class[1] = [1, 0, 0, 0, 0]  # Class 1
        clusters_by_class[2] = [0, 0, 0, 0, 1]  # Class 2
        analyzer = ClusteringAnalyzer()
        assigned_clean_by_class, poison_clusters, report = analyzer.analyze_by_size(clusters_by_class)

        # print("clusters_by_class")
        # print(clusters_by_class)
        # print("assigned_clean_by_class")
        # print(assigned_clean_by_class)
        # print("poison_clusters")
        # print(poison_clusters)

        clean = 0
        poison = 1
        # For class 0, cluster 0 should be marked as poison.
        self.assertEqual(poison_clusters[0][0], poison)
        # For class 0, cluster 1 should be marked as clean.
        self.assertEqual(poison_clusters[0][1], clean)
        self.assertEqual(report["Class_0"]["cluster_0"]["suspicious_cluster"], True)
        self.assertEqual(report["Class_0"]["cluster_1"]["suspicious_cluster"], False)
        total = len(clusters_by_class[0])
        c1 = sum(clusters_by_class[0])
        self.assertEqual(report["Class_0"]["cluster_0"]["ptc_data_in_cluster"], (total - c1) / total)
        self.assertEqual(report["Class_0"]["cluster_1"]["ptc_data_in_cluster"], c1 / total)

        # Inverse relations for class 1
        self.assertEqual(poison_clusters[1][0], clean)
        self.assertEqual(poison_clusters[1][1], poison)
        self.assertEqual(report["Class_1"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_1"]["cluster_1"]["suspicious_cluster"], True)
        total = len(clusters_by_class[1])
        c1 = sum(clusters_by_class[1])
        self.assertEqual(report["Class_1"]["cluster_0"]["ptc_data_in_cluster"], (total - c1) / total)
        self.assertEqual(report["Class_1"]["cluster_1"]["ptc_data_in_cluster"], c1 / total)

        self.assertEqual(poison_clusters[2][0], clean)
        self.assertEqual(poison_clusters[2][1], poison)
        self.assertEqual(report["Class_2"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_2"]["cluster_1"]["suspicious_cluster"], True)
        total = len(clusters_by_class[2])
        c1 = sum(clusters_by_class[2])
        self.assertEqual(report["Class_2"]["cluster_0"]["ptc_data_in_cluster"], (total - c1) / total)
        self.assertEqual(report["Class_2"]["cluster_1"]["ptc_data_in_cluster"], c1 / total)

        poison = 0
        self.assertEqual(assigned_clean_by_class[0][0], poison)
        self.assertEqual(assigned_clean_by_class[1][0], poison)
        self.assertEqual(assigned_clean_by_class[2][4], poison)

    def test_size_analyzer_three(self):
        nb_clusters = 3
        nb_classes = 3
        clusters_by_class = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        clusters_by_class[0] = [0, 1, 1, 2, 2]  # Class 0
        clusters_by_class[1] = [1, 0, 0, 2, 2]  # Class 1
        clusters_by_class[2] = [0, 0, 0, 2, 1, 1]  # Class 2
        analyzer = ClusteringAnalyzer()
        assigned_clean_by_class, poison_clusters, report = analyzer.analyze_by_size(clusters_by_class)

        # print("clusters_by_class")
        # print(clusters_by_class)
        # print("assigned_clean_by_class")
        # print(assigned_clean_by_class)
        # print("poison_clusters")
        # print(poison_clusters)

        clean = 0
        poison = 1
        # For class 0, cluster 0 should be marked as poison.
        self.assertEqual(poison_clusters[0][0], poison)
        # For class 0, cluster 1 and 2 should be marked as clean.
        self.assertEqual(poison_clusters[0][1], clean)
        self.assertEqual(poison_clusters[0][2], clean)
        self.assertEqual(report["Class_0"]["cluster_0"]["suspicious_cluster"], True)
        self.assertEqual(report["Class_0"]["cluster_1"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_0"]["cluster_2"]["suspicious_cluster"], False)
        total = len(clusters_by_class[0])
        counts = np.bincount(clusters_by_class[0])
        self.assertEqual(report["Class_0"]["cluster_0"]["ptc_data_in_cluster"], counts[0] / total)
        self.assertEqual(report["Class_0"]["cluster_1"]["ptc_data_in_cluster"], counts[1] / total)
        self.assertEqual(report["Class_0"]["cluster_2"]["ptc_data_in_cluster"], counts[2] / total)

        self.assertEqual(poison_clusters[1][0], clean)
        self.assertEqual(poison_clusters[1][1], poison)
        self.assertEqual(poison_clusters[1][2], clean)
        self.assertEqual(report["Class_1"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_1"]["cluster_1"]["suspicious_cluster"], True)
        self.assertEqual(report["Class_1"]["cluster_2"]["suspicious_cluster"], False)
        total = len(clusters_by_class[1])
        counts = np.bincount(clusters_by_class[1])
        self.assertEqual(report["Class_1"]["cluster_0"]["ptc_data_in_cluster"], counts[0] / total)
        self.assertEqual(report["Class_1"]["cluster_1"]["ptc_data_in_cluster"], counts[1] / total)
        self.assertEqual(report["Class_1"]["cluster_2"]["ptc_data_in_cluster"], counts[2] / total)

        self.assertEqual(poison_clusters[2][0], clean)
        self.assertEqual(poison_clusters[2][1], clean)
        self.assertEqual(poison_clusters[2][2], poison)
        self.assertEqual(report["Class_2"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_2"]["cluster_1"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_2"]["cluster_2"]["suspicious_cluster"], True)
        total = len(clusters_by_class[2])
        counts = np.bincount(clusters_by_class[2])
        self.assertEqual(report["Class_2"]["cluster_0"]["ptc_data_in_cluster"], round(counts[0] / total, 2))
        self.assertEqual(report["Class_2"]["cluster_1"]["ptc_data_in_cluster"], round(counts[1] / total, 2))
        self.assertEqual(report["Class_2"]["cluster_2"]["ptc_data_in_cluster"], round(counts[2] / total, 2))

        poison = 0
        self.assertEqual(assigned_clean_by_class[0][0], poison)
        self.assertEqual(assigned_clean_by_class[1][0], poison)
        self.assertEqual(assigned_clean_by_class[2][3], poison)

    def test_relative_size_analyzer(self):
        nb_clusters = 2
        nb_classes = 4
        clusters_by_class = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        clusters_by_class[0] = [0, 1, 1, 1, 1]  # Class 0
        clusters_by_class[1] = [1, 0, 0, 0, 0]  # Class 1
        clusters_by_class[2] = [0, 0, 0, 0, 1]  # Class 2
        clusters_by_class[3] = [0, 0, 1, 1, 1]  # Class 3
        analyzer = ClusteringAnalyzer()
        assigned_clean_by_class, poison_clusters, report = analyzer.analyze_by_relative_size(clusters_by_class)

        # print("clusters_by_class")
        # print(clusters_by_class)
        # print("assigned_clean_by_class")
        # print(assigned_clean_by_class)
        # print("poison_clusters")
        # print(poison_clusters)

        clean = 0
        poison = 1
        # For class 0, cluster 0 should be marked as poison.
        self.assertEqual(poison_clusters[0][0], poison)
        # For class 0, cluster 1 should be marked as clean.
        self.assertEqual(poison_clusters[0][1], clean)
        self.assertEqual(report["Class_0"]["cluster_0"]["suspicious_cluster"], True)
        self.assertEqual(report["Class_0"]["cluster_1"]["suspicious_cluster"], False)
        total = len(clusters_by_class[0])
        c1 = sum(clusters_by_class[0])
        self.assertEqual(report["Class_0"]["cluster_0"]["ptc_data_in_cluster"], round((total - c1) / total, 2))
        self.assertEqual(report["Class_0"]["cluster_1"]["ptc_data_in_cluster"], round(c1 / total, 2))

        # Inverse relations for class 1
        self.assertEqual(poison_clusters[1][0], clean)
        self.assertEqual(poison_clusters[1][1], poison)
        self.assertEqual(report["Class_1"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_1"]["cluster_1"]["suspicious_cluster"], True)
        total = len(clusters_by_class[1])
        c1 = sum(clusters_by_class[1])
        self.assertEqual(report["Class_1"]["cluster_0"]["ptc_data_in_cluster"], round((total - c1) / total, 2))
        self.assertEqual(report["Class_1"]["cluster_1"]["ptc_data_in_cluster"], round(c1 / total, 2))

        self.assertEqual(poison_clusters[2][0], clean)
        self.assertEqual(poison_clusters[2][1], poison)
        self.assertEqual(report["Class_2"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_2"]["cluster_1"]["suspicious_cluster"], True)
        total = len(clusters_by_class[2])
        c1 = sum(clusters_by_class[2])
        self.assertEqual(report["Class_2"]["cluster_0"]["ptc_data_in_cluster"], round((total - c1) / total, 2))
        self.assertEqual(report["Class_2"]["cluster_1"]["ptc_data_in_cluster"], round(c1 / total, 2))

        self.assertEqual(poison_clusters[3][0], clean)
        self.assertEqual(poison_clusters[3][1], clean)
        self.assertEqual(report["Class_3"]["cluster_0"]["suspicious_cluster"], False)
        self.assertEqual(report["Class_3"]["cluster_1"]["suspicious_cluster"], False)
        total = len(clusters_by_class[3])
        c1 = sum(clusters_by_class[3])
        self.assertEqual(report["Class_3"]["cluster_0"]["ptc_data_in_cluster"], round((total - c1) / total, 2))
        self.assertEqual(report["Class_3"]["cluster_1"]["ptc_data_in_cluster"], round(c1 / total, 2))

        poison = 0
        self.assertEqual(assigned_clean_by_class[0][0], poison)
        self.assertEqual(assigned_clean_by_class[1][0], poison)
        self.assertEqual(assigned_clean_by_class[2][4], poison)
        self.assertEqual(sum(assigned_clean_by_class[3]), len(assigned_clean_by_class[3]))

    @unittest.expectedFailure
    def test_relative_size_analyzer_three(self):
        nb_clusters = 3
        nb_classes = 3
        clusters_by_class = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        clusters_by_class[0] = [0, 1, 1, 2, 2]  # Class 0
        clusters_by_class[1] = [1, 0, 0, 2, 2]  # Class 1
        clusters_by_class[2] = [0, 0, 0, 2, 1, 1]  # Class 2
        analyzer = ClusteringAnalyzer()
        analyzer.analyze_clusters(clusters_by_class)


if __name__ == "__main__":
    unittest.main()

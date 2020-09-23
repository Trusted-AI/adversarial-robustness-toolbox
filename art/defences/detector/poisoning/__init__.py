"""
Module implementing detector-based defences against poisoning attacks.
"""
from art.defences.detector.poisoning.poison_filtering_defence import PoisonFilteringDefence
from art.defences.detector.poisoning.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poisoning.activation_defence import ActivationDefence
from art.defences.detector.poisoning.clustering_analyzer import ClusteringAnalyzer
from art.defences.detector.poisoning.provenance_defense import ProvenanceDefense
from art.defences.detector.poisoning.roni import RONIDefense
from art.defences.detector.poisoning.spectral_signature_defense import SpectralSignatureDefense

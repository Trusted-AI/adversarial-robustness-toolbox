from enum import Enum

class ReducerType(Enum):
    PCA = "PCA"
    UMAP = "UMAP"
    FASTICA = "FastICA"

class ClustererType(Enum):
    DBSCAN = "DBSCAN"

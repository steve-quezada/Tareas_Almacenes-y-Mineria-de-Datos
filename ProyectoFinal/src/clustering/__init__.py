"""
Paquete de agrupamiento no supervisado:

- AccidentClusteringPreprocessor: preparación de las variables de entrada.
- ClusteringStrategy y sus implementaciones (KMeans, Aglomerativo, DBSCAN)
- ClusterAnalyzer: orquestador de la exploración y evaluación.
- ClusteringModel: artefacto para persistir y reutilizar el modelo final.
"""

from .analyzer import ClusterAnalyzer
from .model import ClusteringModel
from .preprocessor import AccidentClusteringPreprocessor
from .strategies import (
    AgglomerativeStrategy,
    ClusteringStrategy,
    DBSCANStrategy,
    KMeansStrategy,
)

__all__ = [
    "AccidentClusteringPreprocessor",
    "ClusteringStrategy",
    "KMeansStrategy",
    "AgglomerativeStrategy",
    "DBSCANStrategy",
    "ClusterAnalyzer",
    "ClusteringModel",
]

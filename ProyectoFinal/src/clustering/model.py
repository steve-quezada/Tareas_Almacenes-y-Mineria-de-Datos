"""
model.py

Artefacto del clustering.

`ClusteringModel` empaqueta, el preprocesador ajustado y el modelo K-Means entrenado. 
Se guarda con `joblib`, puede recargarse para asignar el perfil de accidentes nuevos
sin necesidad de re-entrenar.

El objeto guardado contiene todo lo necesario para transformar datos crudos y 
predecir su grupo.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessor import AccidentClusteringPreprocessor
from .strategies import KMeansStrategy


class ClusteringModel:
    """
    Modelo de clustering.

    Combina por composición un `AccidentClusteringPreprocessor` ajustado y una
    estrategia K-Means entrenada. Una vez ajustado, puede:

    - asignar el cluster de registros nuevos (`predict`),
    - guardarse en disco (`save`),
    - recargarse sin re-entrenar (`load`).

    Attributes
    ----------
    preprocessor : AccidentClusteringPreprocessor
        Preprocesador ajustado sobre los datos de entrenamiento.
    strategy : KMeansStrategy
        Estrategia K-Means con el estimador entrenado.
    cluster_labels : dict or None
        Mapa opcional de id de cluster a etiqueta interpretativa.
    """

    def __init__(
        self,
        preprocessor: AccidentClusteringPreprocessor,
        strategy: KMeansStrategy,
        cluster_labels: dict | None = None,
    ):
        self.preprocessor = preprocessor
        self.strategy = strategy
        self.cluster_labels = cluster_labels or {}

    @classmethod
    def train(
        cls,
        df: pd.DataFrame,
        n_clusters: int,
        random_state: int = 42,
    ) -> "ClusteringModel":
        """
        Entrena un modelo de clustering completo desde el dataset limpio.

        Ajusta el preprocesador y entrena K-Means sobre los datos transformados.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset limpio de accidentes.
        n_clusters : int
            Número de grupos elegido.
        random_state : int
            Semilla aleatoria fija.

        Returns
        -------
        ClusteringModel
            Modelo entrenado, listo para predecir o guardarse.
        """
        preprocessor = AccidentClusteringPreprocessor()
        matrix = preprocessor.fit_transform(df)

        strategy = KMeansStrategy(n_clusters=n_clusters, random_state=random_state)
        strategy.fit_predict(matrix)

        return cls(preprocessor=preprocessor, strategy=strategy)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Asigna el cluster a cada fila de un DataFrame nuevo.

        Parameters
        ----------
        df : pd.DataFrame
            Registros nuevos con las columnas de entrada esperadas.

        Returns
        -------
        numpy.ndarray
            Id de cluster asignado a cada fila.
        """
        matrix = self.preprocessor.transform(df)
        return self.strategy.estimator_.predict(matrix)

    def predict_labels(self, df: pd.DataFrame) -> list[str]:
        """
        Igual que `predict`, pero devuelve la etiqueta interpretativa del
        cluster si se configuró `cluster_labels`.
        """
        ids = self.predict(df)
        return [self.cluster_labels.get(int(c), f"Cluster {int(c)}") for c in ids]

    def set_cluster_labels(self, labels: dict) -> None:
        """Asigna nombres interpretativos a los ids de cluster."""
        self.cluster_labels = {int(k): v for k, v in labels.items()}

    @property
    def n_clusters(self) -> int:
        """Número de grupos del modelo entrenado."""
        return self.strategy.n_clusters

    def save(self, path: str) -> None:
        """
        Guarda el modelo completo (preprocesador + K-Means) con joblib.

        Parameters
        ----------
        path : str
            Ruta destino, por ejemplo 'models/clustering_kmeans.joblib'.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str) -> "ClusteringModel":
        """
        Recarga un modelo previamente guardado con `save`.

        Soporta dos formatos:
        - Formato actual: un objeto ``ClusteringModel`` serializado.
        - Formato legacy (dict): ``{'kmeans': KMeans, 'preprocessor': ColumnTransformer}``,
          producido por versiones anteriores del notebook antes de adoptar esta clase.

        Parameters
        ----------
        path : str
            Ruta al archivo .joblib.

        Returns
        -------
        ClusteringModel
            Modelo listo para predecir, sin necesidad de re-entrenar.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe.
        """
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"No se encontró el modelo guardado en: {source}")
        obj = joblib.load(source)

        # Compatibilidad con el formato dict anterior del notebook
        if isinstance(obj, dict) and "kmeans" in obj and "preprocessor" in obj:
            strategy = KMeansStrategy(n_clusters=obj["kmeans"].n_clusters)
            strategy.estimator_ = obj["kmeans"]
            strategy.labels_ = obj["kmeans"].labels_
            preprocessor = AccidentClusteringPreprocessor()
            preprocessor._column_transformer = obj["preprocessor"]
            preprocessor._fitted = True
            return cls(preprocessor=preprocessor, strategy=strategy)

        return obj

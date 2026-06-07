"""
strategies.py

Patrón de diseño Strategy.

Cada algoritmo de clustering se encapsula en una clase que comparte una interfaz 
común `ClusteringStrategy`. 
Esto permite intercambiar el algoritmo en tiempo de ejecución sin modificar el
código cliente (`ClusterAnalyzer`), que solo conoce la interfaz abstracta.

Justificación del patrón
------------------------
El clustering es un problema exploratorio en el que conviene probar y
comparar varios algoritmos bajo las mismas condiciones de preprocesamiento.
Strategy hace que esa comparación sea limpia: el analizador pide
`fit_predict` a la estrategia activa sin saber cuál es, y se puede cambiar de
K-Means a DBSCAN con una sola línea.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


class ClusteringStrategy(ABC):
    """
    Interfaz común para todas las estrategias de clustering.

    Las subclases envuelven un estimador de scikit-learn y exponen una API
    uniforme: `fit_predict`, `name` y `get_params`.

    Attributes
    ----------
    random_state : int
        Semilla aleatoria fija para garantizar reproducibilidad.
    estimator_ : object or None
        Estimador de scikit-learn ajustado tras llamar a `fit_predict`.
    labels_ : numpy.ndarray or None
        Etiquetas de grupo asignadas en el último ajuste.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.estimator_ = None
        self.labels_ = None

    @abstractmethod
    def build_estimator(self):
        """Construye y devuelve el estimador de scikit-learn correspondiente."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre legible del algoritmo."""
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta el algoritmo sobre `X` y devuelve las etiquetas de grupo.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz de características ya preprocesada (escalada y codificada).

        Returns
        -------
        numpy.ndarray
            Etiqueta de grupo asignada a cada fila.
        """
        self.estimator_ = self.build_estimator()
        self.labels_ = self.estimator_.fit_predict(X)
        return self.labels_

    def get_params(self) -> dict:
        """Devuelve los hiperparámetros configurables de la estrategia."""
        return {"random_state": self.random_state}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


class KMeansStrategy(ClusteringStrategy):
    """
    Estrategia de clustering por K-Means.

    Adecuada cuando se busca un número fijo de grupos compactos y de tamaño
    comparable. Requiere fijar `n_clusters` y datos estandarizados.

    Parameters
    ----------
    n_clusters : int
        Número de grupos a formar.
    random_state : int
        Semilla aleatoria fija.
    n_init : int
        Número de inicializaciones distintas; se conserva la mejor.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42, n_init: int = 10):
        super().__init__(random_state=random_state)
        self.n_clusters = n_clusters
        self.n_init = n_init

    def build_estimator(self) -> KMeans:
        return KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

    @property
    def name(self) -> str:
        return f"K-Means (k={self.n_clusters})"

    def get_params(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": self.n_init,
        }


class AgglomerativeStrategy(ClusteringStrategy):
    """
    Estrategia de clustering jerárquico aglomerativo.

    Útil como contraste frente a K-Means: no asume grupos esféricos y permite
    analizar la estructura mediante un dendrograma. Es costosa en memoria, por
    lo que conviene aplicarla sobre una muestra.

    Parameters
    ----------
    n_clusters : int
        Número de grupos a formar.
    linkage : str
        Criterio de enlace ('ward', 'complete', 'average', 'single').
    """

    def __init__(self, n_clusters: int = 4, linkage: str = "ward", random_state: int = 42):
        super().__init__(random_state=random_state)
        self.n_clusters = n_clusters
        self.linkage = linkage

    def build_estimator(self) -> AgglomerativeClustering:
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )

    @property
    def name(self) -> str:
        return f"Aglomerativo (k={self.n_clusters}, {self.linkage})"

    def get_params(self) -> dict:
        return {"n_clusters": self.n_clusters, "linkage": self.linkage}


class DBSCANStrategy(ClusteringStrategy):
    """
    Estrategia de clustering por densidad (DBSCAN).

    Adecuada cuando se desea descubrir grupos de forma arbitraria y detectar
    ruido (etiqueta -1). No requiere fijar el número de grupos, pero sí ajustar
    `eps` y `min_samples`. Conviene aplicarla sobre una muestra.

    Parameters
    ----------
    eps : float
        Radio de vecindad.
    min_samples : int
        Número mínimo de puntos para formar una región densa.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 10, random_state: int = 42):
        super().__init__(random_state=random_state)
        self.eps = eps
        self.min_samples = min_samples

    def build_estimator(self) -> DBSCAN:
        return DBSCAN(eps=self.eps, min_samples=self.min_samples)

    @property
    def name(self) -> str:
        return f"DBSCAN (eps={self.eps}, min_samples={self.min_samples})"

    def get_params(self) -> dict:
        return {"eps": self.eps, "min_samples": self.min_samples}

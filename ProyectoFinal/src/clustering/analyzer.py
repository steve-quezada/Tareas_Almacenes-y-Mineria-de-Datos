"""
analyzer.py

Orquestador del análisis de clustering.

`ClusterAnalyzer` coordina el flujo de:

- el `DataRepository` para cargar el dataset,
- el `AccidentClusteringPreprocessor` para preparar las variables,
- las estrategias de clustering (patrón Strategy) para entrenar y comparar.

Reúne el método del codo (inercia), coeficiente de silueta, comparación de
algoritmos, reducción de dimensionalidad (PCA y t-SNE) y perfilado de los
grupos a partir de las variables originales.

El analizador trabaja sobre una muestra reproducible para los cálculos
costosos (silueta, t-SNE, aglomerativo), mientras que el modelo final puede
entrenarse sobre el conjunto completo mediante `ClusteringModel`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Permite importar DataRepository
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from supervised.data_repository import DataRepository

from .preprocessor import AccidentClusteringPreprocessor
from .strategies import ClusteringStrategy, KMeansStrategy


class ClusterAnalyzer:
    """
    Coordina la exploración y evaluación del clustering.

    Attributes
    ----------
    data_path : str
        Ruta al dataset limpio.
    random_state : int
        Semilla aleatoria fija para muestreo y algoritmos.
    repository : DataRepository
        Acceso a los datos (patrón Repository).
    preprocessor : AccidentClusteringPreprocessor
        Preprocesador de las variables de entrada.
    df_full : pd.DataFrame or None
        Dataset completo cargado.
    df_sample : pd.DataFrame or None
        Muestra reproducible usada en los cálculos costosos.
    matrix_sample : numpy.ndarray or None
        Matriz preprocesada de la muestra.
    """

    def __init__(
        self,
        data_path: str,
        config_path: str | None = None,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.random_state = random_state
        self.repository = DataRepository(data_path, config_path=config_path)
        self.preprocessor = AccidentClusteringPreprocessor()

        self.df_full: pd.DataFrame | None = None
        self.df_sample: pd.DataFrame | None = None
        self.matrix_sample: np.ndarray | None = None

    # Preparación de datos
    def prepare(self, sample_size: int = 40_000) -> "ClusterAnalyzer":
        """
        Carga el dataset, toma una muestra reproducible y la preprocesa.

        Parameters
        ----------
        sample_size : int
            Número de filas de la muestra para los cálculos. Si el dataset es menor,
            se usa completo.

        Returns
        -------
        ClusterAnalyzer
            El propio analizador, para encadenar llamadas.
        """
        self.df_full = self.repository.load()

        if sample_size >= len(self.df_full):
            self.df_sample = self.df_full.copy()
        else:
            self.df_sample = self.df_full.sample(
                n=sample_size, random_state=self.random_state
            ).reset_index(drop=True)

        self.matrix_sample = self.preprocessor.fit_transform(self.df_sample)
        return self

    def _check_prepared(self) -> None:
        if self.matrix_sample is None:
            raise RuntimeError("Llama a prepare() antes de ejecutar el análisis.")

    # Selección del número de grupos
    def elbow_scores(self, k_range=range(2, 11)) -> pd.DataFrame:
        """
        Calcula la inercia de K-Means para un rango de k (método del codo).

        Returns
        -------
        pd.DataFrame
            Columnas: 'k' e 'inercia'.
        """
        self._check_prepared()
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(self.matrix_sample)
            rows.append({"k": k, "inercia": km.inertia_})
        return pd.DataFrame(rows)

    def silhouette_scores(self, k_range=range(2, 11)) -> pd.DataFrame:
        """
        Calcula el coeficiente de silueta promedio de K-Means para varios k.

        Returns
        -------
        pd.DataFrame
            Columnas: 'k' y 'silueta'.
        """
        self._check_prepared()
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(self.matrix_sample)
            score = silhouette_score(
                self.matrix_sample, labels, random_state=self.random_state
            )
            rows.append({"k": k, "silueta": score})
        return pd.DataFrame(rows)

    # Comparación de algoritmos (patrón Strategy)
    def compare_strategies(self, strategies: list[ClusteringStrategy]) -> pd.DataFrame:
        """
        Ajusta varias estrategias sobre la misma muestra y las compara.

        Para cada estrategia reporta el número de grupos encontrados, el número
        de puntos clasificados como ruido (solo DBSCAN) y el coeficiente de
        silueta cuando es aplicable.

        Parameters
        ----------
        strategies : list of ClusteringStrategy
            Estrategias a comparar.

        Returns
        -------
        pd.DataFrame
            Tabla comparativa.
        """
        self._check_prepared()
        rows = []
        for strategy in strategies:
            labels = strategy.fit_predict(self.matrix_sample)
            unique = set(labels)
            n_noise = int(np.sum(labels == -1))
            n_groups = len(unique - {-1})

            if n_groups >= 2:
                mask = labels != -1
                score = silhouette_score(
                    self.matrix_sample[mask],
                    labels[mask],
                    random_state=self.random_state,
                )
            else:
                score = np.nan

            rows.append(
                {
                    "algoritmo": strategy.name,
                    "n_grupos": n_groups,
                    "n_ruido": n_noise,
                    "silueta": score,
                }
            )
        return pd.DataFrame(rows)

    # Reducción de dimensionalidad para visualización
    def reduce_pca(self, n_components: int = 2) -> np.ndarray:
        """Proyecta la muestra preprocesada con PCA."""
        self._check_prepared()
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(self.matrix_sample)

    def pca_explained_variance(self, n_components: int = 10) -> pd.DataFrame:
        """
        Varianza explicada por las primeras componentes principales.

        Returns
        -------
        pd.DataFrame
            Columnas: 'componente', 'varianza' y 'varianza_acumulada'.
        """
        self._check_prepared()
        n = min(n_components, self.matrix_sample.shape[1])
        pca = PCA(n_components=n, random_state=self.random_state)
        pca.fit(self.matrix_sample)
        ratio = pca.explained_variance_ratio_
        return pd.DataFrame(
            {
                "componente": range(1, n + 1),
                "varianza": ratio,
                "varianza_acumulada": np.cumsum(ratio),
            }
        )

    def reduce_tsne(self, sample_size: int = 5_000, perplexity: float = 30.0) -> tuple:
        """
        Proyecta una submuestra con t-SNE (costoso; usa pocos miles de puntos).

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Coordenadas 2D y los índices de la submuestra usada (relativos a
            `df_sample`).
        """
        self._check_prepared()
        n = min(sample_size, self.matrix_sample.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(self.matrix_sample.shape[0], size=n, replace=False)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            init="pca",
        )
        coords = tsne.fit_transform(self.matrix_sample[idx])
        return coords, idx

    # Perfilado de grupos
    def profile_numeric(self, labels: np.ndarray, columns: list[str]) -> pd.DataFrame:
        """
        Promedio de variables numéricas por grupo (perfil cuantitativo).

        Parameters
        ----------
        labels : numpy.ndarray
            Etiqueta de grupo de cada fila de `df_sample`.
        columns : list of str
            Columnas numéricas originales a resumir.
        """
        self._check_prepared()
        tmp = self.df_sample[columns].copy()
        tmp["cluster"] = labels
        return tmp.groupby("cluster").mean(numeric_only=True).round(3)

    def profile_categorical(self, labels: np.ndarray, column: str) -> pd.DataFrame:
        """
        Categoría más frecuente de una variable por grupo, con su proporción.

        Parameters
        ----------
        labels : numpy.ndarray
            Etiqueta de grupo de cada fila de `df_sample`.
        column : str
            Variable categórica original a resumir.
        """
        self._check_prepared()
        tmp = self.df_sample[[column]].copy()
        tmp["cluster"] = labels
        rows = []
        for cluster, group in tmp.groupby("cluster"):
            counts = group[column].value_counts(normalize=True)
            rows.append(
                {
                    "cluster": cluster,
                    "categoria_dominante": counts.index[0],
                    "proporcion": round(float(counts.iloc[0]), 3),
                }
            )
        return pd.DataFrame(rows).set_index("cluster")

    def crosstab_with_target(
        self, labels: np.ndarray, target: str = "CLASACC", normalize: str = "index"
    ) -> pd.DataFrame:
        """
        Tabla cruzada entre los grupos y la variable objetivo del modelo
        supervisado.

        Permite conectar los perfiles del clustering con la severidad que
        predice el modelo supervisado, sin que la severidad haya intervenido en
        la formación de los grupos.

        Parameters
        ----------
        labels : numpy.ndarray
            Etiqueta de grupo de cada fila de `df_sample`.
        target : str
            Variable objetivo (por defecto `CLASACC`).
        normalize : str
            Igual que en `pd.crosstab` ('index', 'columns', False).
        """
        self._check_prepared()
        return pd.crosstab(
            labels,
            self.df_sample[target],
            normalize=normalize,
        ).round(3)

    def cluster_sizes(self, labels: np.ndarray) -> pd.DataFrame:
        """Tamaño absoluto y relativo de cada grupo."""
        s = pd.Series(labels, name="cluster")
        sizes = s.value_counts().sort_index()
        return pd.DataFrame(
            {
                "n": sizes,
                "porcentaje": (sizes / len(s) * 100).round(2),
            }
        )

    def fit_strategy(self, strategy: ClusteringStrategy) -> np.ndarray:
        """
        Ajusta una estrategia sobre la muestra y devuelve sus etiquetas.

        Atajo cuando solo se quiere el resultado de un algoritmo concreto
        (por ejemplo, el K-Means final con el k elegido).
        """
        self._check_prepared()
        return strategy.fit_predict(self.matrix_sample)

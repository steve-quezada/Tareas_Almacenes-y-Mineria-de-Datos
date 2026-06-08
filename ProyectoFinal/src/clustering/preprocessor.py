"""
preprocessor.py

Preprocesamiento de las variables de entrada.

La clase AccidentClusteringPreprocessor encapsula, en un solo objeto, todas
las decisiones de preparación de datos necesarias antes de aplicar K-Means:

- Selección de las variables de *contexto* del accidente (temporales, de zona,
  de tipo/causa, vehiculares y del conductor).
- Exclusión explícita de la variable objetivo `CLASACC` (que pertenece al
  modelo supervisado) y de las columnas de víctimas, para que los grupos no
  se formen a partir de la severidad que después se quiere interpretar.
- Imputación de la edad faltante por la mediana.
- Codificación one-hot de las variables categóricas.
- Estandarización de todas las variables numéricas.

El preprocesador se construye sobre un `ColumnTransformer` de scikit-learn, de
modo que el mismo objeto ajustado pueda transformar datos nuevos de forma
idéntica (requisito para reutilizar el modelo guardado sin re-entrenar).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class AccidentClusteringPreprocessor:
    """
    Prepara el dataset limpio.

    Encapsula la selección de variables y un `ColumnTransformer` que imputa,
    codifica y escala. La separación entre variables de contexto y variables
    excluidas se mantiene como atributos de clase para que las decisiones sean
    explícitas.

    Patrón: forma parte de la composición usada por `ClusterAnalyzer` y
    `ClusteringModel`; aquí se concentra toda la lógica de transformación de
    datos, separada de la lógica de modelado.

    Attributes
    ----------
    TARGET_COLUMN : str
        Variable objetivo del modelo supervisado. Nunca entra al clustering.
    VICTIM_COLUMNS : list of str
        Conteos de víctimas. Se excluyen porque determinan directamente la
        severidad (`CLASACC`) y filtrarían el resultado del clustering.
    HIGH_CARDINALITY_COLUMNS : list of str
        Columnas geográficas de muy alta cardinalidad. Se excluyen del
        clustering, pero se conservan en el dataset para perfilar los grupos.
    NOISE_COLUMNS : list of str
        Componentes temporales demasiado finos para describir un perfil.
    temporal_features, zone_features, accident_features, vehicle_features,
    driver_categorical_features, driver_numeric_features : list of str
        Grupos de variables de contexto que sí entran al clustering.
    """

    TARGET_COLUMN = "CLASACC"

    VICTIM_COLUMNS = [
        "CONDMUERTO", "CONDHERIDO", "PASAMUERTO", "PASAHERIDO",
        "PEATMUERTO", "PEATHERIDO", "CICLMUERTO", "CICLHERIDO",
        "OTROMUERTO", "OTROHERIDO",
    ]

    HIGH_CARDINALITY_COLUMNS = ["NOM_ENT", "NOM_MUN"]

    NOISE_COLUMNS = ["ID_MINUTO", "ID_DIA"]

    # Variables de contexto que SÍ entran al clustering
    temporal_features = ["MES", "ID_HORA", "DIASEMANA"]
    zone_features = ["URBANA", "SUBURBANA"]
    accident_features = ["TIPACCID", "CAUSAACCI", "CAPAROD"]
    vehicle_features = [
        "AUTOMOVIL", "CAMPASAJ", "MICROBUS", "PASCAMION", "OMNIBUS",
        "TRANVIA", "CAMIONETA", "CAMION", "TRACTOR", "FERROCARRI",
        "MOTOCICLET", "BICICLETA", "OTROVEHIC",
    ]
    driver_categorical_features = ["SEXO", "ALIENTO", "CINTURON"]
    driver_numeric_features = ["ID_EDAD", "CONDUCTOR_FUGADO", "EDAD_DESCONOCIDA"]

    def __init__(self):
        self._column_transformer: ColumnTransformer | None = None
        self._fitted = False

    # Definición de columnas
    @property
    def categorical_features(self) -> list[str]:
        """Variables categóricas que se codifican con one-hot."""
        return (
            ["DIASEMANA"]
            + self.zone_features
            + self.accident_features
            + self.driver_categorical_features
        )

    @property
    def numeric_features(self) -> list[str]:
        """Variables numéricas que se imputan y se estandarizan."""
        return (
            ["MES", "ID_HORA"]
            + self.vehicle_features
            + self.driver_numeric_features
        )

    @property
    def input_features(self) -> list[str]:
        """Lista completa de columnas de entrada al clustering."""
        return self.numeric_features + self.categorical_features

    # Construcción del transformador
    def _build_transformer(self) -> ColumnTransformer:
        """
        Construye el `ColumnTransformer` con dos ramas:

        - Numérica: imputación por mediana + estandarización.
        - Categórica: one-hot (ignora categorías no vistas en datos nuevos).
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

    # API pública
    def fit_transform(self, df: pd.DataFrame):
        """
        Ajusta el preprocesador sobre `df` y devuelve la matriz transformada.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset limpio que contiene al menos las columnas de entrada.

        Returns
        -------
        numpy.ndarray
            Matriz de características escaladas y codificadas, lista para el
            algoritmo de clustering.
        """
        self._validate_columns(df)
        self._column_transformer = self._build_transformer()
        matrix = self._column_transformer.fit_transform(df[self.input_features])
        self._fitted = True
        return matrix

    def transform(self, df: pd.DataFrame):
        """
        Transforma datos nuevos con el preprocesador ya ajustado.

        Raises
        ------
        RuntimeError
            Si se llama antes de `fit_transform`.
        """
        if not self._fitted:
            raise RuntimeError(
                "El preprocesador no ha sido ajustado. Llama a fit_transform() primero."
            )
        self._validate_columns(df)
        return self._column_transformer.transform(df[self.input_features])

    def get_feature_names(self) -> list[str]:
        """
        Devuelve los nombres de las columnas resultantes tras la
        transformación (numéricas + one-hot).

        Raises
        ------
        RuntimeError
            Si se llama antes de `fit_transform`.
        """
        if not self._fitted:
            raise RuntimeError(
                "El preprocesador no ha sido ajustado. Llama a fit_transform() primero."
            )
        return list(self._column_transformer.get_feature_names_out())

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Verifica que el DataFrame contenga las columnas de entrada."""
        missing = [c for c in self.input_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"El DataFrame no contiene las columnas de entrada requeridas: {missing}"
            )

"""
preprocessor.py
===============

Encapsula el pipeline completo de preprocesamiento de features para
el dataset ATUS 2024. Construye internamente un ColumnTransformer de
scikit-learn que aplica transformaciones específicas a cada grupo de
variables según su tipo.

Las columnas por grupo y las columnas a excluir se configuran mediante
un archivo YAML externo, lo que permite modificarlas sin tocar el código.

Transformaciones aplicadas:
    - Numéricas:    SimpleImputer(mediana) -> StandardScaler
    - Binarias:     passthrough
    - Categóricas:  OneHotEncoder(drop='first', handle_unknown='ignore')
    - Cíclicas:     Encoding seno/coseno (MES, ID_HORA)
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Transformador personalizado para encoding cíclico
# ---------------------------------------------------------------------------

class CyclicEncoder(BaseEstimator, TransformerMixin):
    """
    Transforma variables temporales cíclicas en pares seno/coseno.

    Convierte una variable con naturaleza circular (como hora del día
    o mes del año) en dos componentes que preservan la continuidad
    entre el último y el primer valor del ciclo.

    Parameters
    ----------
    period : int
        Valor máximo del ciclo.

    Example
    -------
    >>> enc = CyclicEncoder(period=24)
    >>> enc.fit_transform(np.array([[0], [6], [12], [23]]))
    """

    def __init__(self, period: int):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = np.array(X).flatten()
        sin = np.sin(2 * np.pi * X_ / self.period)
        cos = np.cos(2 * np.pi * X_ / self.period)
        return np.column_stack([sin, cos])

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            name = input_features[0]
        else:
            name = "x"
        return np.array([f"{name}_SEN", f"{name}_COS"])


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Pipeline de preprocesamiento para el dataset ATUS 2024.

    Construye un ColumnTransformer de scikit-learn que aplica
    transformaciones específicas a cada grupo de variables. La
    configuración de columnas se carga desde un archivo YAML externo.

    Patrón: Pipeline — encadena etapas de transformación secuenciales
    sobre distintos grupos de variables.

    Attributes
    ----------
    config_path : Path
        Ruta al archivo YAML de configuración.
    exclude_columns : list
        Columnas a excluir antes de cualquier transformación.
    numeric_columns : list
        Columnas numéricas continuas.
    binary_columns : list
        Columnas binarias (passthrough).
    categorical_columns : list
        Columnas categóricas nominales.
    cyclic_columns : list of dict
        Columnas cíclicas con su periodo {'name': str, 'period': int}.
    pipeline : ColumnTransformer or None
        Pipeline construido tras llamar a fit() o fit_transform().

    Example
    -------
    >>> prep = Preprocessor()
    >>> X_train_transformed = prep.fit_transform(X_train)
    >>> X_test_transformed  = prep.transform(X_test)
    >>> print(prep.feature_names)
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "preprocessor_config.yaml"

    def __init__(self, config_path: str = None):
        """
        Parameters
        ----------
        config_path : str, optional
            Ruta al archivo YAML de configuración. Si no se especifica,
            se usa src/config/preprocessor_config.yaml por defecto.
        """
        self.config_path = (
            Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        )
        self.pipeline = None
        self._load_config()

    def _load_config(self) -> None:
        """
        Carga la configuración de columnas desde el archivo YAML.

        Raises
        ------
        FileNotFoundError
            Si el archivo YAML no existe en la ruta especificada.
        KeyError
            Si el archivo YAML no contiene alguna clave requerida.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración en: {self.config_path}\n"
                f"Asegúrate de que el archivo YAML existe en la ruta especificada."
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_keys = [
            "exclude_columns", "numeric_columns", "binary_columns",
            "categorical_columns", "cyclic_columns"
        ]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise KeyError(
                f"El archivo de configuración '{self.config_path}' no contiene "
                f"las siguientes claves requeridas: {missing}"
            )

        self.exclude_columns      = config["exclude_columns"]
        self.numeric_columns      = config["numeric_columns"]
        self.binary_columns       = config["binary_columns"]
        self.categorical_columns  = config["categorical_columns"]
        self.cyclic_columns       = config["cyclic_columns"]

    def _build_pipeline(self) -> ColumnTransformer:
        """
        Construye el ColumnTransformer con las transformaciones por grupo.

        Returns
        -------
        ColumnTransformer
            Pipeline listo para fit/transform.
        """
        # Numéricas: imputación con mediana + escalado
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])

        # Categóricas: OHE con drop='first' para evitar multicolinealidad
        categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers = [
            ("numeric",     numeric_pipeline,     self.numeric_columns),
            ("binary",      "passthrough",         self.binary_columns),
            ("categorical", categorical_pipeline,  self.categorical_columns),
        ]

        # Cíclicas: un transformador por variable con su propio periodo
        for col_info in self.cyclic_columns:
            name   = col_info["name"]
            period = col_info["period"]
            transformers.append((
                f"cyclic_{name.lower()}",
                CyclicEncoder(period=period),
                [name]
            ))

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop"    # Descarta columnas no listadas (exclude_columns)
        )

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Ajusta el pipeline sobre el conjunto de entrenamiento.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento. No debe incluir la variable objetivo.

        Returns
        -------
        Preprocessor
            La instancia ajustada (permite encadenamiento).
        """
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Aplica las transformaciones aprendidas a un nuevo conjunto.

        Parameters
        ----------
        X : pd.DataFrame
            Features a transformar.

        Returns
        -------
        np.ndarray
            Array transformado listo para el modelo.

        Raises
        ------
        RuntimeError
            Si se llama antes de fit().
        """
        if self.pipeline is None:
            raise RuntimeError(
                "El pipeline no ha sido ajustado. Llama a fit() primero."
            )
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ajusta el pipeline y transforma en un solo paso.

        Equivalente a llamar fit(X).transform(X). Usar solo sobre el
        conjunto de entrenamiento para evitar data leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento.

        Returns
        -------
        np.ndarray
            Array transformado.
        """
        return self.fit(X).transform(X)

    @property
    def feature_names(self) -> list:
        """
        Retorna los nombres de las features tras la transformación.

        Returns
        -------
        list
            Nombres de columnas del array transformado.

        Raises
        ------
        RuntimeError
            Si se llama antes de fit().
        """
        if self.pipeline is None:
            raise RuntimeError(
                "El pipeline no ha sido ajustado. Llama a fit() primero."
            )
        return self.pipeline.get_feature_names_out().tolist()

    def summary(self) -> str:
        """
        Retorna un resumen de la configuración del pipeline.

        Returns
        -------
        str
            Resumen con grupos de variables y transformaciones aplicadas.
        """
        lines = [
            "=" * 55,
            "        CONFIGURACIÓN DEL PREPROCESADOR",
            "=" * 55,
            f"  Configuración: {self.config_path}",
            "",
            f"  Columnas excluidas      ({len(self.exclude_columns)}):",
        ]
        for col in self.exclude_columns:
            lines.append(f"    - {col}")

        lines += [
            "",
            f"  Numéricas               ({len(self.numeric_columns)}): "
            f"Imputer(mediana) + StandardScaler",
        ]
        for col in self.numeric_columns:
            lines.append(f"    - {col}")

        lines += [
            "",
            f"  Binarias                ({len(self.binary_columns)}): "
            f"passthrough",
        ]
        for col in self.binary_columns:
            lines.append(f"    - {col}")

        lines += [
            "",
            f"  Categóricas             ({len(self.categorical_columns)}): "
            f"OneHotEncoder(drop='first')",
        ]
        for col in self.categorical_columns:
            lines.append(f"    - {col}")

        lines += [
            "",
            f"  Cíclicas                ({len(self.cyclic_columns)}): "
            f"encoding seno/coseno",
        ]
        for col_info in self.cyclic_columns:
            lines.append(
                f"    - {col_info['name']} (periodo={col_info['period']}) "
                f"-> {col_info['name']}_SEN, {col_info['name']}_COS"
            )

        if self.pipeline is not None:
            lines += [
                "",
                f"  Features tras transformación: {len(self.feature_names)}",
            ]

        lines.append("=" * 55)
        return "\n".join(lines)
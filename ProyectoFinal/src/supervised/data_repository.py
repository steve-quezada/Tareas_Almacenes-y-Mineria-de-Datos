"""
data_repository.py
==================
Patrón de diseño: Repository

Abstrae el acceso al dataset limpio detrás de una interfaz uniforme.
El resto del proyecto (notebook de modelado, trainer, etc.) nunca
accede directamente al sistema de archivos — siempre usa esta clase.
Esto permite cambiar la fuente de datos (CSV, base de datos, API)
sin modificar el código cliente.

Las columnas requeridas se configuran externamente mediante un archivo
YAML, lo que permite ajustar la validación sin modificar el código.
"""

import pandas as pd
import yaml
from pathlib import Path


class DataRepository:
    """
    Repositorio de datos para el dataset ATUS 2024 limpio.

    Encapsula la carga y validación básica del CSV limpio generado
    en el notebook de análisis exploratorio. Garantiza que el dataset
    cumpla con las dimensiones y columnas mínimas esperadas antes de
    ser utilizado en cualquier etapa del pipeline de modelado.

    Las columnas requeridas se leen desde un archivo YAML externo,
    lo que permite modificarlas sin tocar el código fuente.

    Patrón: Repository — abstrae el acceso a la fuente de datos
    detrás de una interfaz uniforme.

    Attributes
    ----------
    data_path : Path
        Ruta al archivo CSV del dataset limpio.
    config_path : Path
        Ruta al archivo YAML de configuración.
    _df : pd.DataFrame or None
        Dataset cargado en memoria. None hasta que se llame a load().
    _required_columns : list
        Lista de columnas mínimas leída desde el archivo de configuración.

    Example
    -------
    >>> # Uso con configuración por defecto
    >>> repo = DataRepository("data/atus_anual_2024_limpio.csv")
    >>> df = repo.load()
    >>> print(repo.summary())

    >>> # Uso con configuración personalizada
    >>> repo = DataRepository(
    ...     "data/atus_anual_2024_limpio.csv",
    ...     config_path="src/config/mi_config.yaml"
    ... )
    """

    # Ruta relativa al propio archivo data_repository.py, no al cwd.
    # Esto garantiza que el config se encuentre independientemente de
    # desde donde se ejecute el codigo (notebook, script, tests, etc.)
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "repository_config.yaml"

    def __init__(self, data_path: str, config_path: str = None):
        """
        Parameters
        ----------
        data_path : str
            Ruta al archivo CSV del dataset limpio.
        config_path : str, optional
            Ruta al archivo YAML de configuración. Si no se especifica,
            se usa src/config/repository_config.yaml por defecto.
        """
        self.data_path   = Path(data_path)
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._df                = None
        self._required_columns  = self._load_config()

    def _load_config(self) -> list:
        """
        Carga la lista de columnas requeridas desde el archivo YAML.

        Returns
        -------
        list
            Lista de nombres de columnas requeridas.

        Raises
        ------
        FileNotFoundError
            Si el archivo de configuración no existe.
        KeyError
            Si el archivo YAML no contiene la clave 'required_columns'.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración en: {self.config_path}\n"
                f"Asegúrate de que el archivo YAML existe en la ruta especificada."
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "target_column" not in config:
            raise KeyError(
                f"El archivo de configuración '{self.config_path}' no contiene "
                f"la clave 'target_column'."
            )

        self.target_column = config["target_column"]

        if "required_columns" not in config:
            raise KeyError(
                f"El archivo de configuración '{self.config_path}' no contiene "
                f"la clave 'required_columns'."
            )

        return config["required_columns"]

    def load(self) -> pd.DataFrame:
        """
        Carga el dataset limpio desde disco y valida su estructura.

        Returns
        -------
        pd.DataFrame
            Dataset limpio listo para ser procesado.

        Raises
        ------
        FileNotFoundError
            Si el archivo CSV no existe en la ruta especificada.
        ValueError
            Si el dataset no contiene las columnas mínimas requeridas
            o si está vacío.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset en: {self.data_path}\n"
                f"Asegúrate de haber ejecutado el notebook de EDA "
                f"y exportado el dataset limpio."
            )

        self._df = pd.read_csv(self.data_path, low_memory=False)

        self._validate()

        return self._df.copy()

    def _validate(self) -> None:
        """
        Valida que el dataset cumpla con los requisitos mínimos.

        Raises
        ------
        ValueError
            Si el dataset está vacío o faltan columnas requeridas.
        """
        if self._df.empty:
            raise ValueError("El dataset está vacío.")

        missing_cols = [
            col for col in self._required_columns
            if col not in self._df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"El dataset no contiene las siguientes columnas "
                f"requeridas: {missing_cols}\n"
                f"Verifica que el CSV exportado corresponde a la "
                f"versión final del notebook de limpieza, o actualiza "
                f"'{self.config_path}' si las columnas cambiaron."
            )

    def summary(self) -> str:
        """
        Retorna un resumen del dataset cargado.

        Returns
        -------
        str
            Resumen con dimensiones, tipos de datos y distribución
            de la variable objetivo configurada.

        Raises
        ------
        RuntimeError
            Si se llama antes de ejecutar load().
        """
        if self._df is None:
            raise RuntimeError(
                "El dataset no ha sido cargado. Llama a load() primero."
            )

        lines = [
            "=" * 55,
            "           RESUMEN DEL DATASET CARGADO",
            "=" * 55,
            f"  Ruta:          {self.data_path}",
            f"  Configuración: {self.config_path}",
            f"  Filas:         {self._df.shape[0]:,}",
            f"  Columnas:      {self._df.shape[1]}",
            "",
            f"  Distribución de {self.target_column} (variable objetivo):",
        ]

        dist = self._df[self.target_column].value_counts()
        pct  = self._df[self.target_column].value_counts(normalize=True).mul(100)
        for cls in dist.index:
            lines.append(f"    {cls:<12} {dist[cls]:>7,}  ({pct[cls]:.2f}%)")

        nulos = self._df.isnull().sum()
        nulos = nulos[nulos > 0]
        if not nulos.empty:
            lines += ["", "  Valores nulos por columna:"]
            for col, n in nulos.items():
                lines.append(f"    {col:<25} {n:>7,}")
        else:
            lines += ["", "  Sin valores nulos."]

        lines.append("=" * 55)

        return "\n".join(lines)

    @property
    def shape(self):
        """Retorna las dimensiones del dataset cargado."""
        if self._df is None:
            raise RuntimeError(
                "El dataset no ha sido cargado. Llama a load() primero."
            )
        return self._df.shape

    @property
    def columns(self):
        """Retorna las columnas del dataset cargado."""
        if self._df is None:
            raise RuntimeError(
                "El dataset no ha sido cargado. Llama a load() primero."
            )
        return self._df.columns.tolist()

    @property
    def required_columns(self):
        """Retorna la lista de columnas requeridas según la configuración."""
        return self._required_columns.copy()
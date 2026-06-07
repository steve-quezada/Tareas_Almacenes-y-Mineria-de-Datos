"""
model_factory.py
================

Encapsula la creación de instancias de modelos de clasificación.
El código cliente solicita un modelo por nombre y recibe una instancia
configurada y lista para entrenar, sin conocer los detalles internos
de cada algoritmo.

Los hiperparámetros base de cada modelo se configuran mediante un
archivo YAML externo, lo que permite ajustarlos sin modificar el
código fuente.

Modelos disponibles:
    - 'dummy'               DummyClassifier (línea base trivial)
    - 'logistic_regression' LogisticRegression
    - 'random_forest'       RandomForestClassifier
    - 'xgboost'             XGBClassifier
"""

import yaml
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelFactory:
    """
    Fábrica de modelos de clasificación para el dataset ATUS 2024.

    Proporciona una interfaz uniforme para instanciar cualquier modelo
    disponible con sus hiperparámetros base configurados desde un YAML
    externo. Permite sobreescribir parámetros individuales en tiempo
    de ejecución sin modificar la configuración base.

    Patrón: Factory Method — centraliza la creación de objetos de
    distintas clases bajo una interfaz común, desacoplando el código
    cliente de las implementaciones concretas.

    Example
    -------
    >>> # Modelo con configuración base del YAML
    >>> model = ModelFactory.create('random_forest')

    >>> # Modelo con parámetro sobreescrito
    >>> model = ModelFactory.create('random_forest', n_estimators=200)

    >>> # Listar modelos disponibles
    >>> print(ModelFactory.available_models())
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "model_factory_config.yaml"

    # Registro de modelos disponibles: nombre -> clase
    _REGISTRY = {
        "dummy":                DummyClassifier,
        "logistic_regression":  LogisticRegression,
        "random_forest":        RandomForestClassifier,
    }

    @classmethod
    def _load_config(cls, config_path: Path) -> dict:
        """
        Carga los hiperparámetros base desde el archivo YAML.

        Parameters
        ----------
        config_path : Path
            Ruta al archivo YAML de configuración.

        Returns
        -------
        dict
            Diccionario con hiperparámetros base por modelo.

        Raises
        ------
        FileNotFoundError
            Si el archivo YAML no existe.
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración en: {config_path}\n"
                f"Asegúrate de que el archivo YAML existe en la ruta especificada."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def create(
        cls,
        model_name: str,
        config_path: str = None,
        **kwargs
    ):
        """
        Instancia un modelo con sus hiperparámetros base configurados.

        Los parámetros del YAML se usan como base. Cualquier argumento
        adicional en kwargs sobreescribe el valor correspondiente del YAML.

        Parameters
        ----------
        model_name : str
            Nombre del modelo. Debe ser uno de los disponibles en
            ModelFactory.available_models().
        config_path : str, optional
            Ruta al archivo YAML de configuración. Si no se especifica,
            se usa src/config/model_factory_config.yaml por defecto.
        **kwargs
            Hiperparámetros adicionales que sobreescriben los del YAML.

        Returns
        -------
        Estimator
            Instancia del modelo solicitado lista para entrenar.

        Raises
        ------
        ValueError
            Si el nombre del modelo no está registrado.

        Example
        -------
        >>> rf = ModelFactory.create('random_forest')
        >>> rf = ModelFactory.create('random_forest', n_estimators=200)
        >>> lr = ModelFactory.create('logistic_regression', max_iter=500)
        """
        if model_name not in cls._REGISTRY:
            raise ValueError(
                f"Modelo '{model_name}' no disponible.\n"
                f"Modelos disponibles: {cls.available_models()}"
            )

        # XGBoost se importa aquí para que la clase no falle si no está
        # instalado y el usuario no lo necesita
        if model_name == "xgboost":
            try:
                from xgboost import XGBClassifier
                cls._REGISTRY["xgboost"] = XGBClassifier
            except ImportError:
                raise ImportError(
                    "XGBoost no está instalado. Instálalo con: pip install xgboost"
                )

        config_path = Path(config_path) if config_path else cls.DEFAULT_CONFIG_PATH
        config      = cls._load_config(config_path)

        # Hiperparámetros base del YAML para este modelo
        base_params = config.get(model_name, {})

        # kwargs sobreescriben los parámetros base
        final_params = {**base_params, **kwargs}

        model_class = cls._REGISTRY[model_name]
        return model_class(**final_params)

    @classmethod
    def available_models(cls) -> list:
        """
        Retorna la lista de nombres de modelos disponibles.

        Returns
        -------
        list
            Nombres de modelos registrados en la fábrica.
        """
        return list(cls._REGISTRY.keys()) + ["xgboost"]

    @classmethod
    def summary(cls, config_path: str = None) -> str:
        """
        Retorna un resumen de los modelos disponibles y sus
        hiperparámetros base.

        Parameters
        ----------
        config_path : str, optional
            Ruta al archivo YAML de configuración.

        Returns
        -------
        str
            Resumen con modelos y parámetros configurados.
        """
        config_path = Path(config_path) if config_path else cls.DEFAULT_CONFIG_PATH
        config      = cls._load_config(config_path)

        lines = [
            "=" * 55,
            "         MODELOS DISPONIBLES — ModelFactory",
            "=" * 55,
            f"  Configuración: {config_path}",
            "",
        ]

        for model_name in cls.available_models():
            params = config.get(model_name, {})
            lines.append(f"  [{model_name}]")
            for k, v in params.items():
                lines.append(f"    {k:<25} {v}")
            lines.append("")

        lines.append("=" * 55)
        return "\n".join(lines)
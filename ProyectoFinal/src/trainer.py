"""
trainer.py
==========
Encapsula el entrenamiento, validación cruzada y ajuste de
hiperparámetros de un modelo de clasificación.

El código cliente entrega un modelo instanciado y los datos de
entrenamiento. Trainer se encarga de:
    1. Validación cruzada inicial (cross_val_score)
    2. Ajuste de hiperparámetros (GridSearchCV)
    3. Medición del tiempo de entrenamiento
    4. Retorno del mejor modelo ajustado y sus resultados
    5. Cargar y serializar modelos

La configuración de folds, métrica y param_grid por modelo se carga
desde un archivo YAML externo.
"""

import time
import yaml
import numpy as np
from pathlib import Path

from sklearn.model_selection import GridSearchCV, cross_val_score


class Trainer:
    """
    Entrena y ajusta un modelo de clasificación sobre el dataset ATUS 2024.

    Encapsula el flujo completo de entrenamiento: validación cruzada
    inicial, búsqueda de hiperparámetros con GridSearchCV y registro
    del tiempo de entrenamiento. Trabaja con un modelo a la vez para
    mantener la trazabilidad de cada experimento.

    Attributes
    ----------
    random_state : int
        Semilla para reproducibilidad.
    config_path : Path
        Ruta al archivo YAML de configuración.
    n_folds : int
        Número de folds para validación cruzada y GridSearchCV.
    scoring : str
        Métrica principal de optimización.
    results_ : dict or None
        Resultados del último entrenamiento. None hasta llamar a fit().

    Example
    -------
    >>> trainer = Trainer(random_state=42)
    >>> best_model, results = trainer.fit(rf, X_train, y_train, 'random_forest')
    >>> print(trainer.summary())
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "trainer_config.yaml"

    def __init__(self, random_state: int = 42, config_path: str = None):
        """
        Parameters
        ----------
        random_state : int
            Semilla para reproducibilidad. Debe coincidir con
            RANDOM_STATE del notebook.
        config_path : str, optional
            Ruta al archivo YAML de configuración. Si no se especifica,
            se usa src/config/trainer_config.yaml por defecto.
        """
        self.random_state = random_state
        self.config_path  = (
            Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        )
        self.results_ = None
        self._config  = self._load_config()
        self.n_folds  = self._config["cross_validation"]["n_folds"]
        self.scoring  = self._config["cross_validation"]["scoring"]

    def _load_config(self) -> dict:
        """
        Carga la configuración desde el archivo YAML.

        Returns
        -------
        dict
            Configuración completa del Trainer.

        Raises
        ------
        FileNotFoundError
            Si el archivo YAML no existe.
        KeyError
            Si el archivo YAML no contiene la clave 'cross_validation'.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración en: {self.config_path}\n"
                f"Asegúrate de que el archivo YAML existe en la ruta especificada."
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "cross_validation" not in config:
            raise KeyError(
                f"El archivo '{self.config_path}' no contiene "
                f"la clave 'cross_validation'."
            )

        return config

    def _get_param_grid(self, model_name: str) -> dict:
        """
        Retorna el param_grid configurado para el modelo solicitado.

        Parameters
        ----------
        model_name : str
            Nombre del modelo según el YAML.

        Returns
        -------
        dict
            Grilla de hiperparámetros. Vacía si el modelo no tiene
            param_grid configurado.
        """
        return self._config.get(model_name, {}).get("param_grid", {})

    def fit(self, model, X_train, y_train, model_name: str):
        """
        Entrena y ajusta el modelo sobre el conjunto de entrenamiento.

        Si el param_grid del modelo está vacío (como en DummyClassifier),
        solo realiza entrenamiento simple y validación cruzada sin
        GridSearchCV.

        Parameters
        ----------
        model : estimator
            Instancia del modelo a entrenar (de ModelFactory.create()).
        X_train : array-like
            Features de entrenamiento transformadas por Preprocessor.
        y_train : array-like
            Variable objetivo de entrenamiento.
        model_name : str
            Nombre del modelo, usado para cargar su param_grid del YAML.
            Debe coincidir con una clave del archivo trainer_config.yaml.

        Returns
        -------
        tuple : (best_model, results)
            best_model : estimator
                Modelo ajustado con los mejores hiperparámetros.
            results : dict
                Diccionario con métricas y detalles del entrenamiento.
        """
        param_grid = self._get_param_grid(model_name)
        results    = {"model_name": model_name}

        # --- Validación cruzada inicial ---
        print(f"[{model_name}] Ejecutando validación cruzada ({self.n_folds} folds)...")
        start = time.time()

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.n_folds,
            scoring=self.scoring,
            n_jobs=-1
        )

        results["cv_scores"]     = cv_scores
        results["cv_mean"]       = cv_scores.mean()
        results["cv_std"]        = cv_scores.std()
        results["cv_time_sec"]   = round(time.time() - start, 2)

        print(
            f"[{model_name}] CV {self.scoring}: "
            f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f} "
            f"({results['cv_time_sec']}s)"
        )

        # --- Ajuste de hiperparámetros (solo si hay param_grid) ---
        if param_grid:
            print(f"[{model_name}] Ejecutando GridSearchCV...")
            start = time.time()

            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.n_folds,
                scoring=self.scoring,
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            results["best_params"]      = grid_search.best_params_
            results["best_cv_score"]    = grid_search.best_score_
            results["grid_time_sec"]    = round(time.time() - start, 2)

            print(
                f"[{model_name}] Mejores parámetros: {results['best_params']}\n"
                f"[{model_name}] Mejor {self.scoring}: "
                f"{results['best_cv_score']:.4f} "
                f"({results['grid_time_sec']}s)"
            )
        else:
            # Sin param_grid: entrenamiento simple
            print(f"[{model_name}] Entrenando modelo (sin GridSearchCV)...")
            start = time.time()

            model.fit(X_train, y_train)
            best_model = model

            results["best_params"]   = {}
            results["best_cv_score"] = results["cv_mean"]
            results["grid_time_sec"] = round(time.time() - start, 2)

            print(
                f"[{model_name}] Entrenamiento completado "
                f"({results['grid_time_sec']}s)"
            )

        results["total_time_sec"] = round(
            results["cv_time_sec"] + results["grid_time_sec"], 2
        )

        self.results_ = results
        return best_model, results

    def summary(self) -> str:
        """
        Retorna un resumen del último entrenamiento ejecutado.

        Returns
        -------
        str
            Resumen con métricas de validación cruzada, mejores
            hiperparámetros y tiempos de entrenamiento.

        Raises
        ------
        RuntimeError
            Si se llama antes de ejecutar fit().
        """
        if self.results_ is None:
            raise RuntimeError(
                "No hay resultados disponibles. Llama a fit() primero."
            )

        r = self.results_
        lines = [
            "=" * 55,
            f"   RESULTADOS DE ENTRENAMIENTO — {r['model_name']}",
            "=" * 55,
            f"  Validación cruzada ({self.n_folds} folds, {self.scoring}):",
            f"    Scores por fold:  {np.round(r['cv_scores'], 4)}",
            f"    Media:            {r['cv_mean']:.4f}",
            f"    Desv. estándar:   {r['cv_std']:.4f}",
            f"    Tiempo CV:        {r['cv_time_sec']}s",
            "",
            f"  Ajuste de hiperparámetros (GridSearchCV):",
        ]

        if r["best_params"]:
            for k, v in r["best_params"].items():
                lines.append(f"    {k:<25} {v}")
            lines.append(f"    Mejor {self.scoring}:      {r['best_cv_score']:.4f}")
        else:
            lines.append(f"    Sin GridSearchCV (param_grid vacío)")

        lines += [
            "",
            f"  Tiempo GridSearchCV:  {r['grid_time_sec']}s",
            f"  Tiempo total:         {r['total_time_sec']}s",
            "=" * 55,
        ]

        return "\n".join(lines)

    def save_model(self, model, model_name: str, output_dir: str = "models") -> Path:
        """
        Serializa el modelo entrenado en disco usando joblib.

        Parameters
        ----------
        model : estimator
            Modelo entrenado a guardar.
        model_name : str
            Nombre del modelo. Se usa como nombre del archivo.
        output_dir : str, optional
            Directorio donde se guarda el modelo. Por defecto 'models/'.
            Se crea automáticamente si no existe.

        Returns
        -------
        Path
            Ruta completa del archivo guardado.

        Example
        -------
        >>> path = trainer.save_model(best_rf, 'random_forest')
        >>> print(f"Modelo guardado en: {path}")
        """
        import joblib

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{model_name}.joblib"
        joblib.dump(model, file_path)

        print(f"[{model_name}] Modelo guardado en: {file_path}")
        return file_path

    @staticmethod
    def load_model(model_name: str, output_dir: str = "models"):
        """
        Carga un modelo serializado desde disco.

        Parameters
        ----------
        model_name : str
            Nombre del modelo a cargar (sin extensión).
        output_dir : str, optional
            Directorio donde se encuentra el modelo. Por defecto 'models/'.

        Returns
        -------
        estimator
            Modelo cargado listo para predecir.

        Raises
        ------
        FileNotFoundError
            Si el archivo .joblib no existe en la ruta especificada.

        Example
        -------
        >>> model = Trainer.load_model('random_forest')
        >>> y_pred = model.predict(X_test)
        """
        import joblib

        file_path = Path(output_dir) / f"{model_name}.joblib"

        if not file_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {file_path}\n"
                f"Asegúrate de haber ejecutado trainer.save_model() primero."
            )

        model = joblib.load(file_path)
        print(f"[{model_name}] Modelo cargado desde: {file_path}")
        return model
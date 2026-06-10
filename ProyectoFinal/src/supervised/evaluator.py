"""
evaluator.py
============
Encapsula el cálculo de métricas, generación de visualizaciones e
interpretación de resultados para modelos de clasificación del
dataset ATUS 2024.

Métricas calculadas (requeridas por el proyecto):
    - Accuracy
    - Precision, Recall, F1-score (macro y ponderado)
    - Matriz de confusión
    - Curva ROC y AUC (one-vs-rest)

Visualizaciones generadas:
    - Matriz de confusión (seaborn/matplotlib)
    - Curva ROC (plotly)
    - Importancia de características (seaborn/matplotlib)

Todas las visualizaciones se guardan en:
    reports/model_evaluation/<model_name>/
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


class Evaluator:
    """
    Evaluador de modelos de clasificación para el dataset ATUS 2024.

    Calcula métricas completas, genera visualizaciones y produce el
    análisis de importancia de características como insumo para la
    etapa de agrupamiento no supervisado.

    Attributes
    ----------
    target_classes : list
        Clases originales de la variable objetivo en el orden del
        LabelEncoder (e.g. ['Fatal', 'No fatal', 'Sólo daños']).
    output_dir : Path
        Directorio base donde se guardan las visualizaciones.
    metrics_ : dict or None
        Métricas calculadas tras llamar a evaluate(). None hasta entonces.

    Example
    -------
    >>> evaluator = Evaluator(prep.target_classes)
    >>> evaluator.evaluate(best_xgb, X_test_t, y_test_enc)
    >>> evaluator.plot_confusion_matrix('xgboost')
    >>> evaluator.plot_roc_curve('xgboost', X_test_t, y_test_enc)
    >>> evaluator.plot_feature_importance(best_xgb, prep.feature_names, 'xgboost')
    >>> print(evaluator.summary())
    """

    def __init__(
        self,
        target_classes: list,
        output_dir: str = "reports/model_evaluation"
    ):
        """
        Parameters
        ----------
        target_classes : list
            Clases originales en el orden del LabelEncoder.
            Usar prep.target_classes.
        output_dir : str, optional
            Directorio base para guardar visualizaciones.
            Por defecto 'reports/model_evaluation'.
        """
        self.target_classes = target_classes
        self.output_dir     = Path(output_dir)
        self.metrics_       = None
        self._y_test        = None
        self._y_pred        = None
        self._y_prob        = None

    def _get_model_dir(self, model_name: str) -> Path:
        """Crea y retorna el directorio de salida para el modelo."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def evaluate(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Calcula métricas completas sobre el conjunto de prueba.

        Parameters
        ----------
        model : estimator
            Modelo entrenado.
        X_test : array-like
            Features de prueba transformadas.
        y_test : array-like
            Variable objetivo de prueba codificada.
        model_name : str
            Nombre del modelo, usado para organizar los archivos de salida.

        Returns
        -------
        dict
            Diccionario con todas las métricas calculadas.
        """
        self._model_name = model_name
        self._y_test     = np.array(y_test)
        self._y_pred     = model.predict(X_test)

        # Probabilidades para curva ROC (si el modelo las soporta)
        if hasattr(model, "predict_proba"):
            self._y_prob = model.predict_proba(X_test)
        else:
            self._y_prob = None

        self.metrics_ = {
            "model_name":       model_name,
            "accuracy":         accuracy_score(self._y_test, self._y_pred),
            "precision_macro":  precision_score(self._y_test, self._y_pred, average="macro",    zero_division=0),
            "precision_weighted": precision_score(self._y_test, self._y_pred, average="weighted", zero_division=0),
            "recall_macro":     recall_score(self._y_test, self._y_pred, average="macro",    zero_division=0),
            "recall_weighted":  recall_score(self._y_test, self._y_pred, average="weighted", zero_division=0),
            "f1_macro":         f1_score(self._y_test, self._y_pred, average="macro",    zero_division=0),
            "f1_weighted":      f1_score(self._y_test, self._y_pred, average="weighted", zero_division=0),
        }

        return self.metrics_

    def plot_confusion_matrix(
        self,
        model_name: str,
        y_pred_override: np.ndarray = None,
        filename: str = "confusion_matrix"
    ) -> Path:
        """
        Genera y guarda la matriz de confusión normalizada.

        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        y_pred_override : np.ndarray, optional
            Predicciones alternativas a usar en lugar de las del umbral
            por defecto. Útil para visualizar el efecto del ajuste de
            umbral. Si es None, usa las predicciones de evaluate().
        filename : str, optional
            Nombre del archivo de salida sin extensión. Por defecto
            "confusion_matrix". Usar un nombre distinto para guardar
            versiones con umbral ajustado sin sobreescribir la original.

        Returns
        -------
        Path
            Ruta del archivo guardado.

        Raises
        ------
        RuntimeError
            Si se llama antes de evaluate().
        """
        if self.metrics_ is None:
            raise RuntimeError("Llama a evaluate() primero.")

        y_pred = y_pred_override if y_pred_override is not None else self._y_pred
        title_suffix = " (umbral ajustado)" if y_pred_override is not None else ""

        cm      = confusion_matrix(self._y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Matriz de Confusión — {model_name}{title_suffix}",
            fontsize=14, fontweight="bold"
        )

        # Absoluta
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.target_classes,
            yticklabels=self.target_classes,
            ax=axes[0]
        )
        axes[0].set_title("Valores absolutos")
        axes[0].set_xlabel("Predicción")
        axes[0].set_ylabel("Real")

        # Normalizada
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=self.target_classes,
            yticklabels=self.target_classes,
            ax=axes[1]
        )
        axes[1].set_title("Normalizada por clase real")
        axes[1].set_xlabel("Predicción")
        axes[1].set_ylabel("Real")

        plt.tight_layout()

        output_path = self._get_model_dir(model_name) / f"{filename}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Guardada en: {output_path}")
        return output_path

    def plot_roc_curve(self, model_name: str, X_test, y_test) -> Path:
        """
        Genera y guarda la curva ROC one-vs-rest con Plotly.

        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        X_test : array-like
            Features de prueba transformadas.
        y_test : array-like
            Variable objetivo de prueba codificada.

        Returns
        -------
        Path
            Ruta del archivo guardado.

        Raises
        ------
        RuntimeError
            Si el modelo no soporta predict_proba o evaluate() no fue llamado.
        """
        if self._y_prob is None:
            raise RuntimeError(
                "El modelo no soporta predict_proba. "
                "No es posible generar la curva ROC."
            )

        n_classes  = len(self.target_classes)
        y_bin      = label_binarize(y_test, classes=list(range(n_classes)))
        colors     = ["#e74c3c", "#3498db", "#2ecc71"]

        fig = go.Figure()

        auc_scores = []
        for i, (clase, color) in enumerate(zip(self.target_classes, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], self._y_prob[:, i])
            auc_score   = auc(fpr, tpr)
            auc_scores.append(auc_score)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{clase} (AUC = {auc_score:.3f})",
                line=dict(color=color, width=2)
            ))

        # Línea diagonal de referencia
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Clasificador aleatorio",
            line=dict(color="gray", dash="dash", width=1)
        ))

        self.metrics_["auc_per_class"] = {
            clase: score
            for clase, score in zip(self.target_classes, auc_scores)
        }
        self.metrics_["auc_macro"] = np.mean(auc_scores)

        fig.update_layout(
            title=f"Curva ROC (One-vs-Rest) — {model_name}",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            legend=dict(x=0.6, y=0.1),
            width=700, height=500
        )

        output_path = self._get_model_dir(model_name) / "roc_curve.html"
        fig.write_html(str(output_path))
        fig.show()
        print(f"Guardada en: {output_path}")
        return output_path

    def plot_feature_importance(
        self,
        model,
        feature_names: list,
        model_name: str,
        top_n: int = 25
    ) -> Path:
        """
        Genera y guarda el gráfico de importancia de características.

        Solo disponible para modelos que exponen feature_importances_
        (Random Forest, XGBoost).

        Parameters
        ----------
        model : estimator
            Modelo entrenado con atributo feature_importances_.
        feature_names : list
            Nombres de las features. Usar prep.feature_names.
        model_name : str
            Nombre del modelo.
        top_n : int, optional
            Número de features más importantes a mostrar. Por defecto 25.

        Returns
        -------
        Path
            Ruta del archivo guardado.

        Raises
        ------
        AttributeError
            Si el modelo no tiene feature_importances_.
        """
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(
                f"El modelo '{model_name}' no expone feature_importances_. "
                f"Este análisis solo está disponible para Random Forest y XGBoost."
            )

        importances = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, top_n * 0.35))
        sns.barplot(
            x=importances.values,
            y=importances.index,
            palette="Blues_r",
            ax=ax
        )
        ax.set_title(
            f"Top {top_n} Features más importantes — {model_name}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Importancia")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        plt.tight_layout()

        output_path = self._get_model_dir(model_name) / "feature_importance.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Guardada en: {output_path}")

        # Guardamos también como CSV para el clustering
        csv_path = self._get_model_dir(model_name) / "feature_importance.csv"
        importances.reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        ).to_csv(csv_path, index=False)
        print(f"CSV guardado en: {csv_path}")

        return output_path

    def summary(self) -> str:
        """
        Retorna un resumen de las métricas calculadas.

        Returns
        -------
        str
            Resumen con todas las métricas del modelo.

        Raises
        ------
        RuntimeError
            Si se llama antes de evaluate().
        """
        if self.metrics_ is None:
            raise RuntimeError("Llama a evaluate() primero.")

        m = self.metrics_
        lines = [
            "=" * 55,
            f"   MÉTRICAS DE EVALUACIÓN — {m['model_name']}",
            "=" * 55,
            f"  Accuracy:             {m['accuracy']:.4f}",
            "",
            f"  {'Métrica':<20} {'Macro':>8} {'Ponderado':>10}",
            f"  {'-'*40}",
            f"  {'Precision':<20} {m['precision_macro']:>8.4f} {m['precision_weighted']:>10.4f}",
            f"  {'Recall':<20} {m['recall_macro']:>8.4f} {m['recall_weighted']:>10.4f}",
            f"  {'F1-score':<20} {m['f1_macro']:>8.4f} {m['f1_weighted']:>10.4f}",
        ]

        if "auc_macro" in m:
            lines += [
                "",
                f"  AUC macro:            {m['auc_macro']:.4f}",
                f"  AUC por clase:",
            ]
            for clase, score in m["auc_per_class"].items():
                lines.append(f"    {clase:<20} {score:.4f}")

        lines += [
            "",
            "  Reporte completo por clase:",
            classification_report(
                self._y_test, self._y_pred,
                target_names=self.target_classes,
                zero_division=0
            ),
            "=" * 55,
        ]

        return "\n".join(lines)
    def find_optimal_threshold(
        self,
        model_name: str,
        fatal_class_idx: int = 0,
        thresholds: np.ndarray = None
    ) -> dict:
        """
        Busca el umbral óptimo para la clase Fatal que maximiza F1-score macro.

        Para cada umbral candidato, clasifica un registro como Fatal si su
        probabilidad supera el umbral. Si no, asigna la clase con mayor
        probabilidad entre No fatal y Sólo daños. El umbral óptimo es el
        que maximiza el F1-score macro sobre el conjunto de prueba.

        Parameters
        ----------
        model_name : str
            Nombre del modelo, usado para guardar la visualización.
        fatal_class_idx : int, optional
            Índice de la clase Fatal en el LabelEncoder. Por defecto 0,
            ya que LabelEncoder ordena alfabéticamente: Fatal=0.
        thresholds : np.ndarray, optional
            Umbrales a evaluar. Por defecto np.arange(0.01, 0.51, 0.01).

        Returns
        -------
        dict
            Diccionario con el umbral óptimo, F1-score macro obtenido,
            y las predicciones ajustadas.

        Raises
        ------
        RuntimeError
            Si se llama antes de evaluate() o si el modelo no soporta
            predict_proba.
        """
        if self._y_prob is None:
            raise RuntimeError(
                "El modelo no soporta predict_proba. "
                "No es posible ajustar umbrales."
            )
        if self.metrics_ is None:
            raise RuntimeError("Llama a evaluate() primero.")

        if thresholds is None:
            thresholds = np.arange(0.01, 0.51, 0.01)

        f1_scores = []

        for t in thresholds:
            y_pred_adj = self._apply_threshold(t, fatal_class_idx)
            f1 = f1_score(self._y_test, y_pred_adj, average="macro", zero_division=0)
            f1_scores.append(f1)

        f1_scores    = np.array(f1_scores)
        best_idx     = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1      = f1_scores[best_idx]

        # Predicciones con el umbral óptimo
        y_pred_optimal = self._apply_threshold(best_threshold, fatal_class_idx)

        result = {
            "best_threshold":   round(best_threshold, 2),
            "best_f1_macro":    round(best_f1, 4),
            "default_f1_macro": round(self.metrics_["f1_macro"], 4),
            "improvement":      round(best_f1 - self.metrics_["f1_macro"], 4),
            "y_pred_optimal":   y_pred_optimal,
            "thresholds":       thresholds,
            "f1_scores":        f1_scores,
        }

        # Visualización de la curva umbral vs F1-score macro
        self._plot_threshold_curve(
            thresholds, f1_scores, best_threshold, best_f1,
            model_name
        )

        print(
            f"\n[{model_name}] Umbral óptimo para Fatal: {best_threshold:.2f}\n"
            f"  F1-score macro (default):  {self.metrics_['f1_macro']:.4f}\n"
            f"  F1-score macro (ajustado): {best_f1:.4f}\n"
            f"  Mejora:                   +{best_f1 - self.metrics_['f1_macro']:.4f}"
        )

        return result

    def _apply_threshold(
        self,
        threshold: float,
        fatal_class_idx: int
    ) -> np.ndarray:
        """
        Aplica un umbral específico para la clase Fatal.

        Parameters
        ----------
        threshold : float
            Umbral de probabilidad para predecir Fatal.
        fatal_class_idx : int
            Índice de la clase Fatal en y_prob.

        Returns
        -------
        np.ndarray
            Predicciones ajustadas con el umbral aplicado.
        """
        y_pred_adj = []
        n_classes  = self._y_prob.shape[1]

        for probs in self._y_prob:
            if probs[fatal_class_idx] >= threshold:
                y_pred_adj.append(fatal_class_idx)
            else:
                # Entre las clases restantes, elegir la de mayor probabilidad
                other_probs = [
                    (i, p) for i, p in enumerate(probs)
                    if i != fatal_class_idx
                ]
                best_class = max(other_probs, key=lambda x: x[1])[0]
                y_pred_adj.append(best_class)

        return np.array(y_pred_adj)

    def _plot_threshold_curve(
        self,
        thresholds: np.ndarray,
        f1_scores: np.ndarray,
        best_threshold: float,
        best_f1: float,
        model_name: str
    ) -> Path:
        """
        Genera y guarda la curva de umbral vs F1-score macro.

        Parameters
        ----------
        thresholds : np.ndarray
            Umbrales evaluados.
        f1_scores : np.ndarray
            F1-score macro para cada umbral.
        best_threshold : float
            Umbral óptimo encontrado.
        best_f1 : float
            Mejor F1-score macro obtenido.
        model_name : str
            Nombre del modelo.

        Returns
        -------
        Path
            Ruta del archivo guardado.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(thresholds, f1_scores, color="#3498db", linewidth=2,
                label="F1-score macro")
        ax.axvline(best_threshold, color="#e74c3c", linestyle="--",
                   label=f"Umbral óptimo = {best_threshold:.2f}")
        ax.axhline(best_f1, color="#e74c3c", linestyle=":", alpha=0.5)
        ax.scatter([best_threshold], [best_f1], color="#e74c3c", zorder=5,
                   s=80, label=f"Mejor F1-macro = {best_f1:.4f}")

        ax.set_title(
            f"Umbral de clasificación para Fatal vs F1-score macro — {model_name}",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Umbral para clase Fatal")
        ax.set_ylabel("F1-score macro")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path = self._get_model_dir(model_name) / "threshold_curve.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Guardada en: {output_path}")
        return output_path

    def evaluate_with_threshold(
        self,
        y_pred_optimal: np.ndarray,
        model_name: str
    ) -> str:
        """
        Muestra el reporte completo con las predicciones ajustadas por umbral.

        Parameters
        ----------
        y_pred_optimal : np.ndarray
            Predicciones ajustadas retornadas por find_optimal_threshold().
        model_name : str
            Nombre del modelo.

        Returns
        -------
        str
            Reporte de clasificación con umbral ajustado.

        Raises
        ------
        RuntimeError
            Si se llama antes de evaluate().
        """
        if self.metrics_ is None:
            raise RuntimeError("Llama a evaluate() primero.")

        report = classification_report(
            self._y_test, y_pred_optimal,
            target_names=self.target_classes,
            zero_division=0
        )

        f1_adj = f1_score(
            self._y_test, y_pred_optimal,
            average="macro", zero_division=0
        )

        lines = [
            "=" * 55,
            f"   MÉTRICAS CON UMBRAL AJUSTADO — {model_name}",
            "=" * 55,
            f"  F1-score macro (default):  {self.metrics_['f1_macro']:.4f}",
            f"  F1-score macro (ajustado): {f1_adj:.4f}",
            f"  Mejora:                   +{f1_adj - self.metrics_['f1_macro']:.4f}",
            "",
            "  Reporte completo por clase:",
            report,
            "=" * 55,
        ]

        return "\n".join(lines)
    def predict_with_threshold(
        self,
        y_prob: np.ndarray,
        threshold: float,
        fatal_class_idx: int = 0
    ) -> np.ndarray:
        """
        Aplica el umbral ajustado sobre probabilidades de nuevos registros.

        Método público para usar en producción o demo. No requiere haber
        llamado a evaluate() previamente — opera directamente sobre las
        probabilidades proporcionadas.

        Parameters
        ----------
        y_prob : np.ndarray
            Matriz de probabilidades por clase de shape (n_samples, n_classes),
            obtenida con model.predict_proba().
        threshold : float
            Umbral de probabilidad para predecir la clase Fatal.
            Usar el valor óptimo encontrado por find_optimal_threshold().
        fatal_class_idx : int, optional
            Índice de la clase Fatal en y_prob. Por defecto 0,
            ya que LabelEncoder ordena alfabéticamente: Fatal=0.

        Returns
        -------
        np.ndarray
            Predicciones numéricas con el umbral aplicado. Usar
            prep.inverse_transform_target() para obtener las etiquetas
            originales.

        Example
        -------
        >>> evaluator = Evaluator(prep.target_classes)
        >>> y_prob    = model.predict_proba(X_transformed)
        >>> y_pred    = evaluator.predict_with_threshold(y_prob, threshold=0.15)
        >>> labels    = prep.inverse_transform_target(y_pred)
        """
        y_pred = []

        for probs in y_prob:
            if probs[fatal_class_idx] >= threshold:
                y_pred.append(fatal_class_idx)
            else:
                other_probs = [
                    (i, p) for i, p in enumerate(probs)
                    if i != fatal_class_idx
                ]
                best_class = max(other_probs, key=lambda x: x[1])[0]
                y_pred.append(best_class)

        return np.array(y_pred)
# Proyecto Final: Almacenes y Minería de Datos

Este repositorio contiene la implementación técnica y el análisis derivado del Proyecto Final para la asignatura de Almacenes y Minería de Datos. La investigación se enfoca en el análisis exhaustivo de los **Accidentes de Tránsito Terrestre en Zonas Urbanas y Suburbanas de México**, empleando la base de datos oficial ATUS 2024 provista por el Instituto Nacional de Estadística y Geografía (INEGI).

## 1. Objetivos del Estudio

El propósito central de este análisis es la identificación de patrones subyacentes y factores de riesgo determinantes en la severidad y mortalidad de los accidentes de tránsito. Para lograrlo, se ha desarrollado un pipeline integral de ciencia de datos que comprende:

*   **Limpieza de Datos y Análisis Exploratorio:** Preprocesamiento, tratamiento de valores nulos, estandarización de formatos y evaluación inicial de las variables del conjunto de datos.
*   **Análisis Descriptivo y Estadístico:** Exploración de variables espaciotemporales y factores humanos mediante las bibliotecas `pandas`, `plotly` y `seaborn`.
*   **Aprendizaje No Supervisado (Clustering):** Aplicación del algoritmo K-Means, complementado con técnicas de reducción de dimensionalidad (PCA y t-SNE), para el descubrimiento de perfiles intrínsecos de siniestralidad.
*   **Modelado Predictivo Supervisado:** Entrenamiento, validación cruzada y evaluación de algoritmos de clasificación (Random Forest, XGBoost, Regresión Logística) orientados a inferir la severidad del accidente y predecir el comportamiento del conductor (e.g., probabilidad de fuga).

---

## 2. Arquitectura del Repositorio

La base de código está estructurada bajo principios de modularidad y reproducibilidad, dividiéndose en los siguientes componentes:

*   `data/`: Directorio de almacenamiento de datos. El conjunto de datos principal (`atus_anual_2024.csv`) se gestiona a través de **Git LFS** debido a sus requerimientos de almacenamiento.
*   `notebooks/`: Entornos interactivos Jupyter para la ejecución paso a paso de los procesos analíticos:
    *   `limpieza.ipynb`: Preprocesamiento, imputación de valores nulos y estandarización de dominios.
    *   `analisis_exploratorio.ipynb` / `analisis_descriptivo_estadistico.ipynb`: Extracción de insights e inferencia estadística.
    *   `modelado.ipynb`: Flujo de entrenamiento y evaluación de los modelos supervisados.
    *   `agrupamiento.ipynb`: Ejecución y evaluación métrica de los clústeres.
*   `src/`: Código fuente modular en Python. Implementa patrones de diseño mediante clases orquestadoras (Trainers, Evaluators, Preprocessors), subdividido en las arquitecturas `supervised/` y `clustering/`.
*   `docs/`: Directorio de despliegue configurado para **GitHub Pages**. El código fuente de renderizado se encuentra en `docs/pagina_interactiva/`.
*   `reports/`: Presentaciones dinámicas en formato RevealJS, ubicadas en `reports/presentacion/`, compiladas mediante Quarto.

---

## 3. Instrucciones de Instalación y Despliegue Local

Para garantizar la reproducibilidad exacta de los resultados y la correcta ejecución del entorno, se requiere seguir la siguiente metodología de despliegue:

### 3.1 Prerrequisitos del Sistema
*   **Python 3.10+**
*   **Git** y **Git LFS** (Large File Storage)
*   **Quarto CLI** (Para la compilación de la documentación interactiva)

**Instalación de Git LFS:**
```bash
# Entornos basados en Debian/Ubuntu
sudo apt install git-lfs

# Entornos basados en RHEL/Fedora
sudo dnf install git-lfs

# macOS (vía Homebrew)
brew install git-lfs

# Windows (vía Chocolatey)
choco install git-lfs
```
Posterior a la instalación, es imperativo inicializar Git LFS en el sistema: 
```bash
git lfs install
```

### 3.2 Clonación y Sincronización de Datos
Durante el proceso de clonación, se debe forzar la descarga de los blobs gestionados por Git LFS:
```bash
git clone <URL_DEL_REPOSITORIO>
cd ProyectoFinal
git lfs pull
```

### 3.3 Aislamiento del Entorno (Virtual Environment)
Se recomienda estrictamente la creación de un entorno virtual para aislar las dependencias del proyecto de los paquetes globales del sistema:

```bash
# Inicialización del entorno virtual
python3 -m venv .venv
```

**Activación del entorno:**
*   Sistemas UNIX (Linux / macOS):
    ```bash
    source .venv/bin/activate
    ```
*   Sistemas Windows (PowerShell):
    ```powershell
    .\.venv\Scripts\Activate.ps1
    ```

### 3.4 Resolución de Dependencias
Con el entorno virtual activado, proceda a la instalación de los requerimientos especificados:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
> *Nota Técnica:* La configuración de Quarto (`_quarto.yml`) está preestablecida para detectar y utilizar automáticamente la instancia de Python del entorno virtual activo.

---

## 4. Ejecución y Reproducción de Resultados

El repositorio ofrece dos vías principales para la evaluación de los resultados técnicos:

### 4.1 Ejecución Interactiva (Jupyter Notebooks)
Inicie su servidor Jupyter local (`jupyter notebook`) o IDE compatible (e.g., VS Code) y asigne como kernel el entorno `.venv` configurado en el paso anterior. Los notebooks ubicados en la carpeta `notebooks/` están diseñados para ejecutarse secuencialmente sin requerir configuraciones adicionales.

### 4.2 Compilación del Informe Interactivo (Quarto)
El proyecto incluye un informe técnico dinámico y navegable. Para generar los artefactos HTML correspondientes en el directorio `docs/`:
```bash
quarto render docs/pagina_interactiva
```

### 4.3 Compilación de la Presentación Ejecutiva
Para compilar la presentación técnica estructurada en formato RevealJS:
```bash
quarto render reports/presentacion
```
Los archivos compilados residirán en: `reports/presentacion/_site/index.html`

---

## 5. Gestión del Control de Versiones
El archivo `.gitignore` se ha configurado rigurosamente para excluir artefactos compilados locales (`_site/`, `.quarto/`), el directorio del entorno virtual (`.venv/`), memorias caché (`__pycache__`) y binarios de modelos serializados (`.joblib`), garantizando la integridad y ligereza del repositorio fuente.

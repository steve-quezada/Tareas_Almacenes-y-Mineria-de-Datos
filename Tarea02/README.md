# Tarea 02 — Análisis de Personas Desaparecidas en México
<!-- Almacenes y Minería de Datos · Semestre 2026-2 · Facultad de Ciencias, UNAM -->

### Integrantes
- Omar Fernando Gramer Muñoz
- Kevin Steve Quezada Ordoñez
- Jesús Antonio Romero Godoy
- Edgar Salgado González

---

## Descripción

Análisis exploratorio y de minería de datos sobre el Registro Nacional de Personas Desaparecidas y No Localizadas (RNPDNO), aplicando el proceso CRISP-DM. Se construye un pipeline de limpieza sobre 133,887 registros, se calculan medidas descriptivas y se analizan correlaciones.

El resultado se publica como un libro Quarto con ocho capítulos: metodologías, descripción del dataset, pipeline de limpieza, medidas descriptivas, análisis de correlación, hallazgos y exposición.

**Dataset:** `data_secretariado.csv` · 133,887 filas × 11 columnas · Periodo 1961–2025 · Equipo 7.

---

## Estructura del proyecto

```
├── README.md                          ← Este archivo
├── _quarto.yml                        ← Configuración del libro Quarto
├── index.qmd                          ← Página de inicio del libro
├── environment.yml                    ← Entorno conda
├── requirements.txt                   ← Dependencias pip
│
├── data/
│   ├── raw/                           ← Dataset original [data_secretariado.csv]
│   └── clean/                         ← CSV limpios generados por el pipeline
│
├── notebooks/
│   ├── 01_metodologia.qmd             ← Sección I:    Metodologías de minería de datos
│   ├── 02_dataset.qmd                 ← Sección II:   Descripción del dataset y CRISP-DM
│   ├── 03_exploracion_limpieza.qmd    ← Sección V:    Pipeline de limpieza
│   ├── 04_medidas_eda.qmd             ← Sección III:  Medidas descriptivas y EDA
│   ├── 05_correlacion_analisis.qmd    ← Sección IV:   Correlación
│   ├── 06_hallazgos.qmd               ← Sección VII:  Hallazgos y limitaciones
│   └── 07_exposicion.qmd              ← Sección VIII: Exposición
│
└── entregables/
    ├── Tarea 2 (EDA - Personas desaparecidas en México).pdf  ← Reporte compilado
    └── medidas_dataframe_desapariciones.ipynb                ← Notebook [Google Colab]
```

---

## Instalación

### Requisitos previos

- **Python 3.11+**
- **Quarto CLI** ≥ 1.4 — [Instalar Quarto](https://quarto.org/docs/get-started/)


### Opción A: Conda/MiniConda

```bash
conda env create -f environment.yml
conda activate almacenes-mineria
```

### Opción B: pip

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```


---

## Instalación y ejecución

### 1. Colocar el dataset

Descargar `data_secretariado.csv` y colocarlo en:

```
data/raw/data_secretariado.csv
```

### 2. Renderizar el libro

```bash
quarto render
```

### 3. Ver el resultado

```
_book/index.html
```

---

## Pipeline de ejecución

1. **03_exploracion_limpieza.qmd** — carga el CSV crudo, analiza la calidad de cada columna, detecta el bloque de registros `CONFIDENCIAL`, parsea y valida fechas, tipado de columnas y crea banderas de integridad temporal; exporta dos CSV limpios a `data/clean/`.
2. **04_medidas_eda.qmd** — carga los CSV limpios y calcula medidas de tendencia central, dispersión, asimetría, curtosis y genera visualizaciones EDA.
3. **05_correlacion_analisis.qmd** — carga los CSV limpios y calcula matrices de correlación de Pearson y Spearman.
4. Los capítulos de texto (01, 02, 06, 07) no ejecutan código Python.

---

## Datasets generados

Después de renderizar se crean dos CSV limpios en `data/clean/`:

| Archivo | Filas | Columnas | Descripción |
|---------|------:|:--------:|-------------|
| `data_secretariado_limpio_general.csv` | 133,858 | 16 | Dataset completo (incluye registros CONFIDENCIAL) |
| `data_secretariado_limpio_temporal.csv` | 84,709 | 18 | Solo registros con fechas válidas (sin CONFIDENCIAL) |

---
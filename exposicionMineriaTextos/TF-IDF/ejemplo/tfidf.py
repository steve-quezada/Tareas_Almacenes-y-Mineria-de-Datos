from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ARCHIVO_DATOS = Path(__file__).with_name("datos.csv")
ARCHIVO_MATRIZ = Path(__file__).with_name("matriz_tfidf.csv")
ARCHIVO_TOP = Path(__file__).with_name("palabras_representativas.csv")

# Stopwords basicas en espanol. Se pueden ampliar segun el corpus.
STOPWORDS = [
    "a",
    "al",
    "ante",
    "como",
    "con",
    "contra",
    "de",
    "del",
    "desde",
    "durante",
    "e",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "es",
    "esa",
    "ese",
    "eso",
    "esta",
    "este",
    "esto",
    "la",
    "las",
    "lo",
    "los",
    "mas",
    "más",
    "muy",
    "o",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "sin",
    "sobre",
    "su",
    "sus",
    "tambien",
    "también",
    "un",
    "una",
    "unas",
    "unos",
    "y",
]


def cargar_datos(ruta_csv):
    """Carga el CSV y valida que existan las columnas esperadas."""
    datos = pd.read_csv(ruta_csv)

    columnas_necesarias = {"id", "texto"}
    columnas_faltantes = columnas_necesarias - set(datos.columns)

    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el CSV: {columnas_faltantes}")

    datos["texto"] = datos["texto"].fillna("").astype(str)
    return datos


def construir_vectorizador():
    """
    Configura TfidfVectorizer.

    Este objeto hace internamente:
    1. Limpieza basica del texto.
    2. Tokenizacion.
    3. Conteo de terminos.
    4. Calculo de TF-IDF.
    """
    return TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-záéíóúñü][a-záéíóúñü]+\b",
        ngram_range=(1, 2),
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
    )


def calcular_tfidf(datos):
    """Calcula la matriz TF-IDF y regresa tambien los nombres de terminos."""
    vectorizador = construir_vectorizador()
    matriz_tfidf = vectorizador.fit_transform(datos["texto"])
    terminos = vectorizador.get_feature_names_out()

    return matriz_tfidf, terminos


def crear_matriz_completa(datos, matriz_tfidf, terminos):
    """Convierte la matriz dispersa de sklearn en una tabla legible."""
    matriz = pd.DataFrame(
        matriz_tfidf.toarray(),
        columns=terminos,
        index=datos["id"],
    )

    matriz.index.name = "id"
    return matriz.round(6)


def obtener_palabras_representativas(datos, matriz_tfidf, terminos, top_n=5):
    """Obtiene las palabras o frases con mayor TF-IDF en cada documento."""
    resultados = []

    for posicion_documento, documento in datos.iterrows():
        fila = matriz_tfidf[posicion_documento].toarray().ravel()
        indices_ordenados = fila.argsort()[::-1]

        for ranking, indice_termino in enumerate(indices_ordenados[:top_n], start=1):
            resultados.append(
                {
                    "id": documento["id"],
                    "ranking": ranking,
                    "termino": terminos[indice_termino],
                    "tfidf": round(float(fila[indice_termino]), 6),
                    "texto": documento["texto"],
                }
            )

    return pd.DataFrame(resultados)


def mostrar_palabras_representativas(palabras_representativas):
    """Imprime resultados resumidos en consola."""
    for id_documento, grupo in palabras_representativas.groupby("id", sort=False):
        texto = grupo["texto"].iloc[0]

        print(f"\nDocumento {id_documento}")
        print(f"Texto: {texto}")
        print("Terminos con mayor TF-IDF:")

        for _, fila in grupo.iterrows():
            print(f"  {fila['termino']:<28} TF-IDF = {fila['tfidf']:.6f}")


def main():
    datos = cargar_datos(ARCHIVO_DATOS)
    matriz_tfidf, terminos = calcular_tfidf(datos)

    matriz_completa = crear_matriz_completa(datos, matriz_tfidf, terminos)
    matriz_completa.to_csv(ARCHIVO_MATRIZ, encoding="utf-8")

    palabras_representativas = obtener_palabras_representativas(
        datos,
        matriz_tfidf,
        terminos,
        top_n=5,
    )
    palabras_representativas.to_csv(ARCHIVO_TOP, index=False, encoding="utf-8")

    mostrar_palabras_representativas(palabras_representativas)

    print(f"\nMatriz TF-IDF guardada en: {ARCHIVO_MATRIZ}")
    print(f"Top terminos por documento guardado en: {ARCHIVO_TOP}")


if __name__ == "__main__":
    main()

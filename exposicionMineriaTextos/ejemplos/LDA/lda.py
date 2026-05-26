from pathlib import Path
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Definición de rutas relativas al archivo de ejecución (.py)
ARCHIVO_DATOS = Path(__file__).with_name("datos.csv")
ARCHIVO_PROBABILIDADES = Path(__file__).with_name("probabilidades_topicos.csv")
ARCHIVO_FRECUENCIAS = Path(__file__).with_name("matriz_frecuencias.csv")

# Stopwords diseñadas como un filtro absoluto para forzar la ortogonalidad en LDA
STOPWORDS = [
    "la", "de", "indica", "batir", "para", "el", "mete", "al", "terminar", 
    "esta", "lleva", "en", "calienta", "tu", "consiste", "realizar", "un", 
    "cada", "genera", "una", "asiatico", "es", "vital", "arma", "usando", 
    "artesanal", "usa", "hacer", "trabajar", "requiere", "poner", "sobre", 
    "gira", "con", "y", "obtendras", "trabaja", "del", "corta", "construir", 
    "lijas", "taller", "las", "los", "se", "este", "su", "sus", "por", "a", "o"
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
    Configura CountVectorizer.
    
    LDA requiere una matriz de conteo de frecuencias (valores enteros)
    en lugar de una matriz con pesos continuos como TF-IDF.
    """
    return CountVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-záéíóúñü][a-záéíóúñü]+\b",
    )


def calcular_dtm(datos):
    """Calcula la matriz de conteos (DTM) y regresa el DataFrame legible y los términos."""
    vectorizador = construir_vectorizador()
    matriz_conteos = vectorizador.fit_transform(datos["texto"])
    terminos = vectorizador.get_feature_names_out()

    # Convertir la DTM dispersa en un DataFrame estructurado
    df_dtm = pd.DataFrame(
        matriz_conteos.toarray(), 
        columns=terminos, 
        index=datos["id"]
    )
    df_dtm.index.name = "id"

    return matriz_conteos, df_dtm, terminos


def ajustar_lda(matriz_conteos):
    """
    Instancia y entrena el modelo LDA con hiperparámetros estrictos
    """
    k = 4
    lda_model = LatentDirichletAllocation(
        n_components=k,
        doc_topic_prior=0.1,      # Alpha bajo: fuerza asignación a un único tópico dominante
        topic_word_prior=0.1,     # Beta bajo: reduce la distribución cruzada de palabras entre tópicos
        learning_method="batch",  # Evalúa todo el dataset de golpe 
        max_iter=1000,            # Asegura iteraciones suficientes para estabilizar probabilidades
        random_state=42,          # Semilla fija
    )
    
    matriz_probabilidades = lda_model.fit_transform(matriz_conteos)
    return lda_model, matriz_probabilidades


def crear_matriz_probabilidades(datos, matriz_probabilidades):
    """Convierte la matriz de distribución documento-tópico en una tabla legible."""
    k = matriz_probabilidades.shape[1]
    columnas_topicos = [f"Tópico {i+1}" for i in range(k)]
    
    matriz = pd.DataFrame(
        matriz_probabilidades,
        columns=columnas_topicos,
        index=datos["id"],
    )
    matriz.index.name = "id"
    return matriz.round(4)


def obtener_palabras_representativas(lda_model, terminos, top_n=5):
    """Obtiene las palabras con mayor peso asignado por el modelo para cada tópico."""
    resultados = []

    for id_topico, componente in enumerate(lda_model.components_, start=1):
        indices_ordenados = componente.argsort()[::-1]
        
        for ranking, indice_termino in enumerate(indices_ordenados[:top_n], start=1):
            resultados.append(
                {
                    "Tópico": f"Tópico {id_topico}",
                    "Ranking": ranking,
                    "Termino": terminos[indice_termino],
                    "Peso Psicológico": round(float(componente[indice_termino]), 4),
                }
            )

    return pd.DataFrame(resultados)


def mostrar_resultados_consola(matriz_probabilidades, df_dtm, palabras_representativas):
    """Imprime los componentes calculados directamente en la terminal."""
    print("\n" + "=" * 50)
    print("--- MATRIZ DE PESOS ASIGNADOS (PROBTOPICOS) ---")
    print("=" * 50)
    print(matriz_probabilidades)
    
    print("\n" + "=" * 50)
    print("--- MATRIZ DE TÉRMINOS POR DOCUMENTO (DTM) ---")
    print("=" * 50)
    print(df_dtm)
    
    print("\n" + "=" * 50)
    print("--- PALABRAS MÁS IMPORTANTES POR TÓPICO ---")
    print("=" * 50)
    for id_topico, grupo in palabras_representativas.groupby("Tópico", sort=False):
        print(f"\n{id_topico}: ", end="")
        palabras = [f"{fila['Termino']}" for _, fila in grupo.iterrows()]
        print(", ".join(palabras))


def main():
    # 1. Pipeline de carga y transformaciones vectoriales
    datos = cargar_datos(ARCHIVO_DATOS)
    matriz_conteos, df_dtm, terminos = calcular_dtm(datos)
    
    # 2. Modelado Matemático con LDA
    lda_model, matriz_probabilidades = ajustar_lda(matriz_conteos)

    # 3. Construcción y exportación de DataFrames resultantes
    df_probabilidades = crear_matriz_probabilidades(datos, matriz_probabilidades)
    palabras_representativas = obtener_palabras_representativas(lda_model, terminos, top_n=5)
    
    # Exportaciones físicas a la carpeta de ejecución
    df_probabilidades.to_csv(ARCHIVO_PROBABILIDADES, encoding="utf-8")
    df_dtm.to_csv(ARCHIVO_FRECUENCIAS, encoding="utf-8")

    # 4. Salida en consola y despliegue gráfico
    mostrar_resultados_consola(df_probabilidades, df_dtm, palabras_representativas)
    
    print(f"\n[OK] Asignación de tópicos guardada en: {ARCHIVO_PROBABILIDADES}")
    print(f"[OK] Matriz de frecuencias guardada en: {ARCHIVO_FRECUENCIAS}")
    



if __name__ == "__main__":
    main()
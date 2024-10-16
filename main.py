from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title='PROYECTO INDIVIDUAL 1', description='Florencia Zapata')

# Crear punto de entrada o endpoint:
@app.get("/", tags=["MLOps"])
def mensaje():
    content = """
    <h2> Bienvenido al PI_MLOps_Engineer </h2>
    <p>Accede a la documentación:</p>
    <ul>
        <li><a href='http://127.0.0.1:8000/docs' >Local (FastAPI)</a></li>
        <li><a href='https://pi-mlops-25.onrender.com/docs' >Producción (Render)</a></li>
    </ul>
    """
    return HTMLResponse(content=content)

# Cargamos los archivos parquet
df1 = pd.read_parquet('endpoint1.parquet')  # endpoint 1/def developer
recomendacion = pd.read_parquet('recomendacionjuego.parquet')  # endpoint 6/similitud del coseno
games = pd.read_parquet('random_sample.parquet')  # muestra aleatoria de steam games
reviews = pd.read_parquet('random_sample_reviews.parquet')  # muestra aleatoria de reviews
items = pd.read_parquet('australian_users_items.parquet')  # muestra aleatoria de items

@app.get("/UserForGenre/", tags=["Funciones"])
def UserForGenre(genero: str):
    """
    Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento. Ejemplo de género: Indie.
    """
    # Convertir genero a minúsculas para que cuando el usario lo escriba, sin importar si es mayúscula o minuscula nos de resultado
    genero = genero.lower()

    # Combinar los df de juegos e items usando la columna de id
    merged_df = games.merge(items, left_on='id_steam', right_on='id_item')

    # Convertir la columna "genero" a string y minúscula
    merged_df["genero"] = merged_df["genero"].astype(str).str.lower()

    # Verificar si la palabra escrita corresponde a un género de nuestra base de datos
    if genero not in merged_df['genero'].unique():
        return JSONResponse(content=f"El género '{genero}' no existe. Prueba ingresando otro, como por ejemplo: Action, Casual, Indie.")

    # Filtrar juegos por género
    filter_games = merged_df[merged_df['genero'] == genero]

    # Calcular el usuario con más horas jugadas
    user_playtime = filter_games.groupby('id_usuario')['tiempo_total_de_juego'].sum().reset_index()
    max_playtime_user = user_playtime.loc[user_playtime['tiempo_total_de_juego'].idxmax()]['id_usuario']

    # Filtrar las filas correspondientes al usuario con más horas jugadas
    user_games = filter_games[filter_games['id_usuario'] == max_playtime_user]

    # Agrupar las horas jugadas por año de lanzamiento
    user_games['release_year'] = pd.to_datetime(user_games['fecha']).dt.year
    playtime_per_year = user_games.groupby('release_year')['tiempo_total_de_juego'].sum().reset_index()

    # Convertir a la estructura requerida
    playtime_per_year_list = [{"Año": row['release_year'], "Horas": row['tiempo_total_de_juego']} for index, row in playtime_per_year.iterrows()]

    # Devolver el resultado
    return JSONResponse(content={f"Usuario con más horas jugadas para Género {genero}": max_playtime_user, "Horas jugadas": playtime_per_year_list})

@app.get('/Developer/', tags=['Funciones'])
def developer(desarrollador: str):
    """
    Devuelve cantidad de items y porcentaje de contenido Free por año según desarrollador. Ej de desarrollador: kotoshiro
    """
    desarrollador = desarrollador.lower()
    
    filter_games = df1[df1['desarrollador'].str.lower() == desarrollador]
    if filter_games.empty:
        return JSONResponse(content=f"No hay datos disponibles para el desarrollador '{desarrollador}'. Asegúrate de haber escrito correctamente el nombre.")
    
    filter_games = filter_games.groupby('fecha').agg(
        total_items=('id_steam', 'count'),
        free_game=('precio', lambda x: (x == 0.0).sum())
    ).reset_index()
    
    filter_games['Contenido Free'] = round((filter_games['free_game'] / filter_games['total_items']) * 100, 2)

    anio = filter_games['fecha'].tolist()
    cantidad_items = filter_games['total_items'].tolist()
    cont_free = filter_games['Contenido Free'].tolist()

    result = [{'Año': anio, 'Cantidad de items': items, 'Contenido Free': free} for anio, items, free in zip(anio, cantidad_items, cont_free)]
    return JSONResponse(content=result)

@app.get("/UserData/", tags=["Funciones"])
def userdata(id_usuario: str):
    """
    Devuelve cantidad de dinero gastado, porcentaje de recomendación y cantidad de items para un usuario. Ej: evcentric, js41637, doctr
    """
    game = games[['id_steam', 'precio']]
    user_items = items[items['id_usuario'] == id_usuario]
    
    if user_items.empty:
        return JSONResponse(content={'mensaje': f"El usuario '{id_usuario}' no existe. Intente con otro usuario."})
    
    user_reviews = reviews[reviews['id_usuario'] == id_usuario]
    recomend = round(user_reviews['recomendación'].mean() * 100, 2) if not user_reviews.empty else 0.0
    
    user_items = user_items.merge(game, left_on='id_item', right_on='id_steam', how='inner')
    if user_items.empty:
        return JSONResponse(content={'mensaje': f"El usuario '{id_usuario}' no existe. Intente con otro usuario."})
    
    aggregated_data = user_items.groupby('id_usuario').agg({'precio': 'sum', 'items_count': 'sum'}).reset_index()
    if aggregated_data.empty:
        return JSONResponse(content={'mensaje': f"El usuario '{id_usuario}' no existe. Intente con otro usuario."})
    
    total_dinero = float(aggregated_data['precio'].iloc[0]) if not aggregated_data.empty else 0.0
    total_items = int(user_items['items_count'].unique()[0]) if not user_items.empty else 0
    
    return JSONResponse(content={
        'Usuario': id_usuario,
        'Dinero gastado': round(total_dinero, 2),
        'Cantidad de items': total_items,
        '% de recomendacion': recomend
      })
@app.get("/BestDeveloperYear/", tags=["Funciones"])
def best_developer_year(year: int):
    """
    Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. Ej: 2017
    """
    game = games[['desarrollador', 'id_steam', 'fecha']]
    if game.empty:
        return JSONResponse(content={'mensaje': f"No han habido desarrollos para el año {year}, prueba ingresando otro año."})

    # Filtrar los juegos por año dado
    game_year = game[game['fecha'] == year]
    if game_year.empty:
        return JSONResponse(content={'mensaje': f"No han habido desarrollos para el año {year}, prueba ingresando otro año."})

    # Filtrar columnas del df reviews
    review = reviews[['id_item', 'recomendación']]
    
    reviews_filter = game_year.merge(review, left_on='id_steam', right_on='id_item', how='inner')
    if reviews_filter.empty:
        return JSONResponse(content={'mensaje': f"No hay reviews para los juegos lanzados en el año {year}."})
    
    reviews_filter = reviews_filter.merge(recomendacion, left_on='id_steam', right_on='id_steam', how='inner')
    reviews_filter = reviews_filter[['desarrollador', 'titulo', 'recomendación']]
    reviews_filter = reviews_filter.groupby(['desarrollador', 'titulo']).agg({'recomendación': 'sum'}).reset_index()
    reviews_filter = reviews_filter.sort_values(by='recomendación', ascending=False)

    result = []
    # Obtener los puestos
    puestos = ['Puesto 1', 'Puesto 2', 'Puesto 3']
    for puesto in puestos:
        if len(reviews_filter) >= int(puesto[-1]):
            developer_name = reviews_filter.iloc[int(puesto[-1]) - 1, 0]
            result.append({puesto: developer_name})
        else:
            result.append({puesto: reviews_filter.iloc[0, 0]})
    
    return JSONResponse(content=result)



@app.get("/DeveloperReviewsAnalysis/", tags=["Funciones"])
def developer_reviews_analysis(desarrolladora: str):
    """
    Devuelve el nombre del desarrollador y la cantidad total de registros de reseñas positivas y negativas. Ejemplo: SCS Software
    """
    # Filtramos los juegos del desarrollador
    develop = df1[['desarrollador', 'id_steam']]
    develop = develop[develop['desarrollador'].str.lower() == desarrolladora.lower()]
    
    # Verificamos si el desarrollador existe
    if develop.empty:
        return JSONResponse(content={"mensaje": f"No existe este desarrollador, asegúrese de haber escrito bien el nombre {desarrolladora}."})
    
    # Seleccionamos las columnas necesarias de las reviews
    review = reviews[['id_item', 'analisis_de_sentimientos']]
    
    # Conectamos las dos tablas
    develop_filter = develop.merge(review, left_on='id_steam', right_on='id_item', how='inner')
    
    # Verificamos si hay reviews para los juegos del desarrollador
    if develop_filter.empty:
        return JSONResponse(content={"mensaje": f"No hay reviews para los juegos lanzados por este desarrollador, prueba con otro desarrollador."})
    
    # Contamos las reseñas positivas y negativas
    develop_filter = develop_filter.groupby(['desarrollador']).agg({'analisis_de_sentimientos': lambda x: x.value_counts().to_dict()}).reset_index()
    
    # Extraemos los resultados
    developer = develop_filter['desarrollador'].iloc[0]
    sentiment_counts = develop_filter['analisis_de_sentimientos'].iloc[0]
    reviews_positivas = sentiment_counts.get('Positive', 0)
    reviews_negativas = sentiment_counts.get('Negative', 0)
    
    return JSONResponse(content={developer: {"Negative": reviews_negativas, "Positive": reviews_positivas}})


@app.get("/RecomendacionJuego/{id_producto}", tags=["Funciones"])
def calcular_similitud(id_producto: int):
    """
    Sistema de recomendación por similitud del coseno.
    Ingresando el id de un producto, recibimos 5 juegos recomendados similares al ingresado. Ej: 670290, 610660  
    """
    # Combina las columnas 'genero', 'etiqueta' y 'especificaciones' en una sola columna
    recomendacion['combined'] = recomendacion.apply(lambda row: f"{row['genero']}, {row['etiqueta']}, {row['especificaciones']}", axis=1)
    
    # Vectorización
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(recomendacion['combined'])
    
    try:
        product_index = recomendacion[recomendacion['id_steam'] == id_producto].index[0]
    except IndexError:
        return JSONResponse(content={'mensaje': 'No se encuentra un juego con el ID proporcionado, prueba con otro ID'})

    product_vector = matrix[product_index]
    cosine_similarity_matrix = cosine_similarity(product_vector, matrix)

    # Obtenemos la similitud con otros items
    product_similarities = cosine_similarity_matrix[0]

    # Obtenemos los índices de los primeros 5 items más similares y luego sus nombres
    most_similar_products_indices = np.argsort(-product_similarities)[1:6]
    
    most_similar_products = recomendacion.loc[most_similar_products_indices, 'titulo']

    return JSONResponse(content=most_similar_products.tolist())
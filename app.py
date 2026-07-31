"""
=========================================================
 APP STREAMLIT — ANÁLISIS EXPLORATORIO DE VIDEOJUEGOS
=========================================================
Ejecutar con:   streamlit run app.py

Estructura:
  0. Configuración de la página
  1. Generación de datos sintéticos
  2. Filtros interactivos (sidebar)
  3. EDA: cuantitativo, cualitativo y gráficos
  4. Interacción del usuario (explorador + descarga)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================================================
# 0. CONFIGURACIÓN DE LA PÁGINA
# =========================================================
st.set_page_config(
    page_title="EDA Videojuegos",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Análisis Exploratorio de Datos — Videojuegos")
st.caption("Datos sintéticos generados con NumPy · Proyecto académico de Ingeniería de Sistemas")


# =========================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# =========================================================
# @st.cache_data evita regenerar los datos en cada interacción.
# Solo se recalcula si cambian los argumentos (n, semilla).
@st.cache_data
def generar_datos(n: int = 600, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # ----- Variables CUALITATIVAS (categóricas) -----
    generos = ["Acción", "Aventura", "RPG", "Deportes", "Estrategia", "Shooter", "Simulación", "Puzzle"]
    p_generos = [0.20, 0.14, 0.13, 0.11, 0.09, 0.15, 0.10, 0.08]

    plataformas = ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile"]
    p_plataformas = [0.28, 0.22, 0.17, 0.15, 0.18]

    desarrolladores = ["Nova Studios", "PixelForge", "Andes Games", "Kraken Interactive",
                       "Bit Republic", "Hydra Soft", "Indie Solo"]
    clasificaciones = ["E", "E10+", "T", "M"]

    genero = rng.choice(generos, size=n, p=p_generos)
    plataforma = rng.choice(plataformas, size=n, p=p_plataformas)
    desarrollador = rng.choice(desarrolladores, size=n)
    clasificacion = rng.choice(clasificaciones, size=n, p=[0.25, 0.25, 0.30, 0.20])
    multijugador = rng.choice([True, False], size=n, p=[0.55, 0.45])

    # ----- Variables CUANTITATIVAS -----
    anio = rng.integers(2005, 2026, size=n)

    # Puntuación de la crítica (0-100), distribución normal recortada
    critica = np.clip(rng.normal(72, 12, size=n), 30, 99).round(0)

    # Puntuación de usuarios: correlacionada con la crítica + ruido
    usuarios = np.clip(critica / 10 + rng.normal(0, 1.1, size=n), 1, 10).round(1)

    # Ventas: crecen exponencialmente con la calidad (relación no lineal)
    ventas = (np.exp((critica - 70) / 18) * rng.lognormal(0, 0.55, size=n)).round(2)

    # Precio: depende de la plataforma (Mobile más barato)
    base_precio = {"PC": 45, "PlayStation": 60, "Xbox": 60, "Nintendo Switch": 55, "Mobile": 8}
    precio = np.array([base_precio[p] for p in plataforma]) + rng.normal(0, 8, size=n)
    precio = np.clip(precio, 0.99, 89.99).round(2)

    # Horas de juego: depende del género (los RPG duran más)
    efecto_genero = {"Acción": 25, "Aventura": 30, "RPG": 70, "Deportes": 40,
                     "Estrategia": 55, "Shooter": 28, "Simulación": 60, "Puzzle": 15}
    horas = np.array([efecto_genero[g] for g in genero]) * rng.lognormal(0, 0.35, size=n)
    horas = np.clip(horas, 1, 300).round(1)

    jugadores_activos = (ventas * rng.uniform(0.05, 0.35, size=n) * 1_000_000).round(0)

    df = pd.DataFrame({
        "titulo": [f"Juego_{i:04d}" for i in range(1, n + 1)],
        "genero": genero,
        "plataforma": plataforma,
        "desarrollador": desarrollador,
        "clasificacion_edad": clasificacion,
        "multijugador": multijugador,
        "anio_lanzamiento": anio,
        "precio_usd": precio,
        "ventas_millones": ventas,
        "puntuacion_critica": critica,
        "puntuacion_usuarios": usuarios,
        "horas_promedio": horas,
        "jugadores_activos": jugadores_activos,
    })

    # Introducimos ~4% de valores nulos a propósito:
    # un dataset real casi nunca está limpio, y el EDA debe detectarlo.
    idx_nulos = rng.choice(df.index, size=int(n * 0.04), replace=False)
    df.loc[idx_nulos, "puntuacion_usuarios"] = np.nan

    return df


# =========================================================
# 2. SIDEBAR — CONTROLES INTERACTIVOS
# =========================================================
st.sidebar.header("⚙️ Configuración de datos")
n_registros = st.sidebar.slider("Número de videojuegos", 100, 2000, 600, step=100)
semilla = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)

df = generar_datos(n_registros, int(semilla))

st.sidebar.header("🔍 Filtros")
generos_sel = st.sidebar.multiselect(
    "Género", sorted(df["genero"].unique()), default=sorted(df["genero"].unique())
)
plataformas_sel = st.sidebar.multiselect(
    "Plataforma", sorted(df["plataforma"].unique()), default=sorted(df["plataforma"].unique())
)
rango_anios = st.sidebar.slider(
    "Año de lanzamiento",
    int(df["anio_lanzamiento"].min()), int(df["anio_lanzamiento"].max()),
    (int(df["anio_lanzamiento"].min()), int(df["anio_lanzamiento"].max()))
)
solo_multi = st.sidebar.checkbox("Solo juegos multijugador", value=False)

# Aplicación de los filtros al DataFrame
dff = df[
    df["genero"].isin(generos_sel)
    & df["plataforma"].isin(plataformas_sel)
    & df["anio_lanzamiento"].between(*rango_anios)
]
if solo_multi:
    dff = dff[dff["multijugador"]]

if dff.empty:
    st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros del panel izquierdo.")
    st.stop()

# Columnas por tipo (útiles para los selectores de las pestañas)
num_cols = dff.select_dtypes(include=np.number).columns.tolist()
cat_cols = ["genero", "plataforma", "desarrollador", "clasificacion_edad", "multijugador"]


# =========================================================
# 3. MÉTRICAS GENERALES
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Juegos analizados", f"{len(dff):,}")
c2.metric("Ventas totales", f"{dff['ventas_millones'].sum():,.1f} M")
c3.metric("Crítica promedio", f"{dff['puntuacion_critica'].mean():.1f} / 100")
c4.metric("Precio promedio", f"$ {dff['precio_usd'].mean():.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Datos", "🔢 Cuantitativo", "🔤 Cualitativo", "📈 Gráficos", "🎮 Explorador"]
)

# ---------------------------------------------------------
# TAB 1 — VISTA GENERAL DEL DATASET
# ---------------------------------------------------------
with tab1:
    st.subheader("Vista previa del dataset")
    st.dataframe(dff.head(50), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Estructura (tipos de dato)**")
        tipos = pd.DataFrame({
            "columna": dff.columns,
            "tipo": dff.dtypes.astype(str).values,
            "valores_únicos": [dff[c].nunique() for c in dff.columns],
        })
        st.dataframe(tipos, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Valores nulos por columna**")
        nulos = dff.isna().sum().reset_index()
        nulos.columns = ["columna", "nulos"]
        nulos["% nulos"] = (nulos["nulos"] / len(dff) * 100).round(2)
        st.dataframe(nulos, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2 — EDA CUANTITATIVO
# ---------------------------------------------------------
with tab2:
    st.subheader("Estadísticos descriptivos")
    st.dataframe(dff[num_cols].describe().T.round(2), use_container_width=True)

    var = st.selectbox("Variable cuantitativa a analizar", num_cols, index=num_cols.index("ventas_millones"))
    serie = dff[var].dropna()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Media", f"{serie.mean():.2f}")
    m2.metric("Mediana", f"{serie.median():.2f}")
    m3.metric("Desv. estándar", f"{serie.std():.2f}")
    m4.metric("Asimetría", f"{stats.skew(serie):.2f}")
    m5.metric("Curtosis", f"{stats.kurtosis(serie):.2f}")

    col1, col2 = st.columns(2)
    with col1:
        bins = st.slider("Número de intervalos (bins)", 5, 60, 30)
        fig_h = px.histogram(dff, x=var, nbins=bins, marginal="rug",
                             title=f"Distribución de {var}")
        st.plotly_chart(fig_h, use_container_width=True)
    with col2:
        fig_b = px.box(dff, y=var, points="outliers", title=f"Boxplot de {var} (detección de atípicos)")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Matriz de correlación (Pearson)")
    corr = dff[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", ax=ax, annot_kws={"size": 7})
    ax.tick_params(labelsize=7)
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3 — EDA CUALITATIVO
# ---------------------------------------------------------
with tab3:
    st.subheader("Análisis de variables categóricas")
    cat = st.selectbox("Variable cualitativa", cat_cols)

    frec = dff[cat].value_counts().reset_index()
    frec.columns = [cat, "frecuencia"]
    frec["frecuencia_relativa_%"] = (frec["frecuencia"] / len(dff) * 100).round(2)
    frec["frecuencia_acumulada"] = frec["frecuencia"].cumsum()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Tabla de frecuencias**")
        st.dataframe(frec, use_container_width=True, hide_index=True)
    with col2:
        tipo_g = st.radio("Tipo de gráfico", ["Barras", "Pastel"], horizontal=True)
        if tipo_g == "Barras":
            fig_c = px.bar(frec, x=cat, y="frecuencia", color=cat, title=f"Frecuencia por {cat}")
        else:
            fig_c = px.pie(frec, names=cat, values="frecuencia", hole=0.4, title=f"Composición por {cat}")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("### Tabla cruzada (dos variables cualitativas)")
    c1, c2 = st.columns(2)
    cat_x = c1.selectbox("Variable en filas", cat_cols, index=0)
    cat_y = c2.selectbox("Variable en columnas", cat_cols, index=1)
    tabla = pd.crosstab(dff[cat_x], dff[cat_y])
    st.dataframe(tabla, use_container_width=True)

# ---------------------------------------------------------
# TAB 4 — GRÁFICOS BIVARIADOS Y TEMPORALES
# ---------------------------------------------------------
with tab4:
    st.subheader("Relación entre variables")
    c1, c2, c3 = st.columns(3)
    eje_x = c1.selectbox("Eje X", num_cols, index=num_cols.index("puntuacion_critica"))
    eje_y = c2.selectbox("Eje Y", num_cols, index=num_cols.index("ventas_millones"))
    color = c3.selectbox("Color por", cat_cols, index=0)

    fig_s = px.scatter(dff, x=eje_x, y=eje_y, color=color, size="precio_usd",
                       hover_data=["titulo", "anio_lanzamiento"], opacity=0.7,
                       trendline="ols" if st.checkbox("Mostrar línea de tendencia") else None,
                       title=f"{eje_y} vs {eje_x}")
    st.plotly_chart(fig_s, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        num_box = st.selectbox("Variable numérica (boxplot por categoría)", num_cols,
                               index=num_cols.index("horas_promedio"))
        cat_box = st.selectbox("Agrupar por", cat_cols, index=0, key="cat_box")
        fig_bx = px.box(dff, x=cat_box, y=num_box, color=cat_box,
                        title=f"{num_box} según {cat_box}")
        st.plotly_chart(fig_bx, use_container_width=True)
    with col2:
        serie_tiempo = (dff.groupby("anio_lanzamiento")
                          .agg(ventas=("ventas_millones", "sum"),
                               critica=("puntuacion_critica", "mean"))
                          .reset_index())
        metrica = st.radio("Métrica temporal", ["ventas", "critica"], horizontal=True)
        fig_t = px.line(serie_tiempo, x="anio_lanzamiento", y=metrica, markers=True,
                        title=f"Evolución anual de {metrica}")
        st.plotly_chart(fig_t, use_container_width=True)

# ---------------------------------------------------------
# TAB 5 — EXPLORADOR INTERACTIVO
# ---------------------------------------------------------
with tab5:
    st.subheader("Explorador y ranking")

    c1, c2 = st.columns(2)
    criterio = c1.selectbox("Ordenar por", num_cols, index=num_cols.index("ventas_millones"))
    top_n = c2.slider("Top N", 5, 50, 10)
    orden = st.radio("Orden", ["Descendente", "Ascendente"], horizontal=True)

    ranking = dff.sort_values(criterio, ascending=(orden == "Ascendente")).head(top_n)
    st.dataframe(
        ranking[["titulo", "genero", "plataforma", "anio_lanzamiento", criterio]],
        use_container_width=True, hide_index=True
    )
    fig_r = px.bar(ranking, x="titulo", y=criterio, color="genero",
                   title=f"Top {top_n} por {criterio}")
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("### Ficha de un juego")
    juego = st.selectbox("Selecciona un título", dff["titulo"].tolist())
    ficha = dff[dff["titulo"] == juego].T
    ficha.columns = ["valor"]
    st.table(ficha)

    st.markdown("### Descargar datos filtrados")
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv,
                       file_name="videojuegos_filtrado.csv", mime="text/csv")

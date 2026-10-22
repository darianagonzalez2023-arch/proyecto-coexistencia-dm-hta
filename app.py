
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Coexistencia DM + HTA",
    page_icon="❤️",
    layout="centered"
)

# Rutas de los archivos
CARPETA = Path(__file__).parent
ruta_modelo = CARPETA / "modelo_coexistencia.pkl"
ruta_datos = CARPETA / "39._Enfermedades_crónicas_en_el_municipio_de_Bucaramanga_20260814 (1).csv"

# Cargar modelo y base de datos
modelo = joblib.load(ruta_modelo)
datos = pd.read_csv(ruta_datos, sep=";")

# Crear variable de coexistencia
datos["COEXISTENCIA_DM_HTA"] = (
    (datos["Diabetes"] == 1) &
    (datos["Hipertensión"] == 1)
).astype(int)

# Diseño visual: debe estar fuera del botón
st.markdown("""
<style>
.stApp {
    background-color: #E3F2FD;
}
h1 {
    color: #1B4965;
}
div.stButton > button {
    background-color: #1B4965 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: bold !important;
}
div.stButton > button:hover {
    background-color: #2876A5 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Título e instrucciones
st.title("🩺❤️ Evaluación de coexistencia DM + HTA")

st.write(
    "Seleccione el sexo e ingrese la edad para consultar "
    "los resultados observados y la estimación del modelo."
)

st.info(
    "Herramienta académica basada en los datos analizados "
    "del municipio de Bucaramanga."
)

# Datos de entrada
st.subheader("Datos de la persona")

sexo_mostrado = st.selectbox(
    "Sexo",
    ["♀️ FEMENINO", "♂️ MASCULINO"]
)

sexos = {
    "♀️ FEMENINO": "FEMENINO",
    "♂️ MASCULINO": "MASCULINO"
}
sexo = sexos[sexo_mostrado]

edad = st.number_input(
    "Edad (años)",
    min_value=0,
    max_value=120,
    value=30,
    step=1
)

# Identificar ciclo de vida
def obtener_ciclo_de_vida(edad):
    if edad <= 6:
        return "PRIMERA INFANCIA"
    elif edad <= 11:
        return "INFANCIA"
    elif edad <= 17:
        return "ADOLESCENCIA"
    elif edad <= 28:
        return "JOVENES"
    elif edad <= 59:
        return "ADULTEZ"
    else:
        return "PERSONA MAYOR"

ciclo = obtener_ciclo_de_vida(edad)
st.caption(f"Grupo etario identificado: {ciclo}")

# Consultar resultados
if st.button("🔍 Consultar resultados"):

    # Resultados observados en la base de datos
    datos_filtrados = datos[
        (datos["Sexo"] == sexo) &
        (datos["CICLO_DE_VIDA"] == ciclo)
    ]

    st.subheader("1. Resultados observados en la base de datos")

    if not datos_filtrados.empty:
        personas = len(datos_filtrados)
        casos = int(
            datos_filtrados["COEXISTENCIA_DM_HTA"].sum()
        )
        porcentaje_observado = casos / personas * 100

        col1, col2 = st.columns(2)
        col1.metric("Personas evaluadas", f"{personas:,}")
        col2.metric("Casos de coexistencia", f"{casos:,}")

        st.metric(
            "Porcentaje observado de coexistencia",
            f"{porcentaje_observado:.2f}%"
        )

        st.caption(
            "Cálculo: Casos observados / Personas evaluadas × 100."
        )
    else:
        st.warning(
            "No se encontraron registros para el sexo y ciclo de vida seleccionados."
        )

    # Predicción del modelo
    persona = pd.DataFrame({
        "CICLO_DE_VIDA": [ciclo],
        "Sexo": [sexo]
    })

    probabilidad = modelo.predict_proba(persona)[0, 1]
    porcentaje = probabilidad * 100

    # Clasificación
    if porcentaje >= 20:
        nivel = "ALTO"
        prioridad = "ALTA"
    elif porcentaje > 15:
        nivel = "MEDIO-ALTO"
        prioridad = "ALTA"
    elif porcentaje >= 10:
        nivel = "MEDIO"
        prioridad = "MEDIA"
    else:
        nivel = "BAJO"
        prioridad = "BAJA"

    # Mostrar predicción
    st.divider()
    st.subheader("2. Resultado estimado por el modelo")

    st.metric(
        "Probabilidad estimada de coexistencia",
        f"{porcentaje:.2f}%"
    )

    if nivel == "ALTO":
        st.error(f"🔴 Nivel de coexistencia: {nivel}")
    elif nivel == "MEDIO-ALTO":
        st.warning(f"🟠 Nivel de coexistencia: {nivel}")
    elif nivel == "MEDIO":
        st.warning(f"🟡 Nivel de coexistencia: {nivel}")
    else:
        st.success(f"🟢 Nivel de coexistencia: {nivel}")

    st.write(f"**Prioridad asignada:** {prioridad}")
    st.progress(float(probabilidad))

    st.caption(
        "El porcentaje observado muestra los registros encontrados en la base de datos. "
        "El porcentaje estimado procede del modelo; no se trata de un diagnóstico "
        "clínico ni una probabilidad individual validada clínicamente."
    )



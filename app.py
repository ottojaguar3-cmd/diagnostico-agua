"""
Sistema de Diagnóstico de Potabilidad de Agua
-----------------------------------------------
App Streamlit que utiliza un Pipeline de scikit-learn (IterativeImputer +
GradientBoostingClassifier) entrenado previamente para predecir si una
muestra de agua es potable, a partir de 14 parámetros fisicoquímicos,
metálicos y microbiológicos/orgánicos.

Coloca este archivo junto a "modelo_agua.pkl" y ejecuta:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ──────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN GENERAL DE LA PÁGINA
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diagnóstico de Potabilidad de Agua",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# 2. ESTILO VISUAL — Tema profesional, formal y elegante
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Source Sans 3', sans-serif;
        }

        :root {
            --agua-900: #0b2e4a;
            --agua-800: #103b5c;
            --agua-700: #14507d;
            --agua-600: #1a6ba3;
            --agua-500: #2489c9;
            --agua-100: #eaf4fa;
            --agua-050: #f4faff;
            --potable-color: #1c7a4c;
            --nopotable-color: #b3261e;
        }

        .stApp {
            background: linear-gradient(180deg, #f7fbfd 0%, #eef5f9 100%);
        }

        /* Encabezado principal */
        .app-header {
            padding: 2.1rem 2.4rem;
            background: linear-gradient(120deg, var(--agua-900) 0%, var(--agua-700) 60%, var(--agua-600) 100%);
            border-radius: 14px;
            color: #ffffff;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(11, 46, 74, 0.18);
        }
        .app-header h1 {
            font-family: 'Playfair Display', serif;
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: 0.2px;
        }
        .app-header p {
            font-size: 1.02rem;
            color: #dcecf6;
            margin: 0;
            font-weight: 400;
        }

        /* Tarjetas de sección */
        .section-card {
            background: #ffffff;
            border: 1px solid #e1ecf2;
            border-radius: 12px;
            padding: 1.4rem 1.6rem 0.6rem 1.6rem;
            margin-bottom: 1.3rem;
            box-shadow: 0 2px 10px rgba(16, 59, 92, 0.05);
        }
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--agua-900);
            margin-bottom: 0.15rem;
        }
        .section-sub {
            font-size: 0.88rem;
            color: #5c7386;
            margin-bottom: 0.9rem;
        }

        /* Botón principal */
        div.stButton > button {
            background: linear-gradient(120deg, var(--agua-700), var(--agua-600));
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.4rem;
            font-weight: 600;
            font-size: 1.02rem;
            letter-spacing: 0.2px;
            box-shadow: 0 4px 14px rgba(20, 80, 125, 0.28);
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(20, 80, 125, 0.35);
        }

        /* Tarjetas de resultado */
        .result-card {
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
            margin-top: 0.6rem;
            margin-bottom: 1.2rem;
        }
        .result-potable {
            background: #eaf7ef;
            border: 1.5px solid var(--potable-color);
        }
        .result-nopotable {
            background: #fbeceb;
            border: 1.5px solid var(--nopotable-color);
        }
        .result-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-potable .result-title { color: var(--potable-color); }
        .result-nopotable .result-title { color: var(--nopotable-color); }

        .footer-note {
            text-align: center;
            font-size: 0.8rem;
            color: #7c8e9c;
            margin-top: 2.2rem;
            padding-top: 1rem;
            border-top: 1px solid #e1ecf2;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--agua-900) 0%, var(--agua-800) 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #eaf4fa !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# 3. DEFINICIÓN DE PARÁMETROS DEL MODELO
#    Orden EXACTO usado en el entrenamiento (no modificar el orden).
# ──────────────────────────────────────────────────────────────────────────
# Cada parámetro: clave, etiqueta, unidad, min, max, valor por defecto, paso, decimales
PARAMETROS = {
    "fisicoquimicos": {
        "titulo": "Parámetros Fisicoquímicos",
        "subtitulo": "Características generales de la muestra de agua.",
        "campos": [
            dict(key="pH",  label="pH",                              unidad="",        min=0.0,  max=14.0,  default=7.0,  step=0.1, fmt="%.1f"),
            dict(key="CE",  label="Conductividad Eléctrica (CE)",     unidad="µS/cm",   min=0.0,  max=5000.0, default=500.0, step=10.0, fmt="%.1f"),
            dict(key="T",   label="Temperatura (T)",                 unidad="°C",      min=0.0,  max=40.0,  default=20.0, step=0.5, fmt="%.1f"),
            dict(key="OD",  label="Oxígeno Disuelto (OD)",           unidad="mg/L",    min=0.0,  max=20.0,  default=6.0,  step=0.1, fmt="%.1f"),
            dict(key="DBO", label="Demanda Bioquímica de Oxígeno (DBO)", unidad="mg/L", min=0.0,  max=100.0, default=3.0,  step=0.1, fmt="%.1f"),
            dict(key="Dureza", label="Dureza Total",                 unidad="mg/L CaCO₃", min=0.0, max=1000.0, default=100.0, step=1.0, fmt="%.1f"),
            dict(key="Ca",  label="Calcio (Ca)",                     unidad="mg/L",    min=0.0,  max=500.0, default=40.0, step=1.0, fmt="%.1f"),
            dict(key="Mg",  label="Magnesio (Mg)",                   unidad="mg/L",    min=0.0,  max=500.0, default=20.0, step=1.0, fmt="%.1f"),
        ],
    },
    "organicos_microbiologicos": {
        "titulo": "Parámetros Orgánicos y Microbiológicos",
        "subtitulo": "Indicadores de contaminación orgánica y biológica.",
        "campos": [
            dict(key="CT",  label="Coliformes Totales (CT)",         unidad="NMP/100mL", min=0.0, max=100000.0, default=500.0, step=10.0, fmt="%.1f"),
            dict(key="AyG", label="Aceites y Grasas (AyG)",          unidad="mg/L",    min=0.0,  max=50.0,  default=0.5,  step=0.1, fmt="%.2f"),
        ],
    },
    "metales_pesados": {
        "titulo": "Metales Pesados",
        "subtitulo": "Concentración total de metales traza.",
        "campos": [
            dict(key="ArT", label="Arsénico Total (ArT)",            unidad="mg/L",    min=0.0,  max=1.0,   default=0.01, step=0.001, fmt="%.3f"),
            dict(key="PbT", label="Plomo Total (PbT)",               unidad="mg/L",    min=0.0,  max=1.0,   default=0.01, step=0.001, fmt="%.3f"),
            dict(key="CuT", label="Cobre Total (CuT)",               unidad="mg/L",    min=0.0,  max=5.0,   default=0.1,  step=0.01, fmt="%.2f"),
            dict(key="MnT", label="Manganeso Total (MnT)",           unidad="mg/L",    min=0.0,  max=5.0,   default=0.1,  step=0.01, fmt="%.2f"),
        ],
    },
}

# Orden exacto que espera el modelo
ORDEN_MODELO = ["pH", "CE", "T", "OD", "DBO", "CT", "AyG", "ArT", "PbT", "CuT", "MnT", "Ca", "Mg", "Dureza"]

# ──────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO DESDE GOOGLE DRIVE (Blindado contra archivos corruptos)
# ──────────────────────────────────────────────────────────────────────────
import os

@st.cache_resource(show_spinner=False)
def cargar_pipeline():
    ruta_local = "modelo_agua.pkl"
    ID_DRIVE = "1hJ06f5McpF-T1PaKz0xbBMgeZ5BEUgNN"
    url = f"https://drive.google.com/uc?export=download&confirm=t&id={ID_DRIVE}"
    
    # Si el archivo no existe, o si existe pero pesa 0 bytes, se fuerza la descarga limpia
    if not os.path.exists(ruta_local) or os.path.getsize(ruta_local) == 0:
        with st.spinner("Descargando el modelo predictivo real... Esto toma unos segundos."):
            if os.path.exists(ruta_local):
                os.remove(ruta_local)
            
            # Forzamos la descarga usando curl (evita bloqueos 403)
            os.system(f'curl -L "{url}" -o {ruta_local}')
            
    # Intentamos cargar el pipeline de forma segura
    try:
        pipeline = joblib.load(ruta_local)
        return pipeline
    except Exception as e:
        # Si falla (por ejemplo, descarga incompleta), borramos el archivo para que reintente en la próxima carga
        if os.path.exists(ruta_local):
            os.remove(ruta_local)
        raise e

modelo_disponible = True
try:
    pipeline = cargar_pipeline()
    imputador = pipeline.named_steps.get("imputador")
    clasificador = pipeline.named_steps.get("clasificador")
except Exception as e:
    modelo_disponible = False
    pipeline, imputador, clasificador = None, None, None
    st.error(f"Error al inicializar el modelo predictivo: {e}. Por favor, recarga la página.")

# ──────────────────────────────────────────────────────────────────────────
# 5. BARRA LATERAL
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💧 Diagnóstico de Agua")
    st.markdown(
        "Herramienta de apoyo para la evaluación temprana de la "
        "potabilidad del agua, basada en un modelo de **Gradient "
        "Boosting** entrenado con parámetros fisicoquímicos, "
        "microbiológicos y de metales pesados."
    )
    st.markdown("---")
    st.markdown("#### Cómo usarla")
    st.markdown(
        "1. Completa los parámetros disponibles.\n"
        "2. Si desconoces algún valor, marca **\"Dato no disponible\"**; "
        "el sistema lo estimará automáticamente.\n"
        "3. Presiona **Realizar Diagnóstico**."
    )
    st.markdown("---")
    st.markdown("#### Estado del modelo")
    if modelo_disponible:
        st.success("Modelo cargado correctamente ✔", icon="✅")
    else:
        st.error("No se pudo cargar el modelo.", icon="⚠️")
    st.markdown("---")
    st.caption(
        "⚠️ Esta herramienta ofrece un diagnóstico orientativo y no "
        "sustituye un análisis de laboratorio certificado."
    )

# ──────────────────────────────────────────────────────────────────────────
# 6. ENCABEZADO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>💧 Sistema de Diagnóstico de Potabilidad de Agua</h1>
        <p>Evaluación técnica asistida por Machine Learning a partir de parámetros de calidad de agua.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not modelo_disponible:
    st.error(
        "No fue posible cargar el archivo **modelo_agua.pkl**. "
        "Verifica que se encuentre en la misma carpeta que este script."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
# 7. FORMULARIO DE ENTRADA DE PARÁMETROS
# ──────────────────────────────────────────────────────────────────────────
valores_usuario = {}
datos_faltantes = {}

for grupo in PARAMETROS.values():
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{grupo['titulo']}</div>
            <div class="section-sub">{grupo['subtitulo']}</div>
        """,
        unsafe_allow_html=True,
    )

    columnas = st.columns(2)
    for i, campo in enumerate(grupo["campos"]):
        col = columnas[i % 2]
        with col:
            sin_dato = st.checkbox(
                "Dato no disponible",
                key=f"chk_{campo['key']}",
                help="Si activas esta opción, el sistema estimará el valor automáticamente.",
            )
            etiqueta = f"{campo['label']}" + (f" ({campo['unidad']})" if campo["unidad"] else "")
            valor = st.number_input(
                etiqueta,
                min_value=campo["min"],
                max_value=campo["max"],
                value=campo["default"],
                step=campo["step"],
                format=campo["fmt"],
                key=f"num_{campo['key']}",
                disabled=sin_dato,
            )
            valores_usuario[campo["key"]] = np.nan if sin_dato else valor
            datos_faltantes[campo["key"]] = sin_dato

    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────
# 8. BOTÓN DE DIAGNÓSTICO Y LÓGICA DE PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────────
col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("🔎 Realizar Diagnóstico", use_container_width=True)

if ejecutar:
    # Construir vector en el orden exacto que espera el modelo
    vector = np.array([[valores_usuario[k] for k in ORDEN_MODELO]], dtype=float)
    hay_faltantes = any(datos_faltantes.values())

    with st.spinner("Analizando la muestra..."):
        try:
            if hay_faltantes:
                vector_completo = imputador.transform(vector)
            else:
                vector_completo = vector

            prediccion = clasificador.predict(vector_completo)[0]
            probabilidades = clasificador.predict_proba(vector_completo)[0]
            porcentaje_potable = probabilidades[1] * 100
            porcentaje_nopotable = probabilidades[0] * 100
        except Exception as e:
            st.error(f"Ocurrió un error al ejecutar el modelo: {e}")
            st.stop()

    st.markdown("## Resultado del Diagnóstico")

    if prediccion == 1:
        st.markdown(
            f"""
            <div class="result-card result-potable">
                <div class="result-title">✅ Agua clasificada como POTABLE</div>
                <p style="margin:0; color:#2b4a3b;">
                    La muestra analizada presenta parámetros compatibles con agua apta para consumo humano,
                    según el modelo predictivo.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-nopotable">
                <div class="result-title">⛔ Agua clasificada como NO POTABLE</div>
                <p style="margin:0; color:#5c2a27;">
                    La muestra analizada presenta parámetros que sugieren riesgo para el consumo humano.
                    Se recomienda un análisis de laboratorio confirmatorio y tratamiento adecuado.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Confianza — Potable", f"{porcentaje_potable:.1f}%")
        st.progress(min(max(porcentaje_potable / 100, 0.0), 1.0))
    with c2:
        st.metric("Confianza — No Potable", f"{porcentaje_nopotable:.1f}%")
        st.progress(min(max(porcentaje_nopotable / 100, 0.0), 1.0))

    # Tabla resumen de los datos utilizados
    with st.expander("📋 Ver detalle de los parámetros utilizados en el diagnóstico"):
        etiquetas_por_key = {
            campo["key"]: campo["label"]
            for grupo in PARAMETROS.values()
            for campo in grupo["campos"]
        }
        unidades_por_key = {
            campo["key"]: campo["unidad"]
            for grupo in PARAMETROS.values()
            for campo in grupo["campos"]
        }

        filas = []
        for idx, key in enumerate(ORDEN_MODELO):
            filas.append(
                {
                    "Parámetro": etiquetas_por_key[key],
                    "Unidad": unidades_por_key[key] or "—",
                    "Valor ingresado": "No disponible" if datos_faltantes[key] else f"{valores_usuario[key]:.3f}",
                    "Valor usado (imputado)" if datos_faltantes[key] else "Valor usado": f"{vector_completo[0][idx]:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        if hay_faltantes:
            st.caption(
                "Los valores marcados como *No disponible* fueron estimados automáticamente "
                "mediante el imputador iterativo del modelo, a partir de los demás parámetros ingresados."
            )

st.markdown(
    """
    <div class="footer-note">
        Sistema de apoyo al diagnóstico de potabilidad de agua · Modelo: Gradient Boosting Classifier<br>
        Uso referencial — no reemplaza un análisis de laboratorio certificado.
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt

emojis = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

# 2. Cargar modelo
@st.cache_resource
def cargar_modelo():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# 3. Inicializar historial 
def inicializar_historial():
    if "historial" not in st.session_state:
        st.session_state.historial = pd.DataFrame(columns=["Texto", "Emoción", "Confianza"])

# 4. Clasificar texto e interactuar
def mostrar_interfaz(modelo):
    st.title("Clasificador de Emociones en Texto")
    st.markdown("Escribe una frase **(En ingles)** y detectaremos la emoción que transmite usando IA")

    texto = st.text_input("Escribe tu texto aquí:")

    if texto:
        resultado = modelo(texto)[0]
        emocion = resultado["label"]
        confianza = round(resultado["score"] * 100, 2)
        emoji = emojis.get(emocion, "🤖")

        st.success(f"**Emoción detectada:** {emocion.upper()} {emoji}")
        st.info(f"**Confianza:** {confianza}%")

        nuevo = pd.DataFrame([{
            "Texto": texto,
            "Emoción": emocion,
            "Confianza": confianza
        }])
        st.session_state.historial = pd.concat([st.session_state.historial, nuevo], ignore_index=True)

# 5. Mostrar historial y gráfico
def mostrar_historial_y_grafica():
    if not st.session_state.historial.empty:
        st.markdown("### Historial de Clasificaciones")
        st.dataframe(st.session_state.historial)

        st.markdown("### Gráfico de Emociones Detectadas")
        conteo = st.session_state.historial["Emoción"].value_counts()

        fig, ax = plt.subplots()
        conteo.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_title("Frecuencia de emociones")
        ax.set_xlabel("Emoción")
        ax.set_ylabel("Cantidad")
        st.pyplot(fig)

# 6. Main
def main():
    modelo = cargar_modelo()
    inicializar_historial()
    mostrar_interfaz(modelo)
    mostrar_historial_y_grafica()

if __name__ == "__main__":
    main()


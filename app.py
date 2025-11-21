from io import BytesIO

import streamlit as st

import models.votingClassifier as vc
import utils.helpersPlot as hp


def button_download(fig):
    buffer = BytesIO()
    fig.saving(buffer, format= "png")
    return buffer.seek(0)


st.title("Compilado de informações pro artigo")

st.subheader("Correlação de Pearson")

fig = hp.exibe_correlacao(vc.result_corr)
st.pyplot(fig)
st.download_button(
    label= "📥 Baixar gráfico",
    data= button_download(fig),
    file_name= "Correlação_Pearson.png",
    mime= "image/png"
)

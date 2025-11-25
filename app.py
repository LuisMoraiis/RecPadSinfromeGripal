from io import BytesIO

import streamlit as st

import models.votingClassifier as vc
import utils.helpersPlot as hp


def button_download(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format= "png")
    buffer.seek(0)
    return buffer


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

st.subheader("Ambiguidade dos modelos em relação ao ensemble")
st.write("A ambiguidade do modelo em relação ao ensemble mede o quanto aquele modelo discorda do ensemble.")
st.latex(r"""
\text{ambi}(h_i) = \frac{1}{N} \sum_{j=1}^{N} \left( h_i(x_j) - H(x_j) \right)^2
""")

fig2 = hp.exibe_ambiguitys(vc.dic_ambiguitys)
st.pyplot(fig2)
st.download_button(
    label= "📥 Baixar gráfico",
    data= button_download(fig2),
    file_name= "Ambiguidade.png",
    mime= "image/png"
)

st.subheader("Bias-Variance-Covariance Decomposition")


fig3 = hp.exibi_bias_variance_covariance_decomposition(vc.decomp)
st.pyplot(fig3)

st.download_button(
    label= "📥 Baixar gráfico",
    data= button_download(fig3),
    file_name= "Ambiguidade.png",
    mime= "image/png"
)

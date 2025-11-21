import streamlit as st
import utils.helpersPlot as hp

import models.votingClassifier as vc

st.title("Compilado de informações pro artigo")

st.subheader("Correlação de Pearson")

st.pyplot(hp.exibe_correlacao(vc.result_corr))

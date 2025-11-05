import streamlit as st
import analysis.pre_processamento as preP

st.title("🔍 Análise Exploratória")

st.dataframe(preP.df.head(200))

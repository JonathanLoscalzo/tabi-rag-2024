import pandas as pd
import streamlit as st
from script import TabiPipeline


st.set_page_config(page_title="🦜🔗 Preguntas al libro de TABI", layout="wide")
st.title("🦜🔗 Preguntas al libro de TABI")
model = st.selectbox(
    "Modelo",
    (
        "llama3:8b",
        "llama3.1:8b",
        "mistral:7b",
        "phi3:latest",
        "gemma:2b",
        "qwen2:0.5b",
    ),
)
tp = TabiPipeline(llm_model=model)

result = None
with st.form("myform", clear_on_submit=False):
    query_text = st.text_input(
        "Ingrese la pregunta:",
    )
    submitted = st.form_submit_button(
        "Submit",
    )
    if submitted:
        with st.spinner("Calculating..."):
            result = tp.rag_with_ensemble(query_text).get("response")


if result:
    st.info(result)

st.table(pd.read_parquet("assets/preguntas_resueltas.parquet"))

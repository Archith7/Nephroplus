import os
import json
import re
import pandas as pd
import easyocr
import logging
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
import requests
from uuid import uuid4
from io import BytesIO
from PyPDF2 import PdfReader
from PIL import Image
import numpy as np
import cv2
from paddleocr import PaddleOCR
# --------------- CONFIGURATION ---------------
GEMINI_API_KEY = "AIzaSyAAPOvYkR2jVEj6s7cYZlCbPhzD7tdTZrk"
DB_FILE = "C:/Users/archi/Desktop/hackday/outputsNew/all_results_new.json"

# --------------- LOGGER SETUP ---------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("App")

# --------------- GEMINI CONFIG ---------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# --------------- FUNCTIONS ---------------
# def extract_text_from_image(uploaded_file):
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     reader = easyocr.Reader(['en'], gpu=False)
#     result = reader.readtext(image_bgr, detail=0, paragraph=True)
#     return "\n".join(result)
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    result = ocr_engine.ocr(image_np, cls=True)

    # Flatten results
    extracted_text = ""
    for line in result:
        for word_info in line:
            extracted_text += word_info[1][0] + " "
        extracted_text += "\n"
    
    return extracted_text.strip()

def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def build_structured_prompt(text):
    return f"""
You are a medical report extractor. From the following report text, extract structured JSON like:
{{
  "Patient Name": "",
  "Age": ,
  "Gender": "",
  "Tests": [
    {{
      "Test Name": "",
      "Result": "",
      "Unit": "",
      "Reference Range": ""
    }}
  ]
}}
If values are missing, use null. TEXT: <<< {text} >>>
"""

def extract_structured_data(text):
    prompt = build_structured_prompt(text)
    try:
        response = model.generate_content(prompt).text
        match = re.search(r'\{[\s\S]+\}', response)
        return json.loads(match.group()) if match else {}
    except Exception as e:
        logger.warning("Failed to parse model output: %s", e)
        return {}

def load_db():
    with open(DB_FILE) as f:
        return json.load(f)

def run_nlp_query_on_db(natural_query, db):
    query_prompt = f"""
Given this JSON lab data:
{json.dumps(db, indent=2)}

Answer this user question by analyzing the data:
"{natural_query}"

Return only the final answer as a sentence or table (no JSON).
"""
    try:
        response = model.generate_content(query_prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌  failed: {e}"

# --------------- STREAMLIT UI ---------------
st.set_page_config(page_title="🧬 Lab Query Assistant", layout="wide")
st.title("🔍 NLP Lab Report Query Assistant")

# Load DB on startup
db = load_db()

# Upload logic (optional if user uploads new reports)
uploaded = st.file_uploader("Upload Image or PDF (optional)", type=["png", "jpg", "jpeg", "pdf"])
if uploaded:
    if uploaded.type == "application/pdf":
        text = extract_text_from_pdf(uploaded)
    elif uploaded.type in ["image/png", "image/jpeg"]:
        text = extract_text_from_image(uploaded)
    else:
        text = ""
    
    if text:
        data = extract_structured_data(text)
        st.subheader("Extracted Data")
        st.json(data)
        if st.button("Add to DB (in-memory only)"):
            db.append(data)
            st.success("Added to current session DB.")

# --------------- QUERY INTERFACE ---------------
st.divider()
st.subheader("💬 Ask in Natural Language")

user_query = st.text_input("Enter a natural question (e.g. 'Show Hemoglobin for females')")

if user_query:
    with st.spinner("Querying..."):
        result = run_nlp_query_on_db(user_query, db)
    st.markdown("### 📋 Response")
    st.markdown(result)
# --------------- DATABASE VIEWER ---------------
with st.expander("📂 Show Raw Patient Database", expanded=False):
    try:
        db_raw = load_db()
        for entry in db_raw:
            with st.container(border=True):
                st.markdown(f"### 🧾 {entry.get('Patient Name', 'N/A')}")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Age:** {entry.get('Age', 'N/A')}")
                col2.markdown(f"**Gender:** {entry.get('Gender', 'N/A')}")
                tests = entry.get("Tests", [])
                if tests:
                    df_tests = pd.DataFrame(tests)
                    st.dataframe(df_tests, use_container_width=True, height=min(35 * len(tests), 400))
                else:
                    st.info("No tests found for this patient.")
    except Exception as e:
        st.warning("Could not load database.")
        st.error(e)

# --------------- Sidebar Info ---------------
st.sidebar.title("ℹ️ Info")
st.sidebar.write("Database File Used:")
st.sidebar.code(DB_FILE)
st.sidebar.write(" ")

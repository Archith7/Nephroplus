import os
import easyocr
import json
import pandas as pd
import re
import logging
import google.generativeai as genai
from datetime import datetime
import requests
from paddleocr import PaddleOCR

# 🔑 Set API key (can also load from .env or config file)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCGOjfbAUhLNph07Otueu6G0srKgRccq98"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("LabReportExtractor") 

# 🧠 Build Prompt for Gemini
def build_prompt(raw_text):
    return f"""
You are a medical data extractor. Extract structured information from lab report text.

Return the following JSON format:

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
if relevant values for the attributes not found , then leave it NULL
Text to extract from:
<<<
{raw_text}
>>>

Only return valid JSON. No extra commentary.
"""

def get_structured_data_gemini(raw_text):
    prompt = build_prompt(raw_text)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    # print("🧠 Gemini raw output:\n", response.text)
    print("🧠 Gemini raw output Generated\n")

    try:
        json_match = re.search(r'\{[\s\S]+\}', response.text)
        if json_match:
            json_data = json.loads(json_match.group(0))
            return json_data
        else:
            raise ValueError("No JSON found in response.")
    except Exception as e:
        print("⚠️ Failed to parse Gemini output:", e)
        return {}
    

def get_structured_data_ollama(raw_text):
    prompt = build_prompt(raw_text)

    try:
        logger.info("📡 Sending request to local Ollama model: %s", OLLAMA_MODEL)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        text_response = response.json().get("response", "")
        print("🧠 Ollama raw output Generated\n")
    except Exception as e:
        logger.error("⚠️ Ollama API error: %s", e)
        return {}

    try:
        json_match = re.search(r'\{[\s\S]+\}', text_response)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in Ollama response.")
    except Exception as e:
        logger.error("⚠️ Failed to parse Ollama output: %s", e)
        return {}

# 📖 OCR: Use EasyOCR to extract text
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

# 💾 Save to JSON & CSV
def save_outputs(all_data, json_path, csv_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)

    # Flatten for CSV
    all_rows = []
    for entry in all_data:
        if not entry.get("Tests"):
            continue
        for test in entry["Tests"]:
            row = {
                "Image Filename": entry.get("Image Filename"),
                "Patient Name": entry.get("Patient Name"),
                "Age": entry.get("Age"),
                "Gender": entry.get("Gender"),
                **test
            }
            all_rows.append(row)

    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

# 🔁 Full pipeline
def main():
    image_folder = "C:/Users/archi/Downloads/testdata/test1"
    output_json = "outputsNew/all_results_new.json"
    output_csv = "outputsNew/all_results_new.csv"
    all_results = []
    total = 0
    success = 0

    logger.info("🔍 Starting batch extraction from folder: %s", image_folder)

    # for file in sorted(os.listdir(image_folder))[19:20]:

    #     if not file.lower().endswith((".png", ".jpg", ".jpeg")):
    #         continue

    #     total += 1
    #     image_path = os.path.join(image_folder, file)
    #     logger.info(f"[{total}] Processing file: {file}")

    #     raw_text = extract_text_from_image(image_path)

    #     print('🧿🧿🧿',raw_text)
    #     if not raw_text.strip():
    #         logger.warning("No text extracted from %s", file)
    #         continue

    #     # structured = get_structured_data(raw_text)
    #     structured = get_structured_data_gemini(raw_text)  # Default: Gemini
    #     # structured = get_structured_data_ollama(raw_text)    # Uncomment for demo

    #     if structured and "Tests" in structured and structured["Tests"]:
    #         structured["Image Filename"] = file 
    #         all_results.append(structured)
    #         success += 1
    #         logger.info(f"✅ Success: Extracted data from {file}")
    #     else:
    #         logger.warning(f"❌ Skipped {file}: Extraction returned empty or incomplete data.")

    # if all_results:
    #     save_outputs(all_results, output_json, output_csv)
    #     logger.info("🎉 Completed. %d/%d files processed successfully.", success, total)
    # else:
    #     logger.error("❌ No successful extractions. Check logs for details.")
    # def main():
    image_folder = "C:/Users/archi/Downloads/testdata/test1"
    image_file = "19.png"  # 🔁 Replace with your actual file name
    image_path = os.path.join(image_folder, image_file)

    output_json = "outputsNew/all_results_new.json"
    output_csv = "outputsNew/all_results_new.csv"
    all_results = []
    total = 0
    success = 0

    logger.info("🔍 Starting extraction for a single image: %s", image_file)

    if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        logger.warning("⚠️ File is not a supported image: %s", image_file)
        return

    total += 1
    logger.info(f"[{total}] Processing file: {image_file}")

    raw_text = extract_text_from_image(image_path)
    print('🧿🧿🧿 Extracted Text:\n', raw_text)

    if not raw_text.strip():
        logger.warning("🚫 No text extracted from %s", image_file)
        return

    # Choose one of the following:
    structured = get_structured_data_gemini(raw_text)
    # structured = get_structured_data_ollama(raw_text)

    if structured and "Tests" in structured and structured["Tests"]:
        structured["Image Filename"] = image_file
        all_results.append(structured)
        success += 1
        logger.info(f"✅ Success: Extracted data from {image_file}")
    else:
        logger.warning(f"❌ Skipped {image_file}: Extraction returned empty or incomplete data.")

    if all_results:
        save_outputs(all_results, output_json, output_csv)
        logger.info("🎉 Completed. %d/%d files processed successfully.", success, total)
    else:
        logger.error("❌ No successful extractions. Check logs for details.")




if __name__ == "__main__":
    main()

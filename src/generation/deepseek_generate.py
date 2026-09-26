#!/usr/bin/env python3
import os
import json
import re
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import requests
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
MODEL_NAME = "deepseek-chat"
# If LIMIT is None, process the entire CSV
LIMIT = None
DATASET = "deptweet"  # irony or deptweet
LANG = "greek"  # bengali or greek
RETRIES = 3
WAIT_SECONDS = 2

#print dataset, lang, modelname
print(f"Dataset: {DATASET}, Language: {LANG}, Model Name: {MODEL_NAME}")

if not api_key:
    print("DEEPSEEK_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

API_URL = "https://api.deepseek.com/v1/chat/completions"

datasets_dir = os.path.join(PROJECT_ROOT, "data")
output_dir = os.path.join(PROJECT_ROOT, "outputs", "translations")
prompts_path = os.path.join(PROJECT_ROOT, "prompts", "prompts.yaml")

if DATASET == "irony":
    default_input = os.path.join(datasets_dir, "irony_main.csv")
    text_col = "tweet_text"
    dataset_name = "irony"
else:
    default_input = os.path.join(datasets_dir, "deptweet_main.csv")
    text_col = "tweet"
    dataset_name = "deptweet"

default_output = os.path.join(output_dir, f"{dataset_name}_{LANG}_deepseek_sample_translations.csv")

input_path = os.getenv("INPUT_PATH") or default_input
output_path = os.getenv("OUTPUT_PATH") or default_output

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)

template = prompts["languages"][LANG]["dual_translation"]

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)
if text_col not in df.columns:
    print(f"Expected column '{text_col}' in the input dataset.", file=sys.stderr)
    sys.exit(1)

rows = []
subset = df if LIMIT is None else df.head(max(0, LIMIT))
for _, row in tqdm(subset.iterrows(), total=len(subset)):
    text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    if not text:
        rows.append({"original_text": "", "literal_translation": None, "cultural_translation": None})
        continue

    prompt = template.replace("{text}", text)
    success = False
    last_err = None
    for attempt in range(RETRIES):
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            s = content.strip()
            try:
                data = json.loads(s)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", s)
                data = json.loads(m.group(0)) if m else {"literal": None, "cultural": None}
            rows.append({
                "original_text": text,
                "literal_translation": data.get("literal"),
                "cultural_translation": data.get("cultural"),
            })
            success = True
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(WAIT_SECONDS)
    if not success:
        rows.append({"original_text": text, "literal_translation": None, "cultural_translation": None, "error": last_err})

pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Wrote translations to: {output_path}")

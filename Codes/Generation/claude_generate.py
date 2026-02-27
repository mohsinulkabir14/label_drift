import os
import json
import re
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(project_root, "CLSLABEL", ".env"))

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
MODEL_NAME = "claude-3-5-haiku-20241022"
# If LIMIT is None, process the entire CSV
LIMIT = None
DATASET = "irony"  # irony or deptweet
LANG = "greek"  # bengali or greek
RETRIES = 3
WAIT_SECONDS = 2

if not api_key:
    print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

datasets_dir = os.path.join(project_root, "CLSLABEL", "Datasets")
output_dir = os.path.join(project_root, "CLSLABEL", "Output")
prompts_path = os.path.join(project_root, "CLSLABEL", "Codes", "prompts.yaml")

if DATASET == "irony":
    default_input = os.path.join(datasets_dir, "irony_main.csv")
    text_col = "tweet_text"
    dataset_name = "irony"
else:
    default_input = os.path.join(datasets_dir, "deptweet_main.csv")
    text_col = "tweet"
    dataset_name = "deptweet"

if LIMIT == None:
    default_output = os.path.join(output_dir, f"{dataset_name}_{LANG}_claude_full_translations.csv")
else:
    default_output = os.path.join(output_dir, f"{dataset_name}_{LANG}_claude_sample_translations.csv")

input_path = os.getenv("INPUT_PATH") or default_input
output_path = os.getenv("OUTPUT_PATH") or default_output

with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)
template = prompts["languages"][LANG]["dual_translation"]

client = anthropic.Anthropic(api_key=api_key)
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
            resp = client.messages.create(model=MODEL_NAME, temperature=0.5, max_tokens=1000, messages=[{"role": "user", "content": prompt}])
            content = resp.content[0].text if resp and resp.content else "{}"
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

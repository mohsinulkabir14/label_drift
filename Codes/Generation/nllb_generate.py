import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import json
import re
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import requests
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



def translate(text, src_lang, tgt_lang, model_name="facebook/nllb-200-1.3B"):
    """
    Translate text using the NLLB model.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Language codes for NLLB
    # src_lang = "eng_Latn"   # English (Latin script)
    # tgt_lang = "ell_Grek"   # Greek
    # tgt_lang = "ben_Beng"   # Bengali

    # Set source language before tokenization
    tokenizer.src_lang = src_lang

    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Get target language ID
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=500
    )

    # Decode to text
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(project_root, ".env"))


# MODEL_NAME = "facebook/nllb-200-1.3B"
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
# If LIMIT is None, process the entire CSV
LIMIT = None
DATASET = "deptweet"  # irony or deptweet
LANG = "greek"  # bengali or greek

SRC_LANG = "eng_Latn"
TGT_LANG_MAP = {
    "greek": "ell_Grek",
    "bengali": "ben_Beng",
}

TGT_LANG = TGT_LANG_MAP[LANG]


#print dataset, lang, modelname
print(f"Dataset: {DATASET}, Language: {LANG}, Model Name: {MODEL_NAME}")


datasets_dir = os.path.join(project_root, "Datasets")
output_dir = os.path.join(project_root, "Output")


if DATASET == "irony":
    default_input = os.path.join(datasets_dir, "irony_main.csv")
    text_col = "tweet_text"
    dataset_name = "irony"
else:
    default_input = os.path.join(datasets_dir, "deptweet_main.csv")
    text_col = "tweet"
    dataset_name = "deptweet"

default_output = os.path.join(output_dir, f"{dataset_name}_{LANG}_nllb_full_translations.csv")


input_path = os.getenv("INPUT_PATH") or default_input
output_path = os.getenv("OUTPUT_PATH") or default_output




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
    literal_translation = translate(text, SRC_LANG, TGT_LANG, model_name=MODEL_NAME)
    rows.append({"original_text": text, "literal_translation": literal_translation, "cultural_translation": None})


pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Wrote translations to: {output_path}")

#!/usr/bin/env python3
import os, sys, json, time, re, csv
from collections import Counter
from typing import Optional
import pandas as pd, yaml
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic
from openai import OpenAI
import requests

# variables
RETRIES = 10
WAIT_SECONDS = 2
LANG = "bengali"  # Greek or Bengali
FILE_NAME = "irony_bengali_nllb_full_translations.csv"
LIMIT = None  # Integer or None
TASK = "irony"  # "deptweet" or "irony"

if LANG.lower() not in FILE_NAME:
    print(f"Warning: LANG ({LANG}) does not match FILE_NAME ({FILE_NAME}). Please check the configuration.")
    sys.exit(1)
if TASK.lower() not in FILE_NAME:
    print(f"Warning: TASK ({TASK}) does not match FILE_NAME ({FILE_NAME}). Please check the configuration.")
    sys.exit(1)

print(f"Starting annotation for {FILE_NAME}...")



ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT, "Output")
ANNOTATION_DIR = os.path.join(OUTPUT_DIR, "Annotation")
PROMPTS_PATH = os.path.join(ROOT, "Codes", "prompts.yaml")
load_dotenv(os.path.join(ROOT, ".env"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
GPT_MODEL = "gpt-4.1-mini"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)
EVAL_TEMPLATE = PROMPTS["evaluation_literal"][TASK]

LOG_DIR = os.path.join(ANNOTATION_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_START = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
LOG_FILE = os.path.join(LOG_DIR, f"annotation_log_literal_{RUN_START}.csv")


def log_attempt(file_name: str, sample_idx: int, sample_name: str, model_name: str, try_idx: int, raw_output: str) -> None:
    header = ["timestamp", "file_name", "sample_idx", "sample_name", "model_name", "try_idx", "raw_output"]
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), file_name, sample_idx, sample_name, model_name, try_idx, raw_output])


def build_prompt_literal(literal: str) -> str:
    return (
        EVAL_TEMPLATE
        .replace("{LANG_NAME}", LANG)
        .replace("{literal}", literal)
    )


def parse_literal(s: str) -> Optional[str]:
    s = s.strip()
    m = re.search(r"(?i)label\s*:\s*\[?\s*([0-3])\s*\]?", s)
    return m.group(1).strip() if m else None


def judge_claude_literal(literal: str, file_name: str, sample_idx: int, sample_name: str) -> Optional[str]:
    if not ANTHROPIC_API_KEY:
        return None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    p = build_prompt_literal(literal)
    for attempt in range(RETRIES):
        try:
            r = client.messages.create(model=CLAUDE_MODEL, temperature=0, max_tokens=200, messages=[{"role": "user", "content": p}])
            s = r.content[0].text if r and r.content else ""
            log_attempt(file_name, sample_idx, sample_name, f"claude:{CLAUDE_MODEL}", attempt, s)
            return parse_literal(s)
        except Exception as e:
            err = str(e)
            base_delay = globals().get("WAIT_SECONDS", 2)
            delay = base_delay * (attempt + 1)
            if "Overloaded" in err or "529" in err:
                delay = max(delay, 10)
            time.sleep(delay)
            log_attempt(file_name, sample_idx, sample_name, f"claude:{CLAUDE_MODEL}", attempt, f"ERROR: {err}")
    return None


def judge_gpt_literal(literal: str, file_name: str, sample_idx: int, sample_name: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    client = OpenAI(api_key=OPENAI_API_KEY)
    p = build_prompt_literal(literal)
    for attempt in range(RETRIES):
        try:
            # r = client.responses.create(model=GPT_MODEL, input=p)
            # s = r.output_text.strip() if hasattr(r, "output_text") else json.dumps(r.dict())
            r = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": p}])
            s = r.choices[0].message.content.strip()
            log_attempt(file_name, sample_idx, sample_name, f"gpt:{GPT_MODEL}", attempt, s)
            return parse_literal(s)
        except Exception as e:
            err = str(e)
            base_delay = globals().get("WAIT_SECONDS", 2)
            delay = base_delay * (attempt + 1)
            time.sleep(delay)
            log_attempt(file_name, sample_idx, sample_name, f"gpt:{GPT_MODEL}", attempt, f"ERROR: {err}")
    return None


def judge_deepseek_literal(literal: str, file_name: str, sample_idx: int, sample_name: str) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    p = build_prompt_literal(literal)
    for attempt in range(RETRIES):
        try:
            payload = {"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0}
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status(); s = r.json()["choices"][0]["message"]["content"].strip()
            log_attempt(file_name, sample_idx, sample_name, f"deepseek:{DEEPSEEK_MODEL}", attempt, s)
            return parse_literal(s)
        except Exception as e:
            err = str(e)
            base_delay = globals().get("WAIT_SECONDS", 2)
            delay = base_delay * (attempt + 1)
            time.sleep(delay)
            log_attempt(file_name, sample_idx, sample_name, f"deepseek:{DEEPSEEK_MODEL}", attempt, f"ERROR: {err}")
    return None


def annotate_literal_file(file_name: str, limit: int = 2) -> str:
    in_path = os.path.join(OUTPUT_DIR, file_name)
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    df = pd.read_csv(in_path)
    cols = list(df.columns)
    # choose literal translation column strictly by language name, with fallback if missing
    preferred = None
    if LANG.lower() == "bengali":
        preferred = "gtrans_bengali"
    elif LANG.lower() == "greek":
        preferred = "gtrans_greek"
    if preferred and preferred in cols:
        lit_col = preferred
    elif "literal_translation" in cols:
        lit_col = "literal_translation"
    else:
        cand = [c for c in cols if "gtrans" in c.lower()]
        lit_col = cand[0] if cand else (cols[1] if len(cols) > 1 else cols[0])

    rows = []
    if limit is None:
        subset = df
        total = len(df)
    else:
        subset = df.head(max(0, int(limit)))
        total = min(int(limit), len(df))

    for idx, (_, row) in enumerate(tqdm(subset.iterrows(), total=total)):
        # choose original text column (dataset-specific)
        if TASK == "deptweet" and "tweet" in cols:
            orig_col = "tweet"
        elif TASK == "irony" and "tweet_text" in cols:
            orig_col = "tweet_text"
        elif 'original_text' in cols:
            orig_col = 'original_text'
        else:
            orig_col = None

        original = str(row[orig_col]) if orig_col and pd.notna(row[orig_col]) else ""
        literal = str(row[lit_col]).strip() if lit_col in row and pd.notna(row[lit_col]) else ""

        if literal == "":
            g_lit = d_lit = c_lit = ""
            lit_final = ""
            ds_lit_field = ""
        else:
            g_lit = judge_gpt_literal(literal, file_name, idx, original)
            d_lit = judge_deepseek_literal(literal, file_name, idx, original)
            # Initial voting between GPT and DeepSeek
            if g_lit and d_lit and g_lit == d_lit:
                c_lit = "-1"  # skipped
                ds_lit_field = d_lit
                lit_final = g_lit
            else:
                c_lit = judge_claude_literal(literal, file_name, idx, original)
                votes = [x for x in [g_lit, d_lit, c_lit] if x]
                if votes and len(set(votes)) == 1:
                    lit_final = votes[0]
                else:
                    cnt = Counter(votes)
                    lit_final = cnt.most_common(1)[0][0] if cnt else (g_lit or d_lit or c_lit or "")
                ds_lit_field = d_lit if d_lit is not None else ""

        rows.append({
            "original_text": original,
            "literal_translation": literal,
            "gpt_label_literal": g_lit,
            "deepseek_label_literal": ds_lit_field,
            "claude_label_literal": c_lit,
            "literal_machine_label": lit_final,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    is_full = (limit is None)
    base_name = file_name
    if not is_full:
        if "_sample" not in base_name:
            if base_name.endswith(".csv"):
                base_name = base_name.replace(".csv", "_sample.csv")
            else:
                root, ext = os.path.splitext(base_name)
                base_name = f"{root}_sample{ext}"
    else:
        base_name = base_name.replace("_sample.csv", ".csv")
    out_path = os.path.join(ANNOTATION_DIR, f"Annotation_{base_name}")
    out_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = annotate_literal_file(FILE_NAME, LIMIT)
    print(f"Wrote: {out}")

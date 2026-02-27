import os, sys, json, time, re
from collections import Counter
from typing import Optional, Tuple
import pandas as pd, yaml
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic
from openai import OpenAI
import requests
import csv

# variables
RETRIES = 10
WAIT_SECONDS = 2
LANG = "Bengali" # Greek or Bengali
FILE_NAME = "irony_bengali_deepseek_full_translations.csv"
LIMIT = None # Integer or None

if LANG.lower() not in FILE_NAME:
    print(f"Warning: LANG ({LANG}) does not match FILE_NAME ({FILE_NAME}). Please check the configuration.")
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
EVAL_TEMPLATE = PROMPTS["evaluation"]["irony"]

LOG_DIR = os.path.join(ANNOTATION_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)
RUN_START = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
LOG_FILE = os.path.join(LOG_DIR, f"annotation_log_{RUN_START}.csv")

def log_attempt(file_name: str, sample_idx: int, sample_name: str, model_name: str, try_idx: int, raw_output: str) -> None:
    header = ["timestamp", "file_name", "sample_idx", "sample_name", "model_name", "try_idx", "raw_output"]
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), file_name, sample_idx, sample_name, model_name, try_idx, raw_output])


def build_prompt_pair(literal: str, cultural: str) -> str:
    return (
        EVAL_TEMPLATE
        .replace("{LANG_NAME}", LANG)
        .replace("{literal}", literal)
        .replace("{cultural}", cultural)
    )


def parse_pair(s: str) -> Tuple[Optional[str], Optional[str]]:
    s = s.strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lit, cul = None, None
    for ln in lines:
        low = ln.lower()
        if low.startswith("literal translation label:"):
            lit = ln.split(":", 1)[-1].strip().strip("[] ")
        elif low.startswith("cultural translation label:"):
            cul = ln.split(":", 1)[-1].strip().strip("[] ")
    if lit or cul:
        return lit, cul
    m = re.search(r"literal translation label:\s*\[(.*?)\].*?cultural translation label:\s*\[(.*?)\]", s, re.I | re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def judge_deepseek_pair(literal: str, cultural: str, file_name: str, sample_idx: int, sample_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not ANTHROPIC_API_KEY:
        return None, None
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    p = build_prompt_pair(literal, cultural)
    err = None
    for attempt in range(RETRIES):
        try:
            r = client.messages.create(model=CLAUDE_MODEL, temperature=0, max_tokens=200, messages=[{"role": "user", "content": p}])
            s = r.content[0].text if r and r.content else ""
            log_attempt(file_name, sample_idx, sample_name, f"claude:{CLAUDE_MODEL}", attempt, s)
            return parse_pair(s)
        except Exception as e:
            err = str(e); time.sleep(WAIT_SECONDS)
            log_attempt(file_name, sample_idx, sample_name, f"claude:{CLAUDE_MODEL}", attempt, f"ERROR: {err}")
    return None, None


def judge_gpt_pair(literal: str, cultural: str, file_name: str, sample_idx: int, sample_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not OPENAI_API_KEY:
        return None, None
    client = OpenAI(api_key=OPENAI_API_KEY)
    p = build_prompt_pair(literal, cultural)
    err = None
    for attempt in range(RETRIES):
        try:
            # r = client.responses.create(model=GPT_MODEL, input=p)
            # s = r.output_text.strip() if hasattr(r, "output_text") else json.dumps(r.dict())
            r = client.chat.completions.create(model=GPT_MODEL, messages=[{"role": "user", "content": p}])
            s = r.choices[0].message.content.strip()
            log_attempt(file_name, sample_idx, sample_name, f"gpt:{GPT_MODEL}", attempt, s)
            return parse_pair(s)
        except Exception as e:
            err = str(e); time.sleep(WAIT_SECONDS)
            log_attempt(file_name, sample_idx, sample_name, f"gpt:{GPT_MODEL}", attempt, f"ERROR: {err}")
    return None, None


def judge_claude_pair(literal: str, cultural: str, file_name: str, sample_idx: int, sample_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not DEEPSEEK_API_KEY:
        return None, None
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    p = build_prompt_pair(literal, cultural)
    err = None
    for attempt in range(RETRIES):
        try:
            payload = {"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0}
            r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status(); s = r.json()["choices"][0]["message"]["content"].strip()
            log_attempt(file_name, sample_idx, sample_name, f"deepseek:{DEEPSEEK_MODEL}", attempt, s)
            return parse_pair(s)
        except Exception as e:
            err = str(e); time.sleep(WAIT_SECONDS)
            log_attempt(file_name, sample_idx, sample_name, f"deepseek:{DEEPSEEK_MODEL}", attempt, f"ERROR: {err}")
    return None, None


def majority_vote(values) -> Optional[str]:
    votes = [v for v in values if v]
    return Counter(votes).most_common(1)[0][0] if votes else None


def annotate_file(file_name: str, limit: int = 2) -> str:
    in_path = os.path.join(OUTPUT_DIR, file_name)
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    df = pd.read_csv(in_path)
    cols = list(df.columns)
    lit = "literal_translation" if "literal_translation" in cols else None
    cul = "cultural_translation" if "cultural_translation" in cols else None
    orig = "original_text" if "original_text" in cols else None

    rows = []
    if limit is None:
        subset = df
        total = len(df)
    else:
        subset = df.head(max(0, int(limit)))
        total = min(int(limit), len(df))
    for idx, (_, row) in enumerate(tqdm(subset.iterrows(), total=total)):
        original = str(row[orig]) if orig else ""
        literal = (str(row[lit]).strip() if lit else "").strip()
        cultural = (str(row[cul]).strip() if cul else "").strip()

        skip_lit = (literal == "")
        skip_cul = (cultural == "")

        g_lit = g_cul = c_lit = c_cul = d_lit = d_cul = None
        sample_name = original
        if not (skip_lit and skip_cul):
            g_lit, g_cul = judge_gpt_pair(literal, cultural, file_name, idx, sample_name)
            c_lit, c_cul = judge_claude_pair(literal, cultural, file_name, idx, sample_name)

        need_ds_lit = (not skip_lit) and not (g_lit and c_lit and g_lit == c_lit)
        need_ds_cul = (not skip_cul) and not (g_cul and c_cul and g_cul == c_cul)
        if need_ds_lit or need_ds_cul:
            d_lit, d_cul = judge_deepseek_pair(literal, cultural, file_name, idx, sample_name)

        # When a side is skipped, ALL its labels should be empty (including DeepSeek)
        if skip_lit:
            g_lit, c_lit, d_lit = "", "", ""
        if skip_cul:
            g_cul, c_cul, d_cul = "", "", ""

        ds_lit_field = ("" if skip_lit else ("-1" if not need_ds_lit else (d_lit if d_lit is not None else "")))
        ds_cul_field = ("" if skip_cul else ("-1" if not need_ds_cul else (d_cul if d_cul is not None else "")))
        lit_labels = [g_lit, ds_lit_field, c_lit]
        cul_labels = [g_cul, ds_cul_field, c_cul]
        if skip_lit:
            lit_final = ""
        elif not need_ds_lit:
            lit_final = g_lit
        else:
            votes = [x for x in [g_lit, d_lit, c_lit] if x]
            if votes and len(set(votes)) == 1:
                lit_final = votes[0]
            else:
                cnt = Counter(votes)
                lit_final = cnt.most_common(1)[0][0] if cnt else d_lit
        if skip_cul:
            cul_final = ""
        elif not need_ds_cul:
            cul_final = g_cul
        else:
            votes = [x for x in [g_cul, d_cul, c_cul] if x]
            if votes and len(set(votes)) == 1:
                cul_final = votes[0]
            else:
                cnt = Counter(votes)
                cul_final = cnt.most_common(1)[0][0] if cnt else d_cul
        rows.append({
            "original_text": original,
            "literal_translation": literal,
            "cultural_translation": cultural,
            "gpt_label_literal": lit_labels[0],
            "claude_label_literal": ds_lit_field,
            "deepseek_label_literal": lit_labels[2],
            "literal_machine_label": lit_final,
            "gpt_label_cultural": cul_labels[0],
            "claude_label_cultural": ds_cul_field,
            "deepseek_label_cultural": cul_labels[2],
            "cultural_machine_label": cul_final,
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    # Full run only when limit is None, regardless of dataset length
    is_full = (limit is None)
    base_name = file_name
    if not is_full:
        if "_sample" not in base_name:
            if base_name.endswith("_translations.csv"):
                base_name = base_name.replace("_translations.csv", "_sample_translations.csv")
            else:
                root, ext = os.path.splitext(base_name)
                base_name = f"{root}_sample{ext}"
    else:
        base_name = base_name.replace("_sample_translations.csv", "_translations.csv")
    out_path = os.path.join(ANNOTATION_DIR, f"Annotation_{base_name}")
    out_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    out = annotate_file(FILE_NAME, LIMIT)
    print(f"Wrote: {out}")

# Semantic Label Drift in Cross-Cultural Translation

[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-2026.lrec--1.297-ed1c24?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iI2ZmZiI+PHBhdGggZD0iTTUgM2gxNHYxOGwtNy00LTcgNFYzeiIvPjwvc3ZnPg==)](https://aclanthology.org/2026.lrec-1.297/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.25967-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.25967)
[![DOI](https://img.shields.io/badge/DOI-10.63317%2F5ae9txdv2s3g-2c6bac?style=flat-square)](https://doi.org/10.63317/5ae9txdv2s3g)
[![Code licence: MIT](https://img.shields.io/badge/Code-MIT-green?style=flat-square)](LICENSE)
[![Data licence: CC BY-NC 4.0](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey?style=flat-square)](data/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

Code and data for the LREC 2026 paper *Semantic Label Drift in Cross-Cultural
Translation* by Mohsinul Kabir, Tasnim Ahmed, Md Mezbaur Rahman, Polydoros
Giannouris, and Sophia Ananiadou.

Machine translation is routinely used to build training data for low-resource
languages by translating from high-resource ones. This paper asks whether the
**label** survives that trip. It does not, reliably. Translating a tweet
labelled *moderately depressed* or *ironic* into another language can move it
into a different class, and the size of that shift tracks how far apart the two
cultures are.

The paper establishes three findings:

1. MT systems, including modern LLMs, induce label drift during translation,
   and the effect is strongest in culturally sensitive domains.
2. Unlike earlier statistical MT tools, LLMs encode cultural knowledge, and
   asking them to *use* that knowledge amplifies the drift rather than reducing it.
3. Cultural similarity between source and target language is a key determinant
   of whether the label is preserved.

---

## Experimental design

Each English tweet is translated twice by the same system:

| Condition | Prompt asks for |
| --- | --- |
| **Literal** | word-for-word accuracy, no cultural adjustment |
| **Cultural** | the same content re-expressed as a local speaker would put it |

Both translations are then relabelled from scratch by a panel of LLM
annotators, using the same guidelines the original corpus used. Drift is the
gap between the original gold label and the label the translated text now
attracts. The literal and cultural conditions isolate how much of the drift
comes from cultural adaptation specifically.

**Languages.** English into Bengali and Greek, chosen for their differing
cultural distance from the English-language source data.

**Translation systems.** `claude-3-5-haiku`, `gpt-4.1-mini`, `deepseek-chat`,
and `facebook/nllb-200-distilled-1.3B` as a non-LLM statistical baseline. NLLB
produces a literal translation only, which is why it serves as the control.

**Annotator panel.** `gpt-4.1-mini` and `claude-3-5-haiku` label every item.
`deepseek-chat` is called only to break a tie, and the final label is the
majority vote. Every raw judge response is written to a log so the votes can be
audited.

---

## Datasets

Both files hold 1,150 English tweets with four-way labels.

| File | Source corpus | Label column | Text column | Classes |
| --- | --- | --- | --- | --- |
| [`data/deptweet_main.csv`](data/deptweet_main.csv) | DepTweet | `target` (`label` is the name) | `tweet` | non-depressed, mild, moderate, severe |
| [`data/irony_main.csv`](data/irony_main.csv) | SemEval-2018 Task 3 | `label` (`class` is the name) | `tweet_text` | non-ironic, ironic-by-clash, situational-irony, other-irony |

DepTweet is close to balanced at roughly 287 items per class. The irony set is
not: 399 non-ironic, 337 ironic-by-clash, 214 situational, 200 other.

Depression severity is the culturally sensitive domain; irony is the
comparatively neutral one. The contrast between them is what finding (1) rests on.

See [`data/LICENSE`](data/LICENSE) for redistribution terms and the upstream
citations you also need to give.

---

## Repository layout

```
label_drift/
├── data/                 1,150-tweet source sets, plus their licence
├── prompts/
│   └── prompts.yaml      dual-translation and annotation prompts, per language
├── src/
│   ├── generation/       English into Bengali/Greek, literal and cultural
│   ├── annotation/       LLM annotator panel with majority vote
│   └── evaluation/       drift, agreement, and refusal metrics
└── outputs/              written by the pipeline, git-ignored
    ├── translations/
    ├── annotations/
    │   └── logs/
    └── evaluation/
```

Every script resolves its paths from its own location, so you can run them from
any working directory.

---

## Setup

```bash
git clone https://github.com/mohsinulkabir14/label_drift.git
cd label_drift
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then add your API keys
```

`torch`, `transformers`, and `sentencepiece` are only needed for the NLLB
baseline. Skip them if you are running the API-based systems only.

---

## Running the pipeline

The scripts are configured by editing the constant block at the top of each
file, not by command-line flags. The relevant constants are named in each step
below.

### 1. Translate

```bash
python src/generation/claude_generate.py
```

Set `DATASET` (`irony` or `deptweet`), `LANG` (`bengali` or `greek`), and
`LIMIT` (`None` for the full file, or an integer for a quick trial). The other
backends are `gpt_generate.py`, `deepseek_generate.py`, and `nllb_generate.py`.

Writes `outputs/translations/{dataset}_{lang}_{model}_full_translations.csv`
with columns `original_text`, `literal_translation`, `cultural_translation`.
A failed or refused translation is left empty and carries an `error` value, so
refusals stay countable rather than silently dropping rows.

### 2. Annotate

```bash
python src/annotation/annotate_v2.py
```

Set `FILE_NAME` to a CSV in `outputs/translations/`, and `LANG` to match it.
The script refuses to start if the two disagree.

Writes `outputs/annotations/Annotation_<file>.csv` with each judge's vote and
the resulting `literal_machine_label` and `cultural_machine_label`. Raw judge
responses land in `outputs/annotations/logs/`.

Three variants exist:

| Script | Use |
| --- | --- |
| `annotate_v2.py` | the main path: literal and cultural, 10 retries |
| `annotate_literal.py` | literal-only output such as NLLB; also checks `TASK` |
| `annotate.py` | the earlier version, kept for provenance. It uses `claude-3-haiku-20240307` and 3 retries, so it will not reproduce the paper |

### 3. Evaluate

```bash
python src/evaluation/comprehensive_evaluation_report.py
```

Takes no configuration. It walks every `Annotation_*.csv` in
`outputs/annotations/`, skips anything with `sample` in the name, and writes one
report per file to `outputs/evaluation/`.

Each report covers:

- **Drift**, as accuracy, per-class recall, precision, F1, MCC, and a
  confusion matrix against the original gold labels, plus KL divergence between
  the original and post-translation label distributions
- **Agreement** among the judges, as Fleiss kappa, Krippendorff alpha, and
  pairwise Cohen kappa
- **Refusals**, counted overall and per class, separately for the literal and
  cultural conditions

---

## Notes on reproduction

- **Row order matters.** The evaluation step attaches gold labels by row
  position, not by matching text, so every stage must keep the rows of
  `data/*.csv` in their original order and must not drop any. This is why the
  generation and annotation scripts emit an empty row for a failed item instead
  of skipping it. If you filter rows, the labels will silently misalign.
- API models drift over time, and translation is sampled at `temperature=0.5`,
  so exact numbers will move between runs. The direction and relative size of
  the effects is what should reproduce.
- Annotation runs at `temperature=0` to keep the judges as stable as possible.
- A full run is 1,150 items per dataset, language, and system. Budget for it.
- `gpt_generate.py` uses `gpt-4.1-mini`. It was named `gpt5_generate.py` in an
  earlier version of this repository, which was misleading.

---

## Licence

Code is MIT, see [`LICENSE`](LICENSE). The CSV files in `data/` are CC BY-NC
4.0, see [`data/LICENSE`](data/LICENSE); the upstream DepTweet and SemEval-2018
terms apply to them as well.

---

## Citation

```bibtex
@inproceedings{kabir-etal-2026-semantic,
    title     = "Semantic Label Drift in Cross-Cultural Translation",
    author    = "Kabir, Mohsinul  and
                 Ahmed, Tasnim  and
                 Rahman, Md Mezbaur  and
                 Giannouris, Polydoros  and
                 Ananiadou, Sophia",
    editor    = "Piperidis, Stelios  and
                 Bel, N{\'u}ria  and
                 van den Heuvel, Henk  and
                 Ide, Nancy  and
                 Krek, Simon  and
                 Toral, Antonio",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    month     = may,
    year      = "2026",
    address   = "Palma de Mallorca, Spain",
    publisher = "ELRA Language Resource Association",
    url       = "https://aclanthology.org/2026.lrec-1.297/",
    doi       = "10.63317/5ae9txdv2s3g",
    pages     = "3714--3724"
}
```

---

## Contact

Mohsinul Kabir, <mdmohsinul.kabir@manchester.ac.uk>

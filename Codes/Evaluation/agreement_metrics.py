import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from collections import Counter

def _labels_sorted(annotations: pd.DataFrame):
    """Robust label discovery"""
    labs = pd.unique(annotations.stack().dropna().astype(int))
    return sorted(labs.tolist())

def _build_count_matrix(annotations: pd.DataFrame, labels):
    """Build count matrix for Fleiss Kappa"""
    label_index = {lab: idx for idx, lab in enumerate(labels)}
    N, K = len(annotations), len(labels)
    M = np.zeros((N, K), dtype=int)
    for i, (_, row) in enumerate(annotations.iterrows()):
        for lab in row.dropna():
            M[i, label_index[int(lab)]] += 1
    return M

def calculate_fleiss_kappa(annotations: pd.DataFrame):
    """Calculate Fleiss Kappa"""
    if annotations.empty:
        return np.nan
    
    labels = _labels_sorted(annotations)
    M = _build_count_matrix(annotations, labels)
    N = M.shape[0]
    row_sums = M.sum(axis=1)
    
    # Use only items with at least two ratings
    valid = row_sums > 1
    if valid.sum() == 0:
        return np.nan
    
    # P_i per item
    Pi = ((M[valid] ** 2).sum(axis=1) - row_sums[valid]) / (row_sums[valid] * (row_sums[valid] - 1))
    P_bar = Pi.mean()
    
    # p_j (category marginals)
    n_annotators = annotations.shape[1]
    pj = M.sum(axis=0) / (N * n_annotators) if (N * n_annotators) > 0 else np.zeros(M.shape[1])
    P_e = (pj ** 2).sum()
    
    denom = (1 - P_e)
    if np.isclose(denom, 0):
        return np.nan
    return (P_bar - P_e) / denom

def calculate_krippendorff_alpha(annotations_with_missing: pd.DataFrame) -> float:
    """Calculate Krippendorff's Alpha"""
    if annotations_with_missing.empty:
        return np.nan
    
    # Build per-item category counts
    def coerce_series(s):
        try:
            return s.dropna().astype(int)
        except Exception:
            return s.dropna().astype(str)
    
    items_counts = []
    all_labels = set()
    for _, row in annotations_with_missing.iterrows():
        vals = coerce_series(row)
        cnt = Counter(vals.tolist())
        items_counts.append(cnt)
        all_labels.update(cnt.keys())
    
    if not all_labels:
        return np.nan
    
    # Calculate coincidence counts
    denom_Do = 0.0
    obs_same = 0.0
    n_c = {c: 0 for c in all_labels}
    total_n = 0
    
    for cnt in items_counts:
        n_i = sum(cnt.values())
        denom_Do += n_i * (n_i - 1)
        obs_same += sum(v * (v - 1) for v in cnt.values())
        for c, v in cnt.items():
            n_c[c] += v
            total_n += v
    
    if denom_Do <= 0 or total_n <= 1:
        return np.nan
    
    Do = 1.0 - (obs_same / denom_Do)
    exp_same = sum(v * (v - 1) for v in n_c.values())
    denom_De = total_n * (total_n - 1)
    
    if denom_De <= 0:
        return np.nan
    
    De = 1.0 - (exp_same / denom_De)
    
    if np.isclose(De, 0.0):
        return np.nan
    
    return 1.0 - (Do / De)

def calculate_pairwise_kappas(annotations: pd.DataFrame):
    """Calculate pairwise Cohen's Kappa"""
    names = list(annotations.columns)
    results = []
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a1, a2 = names[i], names[j]
            mask = annotations[a1].notna() & annotations[a2].notna()
            n = int(mask.sum())
            
            if n == 0:
                kappa = np.nan
            else:
                s1 = annotations.loc[mask, a1].astype(int)
                s2 = annotations.loc[mask, a2].astype(int)
                kappa = cohen_kappa_score(s1, s2)
            
            pair_name = f"{a1} vs {a2}"
            results.append({"pair": pair_name, "kappa": kappa, "n": n})
    
    return pd.DataFrame(results).set_index("pair")

def preprocess_annotations(df):
    """Preprocess annotations for agreement analysis"""
    has_cultural = 'cultural_translation' in df.columns
    
    # Literal annotations
    literal_cols = ['gpt_label_literal', 'deepseek_label_literal', 'claude_label_literal']
    literal = df[literal_cols].copy()
    
    # Replace -1 with GPT's label
    for col in ['deepseek_label_literal', 'claude_label_literal']:
        mask = literal[col] == -1
        literal.loc[mask, col] = literal.loc[mask, 'gpt_label_literal']
    
    literal_with_miss = literal.copy()
    literal_complete = literal.dropna().copy()
    
    if has_cultural:
        # Cultural annotations
        cultural_cols = ['gpt_label_cultural', 'deepseek_label_cultural', 'claude_label_cultural']
        cultural = df[cultural_cols].copy()
        
        # Replace -1 with GPT's label
        for col in ['deepseek_label_cultural', 'claude_label_cultural']:
            mask = cultural[col] == -1
            cultural.loc[mask, col] = cultural.loc[mask, 'gpt_label_cultural']
        
        cultural_with_miss = cultural.copy()
        cultural_complete = cultural.dropna().copy()
    else:
        # No cultural annotations
        cultural_with_miss = pd.DataFrame()
        cultural_complete = pd.DataFrame()
    
    print(f"Literal (complete-case for Fleiss/Cohen): {len(literal_complete)} items")
    print(f"Literal (with missing for Alpha)       : {len(literal_with_miss)} items")
    
    if has_cultural:
        print(f"Cultural (complete-case for Fleiss/Cohen): {len(cultural_complete)} items")
        print(f"Cultural (with missing for Alpha)        : {len(cultural_with_miss)} items")
    else:
        print("Cultural annotations: Not available (gtranslate file)")
    
    return literal_complete, cultural_complete, literal_with_miss, cultural_with_miss

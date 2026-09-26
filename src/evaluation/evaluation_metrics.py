import pandas as pd
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score, 
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from scipy.stats import entropy

def calculate_evaluation_metrics(df, dataset_type):
    """Calculate evaluation metrics comparing original vs machine labels"""
    # Create evaluation dataframe
    has_cultural = 'cultural_translation' in df.columns
    
    if has_cultural:
        df_eval = df[['target', 'label', 'literal_machine_label', 'cultural_machine_label']].copy()
        df_eval['original_label_numeric'] = df_eval['target']
        
        # Remove rows with missing machine labels
        df_eval = df_eval.dropna(subset=['literal_machine_label', 'cultural_machine_label'])
    else:
        # Only literal translations
        df_eval = df[['target', 'label', 'literal_machine_label']].copy()
        df_eval['original_label_numeric'] = df_eval['target']
        
        # Remove rows with missing machine labels
        df_eval = df_eval.dropna(subset=['literal_machine_label'])
    
    # Ensure all labels are integers
    df_eval['original_label_numeric'] = pd.to_numeric(df_eval['original_label_numeric'], errors='coerce')
    df_eval['literal_machine_label'] = pd.to_numeric(df_eval['literal_machine_label'], errors='coerce')
    
    if has_cultural:
        df_eval['cultural_machine_label'] = pd.to_numeric(df_eval['cultural_machine_label'], errors='coerce')
    
    # Remove any rows where conversion failed
    df_eval = df_eval.dropna(subset=['original_label_numeric', 'literal_machine_label'])
    if has_cultural:
        df_eval = df_eval.dropna(subset=['cultural_machine_label'])
    
    # Convert to integers
    df_eval['original_label_numeric'] = df_eval['original_label_numeric'].astype(int)
    df_eval['literal_machine_label'] = df_eval['literal_machine_label'].astype(int)
    if has_cultural:
        df_eval['cultural_machine_label'] = df_eval['cultural_machine_label'].astype(int)
    
    print(f"Evaluation samples: {len(df_eval)}")
    
    labels = [0, 1, 2, 3]  # keep a single source of truth for label order

    # Calculate Matthews Correlation Coefficient
    literal_overall_mcc = matthews_corrcoef(df_eval['original_label_numeric'], df_eval['literal_machine_label'])
    cultural_overall_mcc = matthews_corrcoef(df_eval['original_label_numeric'], df_eval['cultural_machine_label']) if has_cultural else np.nan
    
    # Per-class MCC
    literal_mcc = labelwise_mcc(df_eval['original_label_numeric'], df_eval['literal_machine_label'], labels)
    cultural_mcc = labelwise_mcc(df_eval['original_label_numeric'], df_eval['cultural_machine_label'], labels) if has_cultural else pd.Series([np.nan]*len(labels), index=labels)
    
    # Calculate Confusion Matrices
    literal_cm = confusion_matrix(df_eval['original_label_numeric'], df_eval['literal_machine_label'], labels=labels)
    cultural_cm = confusion_matrix(df_eval['original_label_numeric'], df_eval['cultural_machine_label'], labels=labels) if has_cultural else None
    
    # Print Confusion Matrices
    print("\n" + "="*60)
    print("🎯 CONFUSION MATRIX - LITERAL TRANSLATION")
    print("="*60)
    print("📊 True labels (rows) vs Predicted labels (columns)")
    print(f"🏷️  Classes: {labels}")
    print()
    
    # Create a beautiful formatted confusion matrix
    cm_df = pd.DataFrame(literal_cm, 
                        index=[f'Class {i}' for i in labels],
                        columns=[f'Pred {i}' for i in labels])
    
    # Add row and column totals
    cm_df['Total'] = cm_df.sum(axis=1)
    cm_df.loc['Total'] = cm_df.sum()
    
    # Format the matrix with proper alignment
    print("┌" + "─"*50 + "┐")
    print("│" + " "*20 + "PREDICTED" + " "*21 + "│")
    print("├" + "─"*50 + "┤")
    
    # Header row
    header = "│" + f"{'TRUE':<12}"
    for col in cm_df.columns:
        header += f"{col:>8}"
    header += "│"
    print(header)
    
    # Separator line
    print("├" + "─"*50 + "┤")
    
    # Data rows
    for idx, row in cm_df.iterrows():
        if idx == 'Total':
            print("├" + "─"*50 + "┤")
        row_str = "│" + f"{idx:<12}"
        for col in cm_df.columns:
            row_str += f"{int(row[col]):>8}"
        row_str += "│"
        print(row_str)
    
    print("└" + "─"*50 + "┘")
    
    if has_cultural:
        print("\n" + "="*60)
        print("CONFUSION MATRIX - CULTURAL TRANSLATION")
        print("="*60)
        print("True labels (rows) vs Predicted labels (columns)")
        print(f"Classes: {labels}")
        print()
        
        cm_df_cultural = pd.DataFrame(cultural_cm, 
                                    index=[f'Class {i}' for i in labels],
                                    columns=[f'Pred {i}' for i in labels])
        
        cm_df_cultural['Total'] = cm_df_cultural.sum(axis=1)
        cm_df_cultural.loc['Total'] = cm_df_cultural.sum()
        
        print("┌" + "─"*50 + "┐")
        print("│" + " "*20 + "PREDICTED" + " "*21 + "│")
        print("├" + "─"*50 + "┤")
        
        header = "│" + f"{'TRUE':<12}"
        for col in cm_df_cultural.columns:
            header += f"{col:>8}"
        header += "│"
        print(header)
        
        print("├" + "─"*50 + "┤")
        
        for idx, row in cm_df_cultural.iterrows():
            if idx == 'Total':
                print("├" + "─"*50 + "┤")
            row_str = "│" + f"{idx:<12}"
            for col in cm_df_cultural.columns:
                row_str += f"{int(row[col]):>8}"
            row_str += "│"
            print(row_str)
        
        print("└" + "─"*50 + "┘")
    else:
        print("\n" + "="*60)
        print("CULTURAL TRANSLATION CONFUSION MATRIX")
        print("="*60)
        print("Not available (gtranslate file - no cultural translations)")
        print("="*60)
    
    # NEW: Per-class accuracy (== per-class recall) using sklearn
    literal_per_class_acc = recall_score(
        df_eval['original_label_numeric'],
        df_eval['literal_machine_label'],
        labels=labels,
        average=None,
        zero_division=0
    )
    cultural_per_class_acc = (recall_score(
        df_eval['original_label_numeric'],
        df_eval['cultural_machine_label'],
        labels=labels,
        average=None,
        zero_division=0
    ) if has_cultural else np.array([np.nan]*len(labels)))
    
    # Pretty print per-class accuracy sections
    def _print_per_class(title, arr):
        print("\n" + "="*60)
        print(f"PER-CLASS ACCURACY (via sklearn.recall_score) - {title}")
        print("="*60)
        print("│ {:<10} │ {:>14} │".format("Class", "Accuracy"))
        print("├" + "─"*12 + "┼" + "─"*16 + "┤")
        for lab, val in zip(labels, arr):
            print("│ {:<10} │ {:>13.3f} │".format(f"{lab}", float(val)))
        print("└" + "─"*12 + "┴" + "─"*16 + "┘")
    
    _print_per_class("LITERAL", literal_per_class_acc)
    if has_cultural:
        _print_per_class("CULTURAL", cultural_per_class_acc)
    else:
        print("\n" + "="*60)
        print("PER-CLASS ACCURACY (CULTURAL)")
        print("="*60)
        print("Not available (gtranslate file - no cultural translations)")
    
    # KL Divergence for literal labels
    p_literal = df_eval['original_label_numeric'].value_counts(normalize=True).sort_index()
    q_literal = df_eval['literal_machine_label'].value_counts(normalize=True).sort_index()
    
    # KL divergence for cultural labels
    if has_cultural:
        p_cultural = df_eval['original_label_numeric'].value_counts(normalize=True).sort_index()
        q_cultural = df_eval['cultural_machine_label'].value_counts(normalize=True).sort_index()
        
        kl_cultural_orig_to_cul = kl_divergence(p_cultural, q_cultural)
        kl_cultural_cul_to_orig = kl_divergence(q_cultural, p_cultural)
    else:
        kl_cultural_orig_to_cul = np.nan
        kl_cultural_cul_to_orig = np.nan
    
    # Overall Accuracy
    literal_accuracy = (df_eval['original_label_numeric'] == df_eval['literal_machine_label']).mean()
    cultural_accuracy = (df_eval['original_label_numeric'] == df_eval['cultural_machine_label']).mean() if has_cultural else np.nan
    
    return {
        'df_eval': df_eval,
        'literal_overall_mcc': literal_overall_mcc,
        'cultural_overall_mcc': cultural_overall_mcc,
        'literal_mcc': literal_mcc,
        'cultural_mcc': cultural_mcc,
        'literal_accuracy': literal_accuracy,
        'cultural_accuracy': cultural_accuracy,
        'literal_confusion_matrix': literal_cm,
        'cultural_confusion_matrix': cultural_cm,
        # NEW: expose per-class accuracy
        'literal_per_class_accuracy': pd.Series(literal_per_class_acc, index=labels),
        'cultural_per_class_accuracy': pd.Series(cultural_per_class_acc, index=labels),
        'kl_literal_orig_to_lit': kl_divergence(p_literal, q_literal),
        'kl_literal_lit_to_orig': kl_divergence(q_literal, p_literal),
        'kl_cultural_orig_to_cul': kl_cultural_orig_to_cul,
        'kl_cultural_cul_to_orig': kl_cultural_cul_to_orig,
        'has_cultural': has_cultural
    }

def labelwise_mcc(y_true, y_pred, labels):
    """Calculate per-class MCC"""
    mcc_scores = []
    for label in labels:
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        mcc_scores.append(mcc)
    return pd.Series(mcc_scores, index=labels)

def kl_divergence(p, q):
    """Calculate KL(P||Q)"""
    _p = np.array(p) + 1e-10  # Avoid division by zero
    _q = np.array(q) + 1e-10
    return entropy(_p, _q)

def generate_classification_reports(eval_metrics, class_names):
    """Generate classification reports for literal and cultural translations"""
    df_eval = eval_metrics['df_eval']
    
    # Literal translation report
    lit_report = classification_report(
        df_eval['original_label_numeric'], 
        df_eval['literal_machine_label'], 
        labels=[0, 1, 2, 3],
        target_names=class_names, 
        digits=3, zero_division=0
    )
    
    # Cultural translation report (if available)
    if eval_metrics['has_cultural']:
        cul_report = classification_report(
            df_eval['original_label_numeric'], 
            df_eval['cultural_machine_label'], 
            labels=[0, 1, 2, 3],
            target_names=class_names, 
            digits=3, zero_division=0
        )
    else:
        cul_report = None
    
    return lit_report, cul_report

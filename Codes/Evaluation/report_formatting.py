import pandas as pd
import numpy as np

def section_header(title: str, width: int = 70, char: str = "="):
    """Create a nicely formatted section header"""
    title_centered = f" {title} "
    total_width = len(title_centered) + 4
    left_pad = (width - total_width) // 2
    right_pad = width - total_width - left_pad
    return f"{char * left_pad}{title_centered}{char * right_pad}"

def format_metrics_table(metrics_df: pd.DataFrame) -> str:
    """Format the overall metrics table nicely"""
    if metrics_df.empty:
        return "Overall Metrics:\n"
    
    formatted = "Overall Metrics:\n"
    formatted += "-" * 40 + "\n"
    formatted += f"{'Metric':<20} {'Value':<12} {'Items':<8}\n"
    formatted += "-" * 40 + "\n"
    
    for idx, row in metrics_df.iterrows():
        metric_name = idx.replace('_', ' ').title()
        value = f"{row['value']:.4f}" if pd.notna(row['value']) and not np.isnan(row['value']) else "N/A"
        items = str(int(row['items_used'])) if pd.notna(row['items_used']) else "N/A"
        
        formatted += f"{metric_name:<20} {value:<12} {items:<8}\n"
    
    return formatted + "\n"

def format_pairwise_table(kappa_table: pd.DataFrame) -> str:
    """Format the pairwise kappa table nicely"""
    if kappa_table.empty:
        return "Pairwise statistics: (no data)\n"
    
    formatted = "Pairwise Cohen's κ:\n"
    formatted += "-" * 50 + "\n"
    formatted += f"{'Annotator Pair':<30} {'Kappa':<12} {'Samples':<8}\n"
    formatted += "-" * 50 + "\n"
    
    for idx, row in kappa_table.iterrows():
        # Clean up the pair names to just show gpt, deepseek, claude
        pair_name = idx.replace('_label_literal', '').replace('_label_cultural', '').replace('_', ' ')
        kappa = f"{row['kappa']:.4f}" if pd.notna(row['kappa']) and not np.isnan(row['kappa']) else "N/A"
        samples = str(int(row['n'])) if pd.notna(row['n']) else "N/A"
        
        formatted += f"{pair_name:<30} {kappa:<12} {samples:<8}\n"
    
    return formatted + "\n"

def format_confusion_matrix(cm: pd.DataFrame, title: str) -> str:
    """Format confusion matrix with proper alignment and labels"""
    if cm.empty:
        return f"{title}: (no data)\n"
    
    # Add row and column totals
    cm_with_totals = cm.copy()
    cm_with_totals['Total'] = cm_with_totals.sum(axis=1)
    cm_with_totals.loc['Total'] = cm_with_totals.sum()
    
    # Format the matrix
    formatted = f"\n{title}:\n"
    formatted += "-" * (len(title) + 1) + "\n"
    
    # Header row
    header = f"{'Predicted':<12}"
    for col in cm_with_totals.columns:
        header += f"{col:>8}"
    formatted += header + "\n"
    
    # Data rows
    for idx, row in cm_with_totals.iterrows():
        if idx == 'Total':
            formatted += "-" * (12 + 8 * len(cm_with_totals.columns)) + "\n"
        row_str = f"{idx:<12}"
        for col in cm_with_totals.columns:
            row_str += f"{cm_with_totals.loc[idx, col]:>8}"
        formatted += row_str + "\n"
    
    return formatted + "\n"

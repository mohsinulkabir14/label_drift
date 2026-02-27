import os
import pandas as pd
import numpy as np

# Import modules
from utils import (
    extract_dataset_info, generate_output_filename, get_class_mapping,
    get_dataset_display_name, get_model_display_name, load_data
)
from refusal_analysis import calculate_refusal_stats, per_class_refusal_stats
from evaluation_metrics import calculate_evaluation_metrics, generate_classification_reports
from agreement_metrics import (
    calculate_fleiss_kappa, calculate_krippendorff_alpha, 
    calculate_pairwise_kappas, preprocess_annotations
)
from report_formatting import (
    section_header, format_metrics_table, format_pairwise_table
)

# Configuration
OUTPUT_DIR = "../../Output/Evaluation"

def process_all_annotation_files():
    """Automatically process all annotation files in the Annotation folder"""
    annotation_dir = "../../Output/Annotation"
    annotation_files = []
    
    # Find all annotation CSV files (ignore sample files)
    for file in os.listdir(annotation_dir):
        if (file.startswith("Annotation_") and 
            file.endswith(".csv") and 
            "sample" not in file.lower()):
            annotation_files.append(os.path.join(annotation_dir, file))
    
    print(f"Found {len(annotation_files)} annotation files (excluding samples):")
    for file in annotation_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    for annotation_file in annotation_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(annotation_file)}")
        print(f"{'='*60}")
        
        # Generate report for this file
        try:
            generate_comprehensive_report(annotation_file)
            print(f"Successfully processed: {os.path.basename(annotation_file)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(annotation_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def generate_comprehensive_report(annotation_file):
    """Generate the comprehensive evaluation report"""
    print("Generating comprehensive evaluation report...")
    
    # Load data
    df, dataset_type = load_data(annotation_file)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extract dataset and model info from filename
    dataset_info = extract_dataset_info(annotation_file)
    dataset_name = get_dataset_display_name(dataset_info[0])
    model_name = get_model_display_name(dataset_info[2])
    
    # Generate output filename
    output_filename = generate_output_filename(dataset_info[0], dataset_info[1], dataset_info[2])
    OUTPUT_TXT = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"Dataset: {dataset_name}")
    print(f"Language: {dataset_info[1]}")
    print(f"Model: {model_name}")
    print(f"Output file: {output_filename}")
    
    # 1. Translation Refusal Analysis
    print("1. Analyzing translation refusals...")
    literal_refused, cultural_refused, refusal_stats = calculate_refusal_stats(df)
    class_mapping = get_class_mapping(dataset_type)
    class_refusal_stats = per_class_refusal_stats(df, literal_refused, cultural_refused, class_mapping)
    
    # 2. Evaluation Metrics
    print("2. Calculating evaluation metrics...")
    eval_metrics = calculate_evaluation_metrics(df, dataset_type)
    
    # 3. Inter-Annotator Agreement
    print("3. Calculating inter-annotator agreement...")
    lit_complete, cult_complete, lit_with_miss, cult_with_miss = preprocess_annotations(df)
    
    # Calculate agreement metrics
    lit_fleiss = calculate_fleiss_kappa(lit_complete)
    lit_alpha = calculate_krippendorff_alpha(lit_with_miss)
    
    if not cult_complete.empty:
        cult_fleiss = calculate_fleiss_kappa(cult_complete)
        cult_alpha = calculate_krippendorff_alpha(cult_with_miss)
    else:
        cult_fleiss = cult_alpha = np.nan
    
    # Pairwise kappas
    lit_pairwise = calculate_pairwise_kappas(lit_complete)
    
    if not cult_complete.empty:
        cult_pairwise = calculate_pairwise_kappas(cult_complete)
    else:
        cult_pairwise = pd.DataFrame()
    
    # Generate report
    report_lines = []
    
    # Title
    report_lines.append(section_header("COMPREHENSIVE EVALUATION REPORT"))
    report_lines.append(f"Dataset: {dataset_name} {model_name}")
    report_lines.append(f"Language: {dataset_info[1].title()}")
    report_lines.append(f"Model: {model_name}")
    report_lines.append(f"Total Samples: {len(df)}")
    report_lines.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n")
    
    # 1. TRANSLATION REFUSAL ANALYSIS
    report_lines.append(section_header("TRANSLATION REFUSAL ANALYSIS"))
    
    # Overall refusal stats
    report_lines.append("Overall Refusal Statistics:\n")
    report_lines.append("-" * 40 + "\n")
    report_lines.append(f"{'Metric':<25} {'Count':<10} {'Percentage':<10}\n")
    report_lines.append("-" * 40 + "\n")
    report_lines.append(f"{'Total samples':<25} {refusal_stats['total_samples']:<10} {'100.00%':<10}\n")
    
    total_refused = refusal_stats['either_refused_count']
    refused_pct = f"{total_refused/refusal_stats['total_samples']*100:.2f}%"
    
    report_lines.append(f"{'Total refused':<25} {total_refused:<10} {refused_pct:<10}\n\n")
    
    # Per-class refusal stats
    report_lines.append("Per-Class Refusal Breakdown:\n")
    report_lines.append("-" * 50 + "\n")
    report_lines.append(f"{'Class':<20} {'Samples':<10} {'Refused':<10} {'Rate':<10}\n")
    report_lines.append("-" * 50 + "\n")
    
    for stat in class_refusal_stats:
        rate = f"{stat['either_refused']/stat['total']*100:.2f}%"
        report_lines.append(f"{stat['class']:<20} {stat['total']:<10} {stat['either_refused']:<10} {rate:<10}\n")
    
    report_lines.append("\n")
    
    # 2. EVALUATION METRICS
    report_lines.append(section_header("EVALUATION METRICS"))
    report_lines.append("Comparing Original Labels vs Machine Labels\n")
    
    # Get class names for the specific dataset
    class_names = [class_mapping[i] for i in range(4)]
    
    # Classification reports
    lit_report, cul_report = generate_classification_reports(eval_metrics, class_names)
    
    report_lines.append("LITERAL TRANSLATION METRICS:\n")
    report_lines.append("-" * 60 + "\n")
    report_lines.append(lit_report)
    
    if eval_metrics['has_cultural'] and cul_report is not None:
        report_lines.append("CULTURAL TRANSLATION METRICS:\n")
        report_lines.append("-" * 60 + "\n")
        report_lines.append(cul_report)
    
    # MCC
    report_lines.append("MATTHEWS CORRELATION COEFFICIENT:\n")
    report_lines.append("-" * 50 + "\n")
    report_lines.append(f"{'Class':<20} {'Literal MCC':<15} {'Cultural MCC':<15}\n")
    report_lines.append("-" * 50 + "\n")
    
    for i, class_name in enumerate(class_names):
        lit_mcc = eval_metrics['literal_mcc'].iloc[i] if not pd.isna(eval_metrics['literal_mcc'].iloc[i]) else 0
        cul_mcc = eval_metrics['cultural_mcc'].iloc[i] if not pd.isna(eval_metrics['cultural_mcc'].iloc[i]) else 0
        report_lines.append(f"{class_name:<20} {lit_mcc:<15.3f} {cul_mcc:<15.3f}\n")
    
    report_lines.append(f"\nOverall MCC:\n")
    report_lines.append(f"Literal: {eval_metrics['literal_overall_mcc']:.3f}\n")
    if eval_metrics['has_cultural']:
        report_lines.append(f"Cultural: {eval_metrics['cultural_overall_mcc']:.3f}\n")
    else:
        report_lines.append("Cultural: Not available (gtranslate file)\n")
    
    # KL Divergence
    report_lines.append("\nKL DIVERGENCE:\n")
    report_lines.append("-" * 50 + "\n")
    report_lines.append(f"{'Comparison':<25} {'Value':<15} {'Direction':<10}\n")
    report_lines.append("-" * 50 + "\n")
    report_lines.append(f"{'Literal':<25} {eval_metrics['kl_literal_orig_to_lit']:<15.4f} {'Orig→Lit':<10}\n")
    report_lines.append(f"{'Literal':<25} {eval_metrics['kl_literal_lit_to_orig']:<15.4f} {'Lit→Orig':<10}\n")
    
    if eval_metrics['has_cultural']:
        report_lines.append(f"{'Cultural':<25} {eval_metrics['kl_cultural_orig_to_cul']:<15.4f} {'Orig→Cul':<10}\n")
        report_lines.append(f"{'Cultural':<25} {eval_metrics['kl_cultural_cul_to_orig']:<15.4f} {'Cul→Orig':<10}\n")
    else:
        report_lines.append(f"{'Cultural':<25} {'N/A':<15} {'N/A':<10}\n")
        report_lines.append(f"{'Cultural':<25} {'N/A':<15} {'N/A':<10}\n")
    
    report_lines.append("\n")
    
    # Confusion Matrices
    report_lines.append("CONFUSION MATRICES:")
    report_lines.append("-" * 60 + "\n")
    report_lines.append("LITERAL TRANSLATION CONFUSION MATRIX:")
    report_lines.append("True labels (rows) vs Predicted labels (columns)")
    report_lines.append("Classes: [0, 1, 2, 3]")
    report_lines.append("\n")
    
    # Format the confusion matrix beautifully
    literal_cm = eval_metrics['literal_confusion_matrix']
    
    # Create a formatted table
    report_lines.append("┌" + "─"*50 + "┐")
    report_lines.append("│" + " "*20 + "PREDICTED" + " "*21 + "│")
    report_lines.append("├" + "─"*50 + "┤")
    
    # Header row
    header = "│" + f"{'TRUE':<12}"
    for col in ['Pred 0', 'Pred 1', 'Pred 2', 'Pred 3', 'Total']:
        header += f"{col:>8}"
    header += "│"
    report_lines.append(header)
    
    # Separator line
    report_lines.append("├" + "─"*50 + "┤")
    
    # Data rows with totals
    for i, row in enumerate(literal_cm):
        row_total = row.sum()
        row_str = "│" + f"{'Class '+str(i):<12}"
        for val in row:
            row_str += f"{int(val):>8}"
        row_str += f"{int(row_total):>8}│"
        report_lines.append(row_str)
    
    # Total row
    col_totals = literal_cm.sum(axis=0)
    grand_total = literal_cm.sum()
    report_lines.append("├" + "─"*50 + "┤")
    total_row = "│" + f"{'Total':<12}"
    for val in col_totals:
        total_row += f"{int(val):>8}"
    total_row += f"{int(grand_total):>8}│"
    report_lines.append(total_row)
    
    report_lines.append("└" + "─"*50 + "┘")
    
    # Add Cultural Confusion Matrix if available
    if eval_metrics['has_cultural']:
        report_lines.append("\nCULTURAL TRANSLATION CONFUSION MATRIX:")
        report_lines.append("True labels (rows) vs Predicted labels (columns)")
        report_lines.append("Classes: [0, 1, 2, 3]")
        report_lines.append("\n")
        
        cultural_cm = eval_metrics['cultural_confusion_matrix']
        
        # Create a formatted table
        report_lines.append("┌" + "─"*50 + "┐")
        report_lines.append("│" + " "*20 + "PREDICTED" + " "*21 + "│")
        report_lines.append("├" + "─"*50 + "┤")
        
        # Header row
        header = "│" + f"{'TRUE':<12}"
        for col in ['Pred 0', 'Pred 1', 'Pred 2', 'Pred 3', 'Total']:
            header += f"{col:>8}"
        header += "│"
        report_lines.append(header)
        
        # Separator line
        report_lines.append("├" + "─"*50 + "┤")
        
        # Data rows with totals
        for i, row in enumerate(cultural_cm):
            row_total = row.sum()
            row_str = "│" + f"{'Class '+str(i):<12}"
            for val in row:
                row_str += f"{int(val):>8}"
            row_str += f"{int(row_total):>8}│"
            report_lines.append(row_str)
        
        # Total row
        col_totals = cultural_cm.sum(axis=0)
        grand_total = cultural_cm.sum()
        report_lines.append("├" + "─"*50 + "┤")
        total_row = "│" + f"{'Total':<12}"
        for val in col_totals:
            total_row += f"{int(val):>8}"
        total_row += f"{int(grand_total):>8}│"
        report_lines.append(total_row)
        
        report_lines.append("└" + "─"*50 + "┘")
    else:
        report_lines.append("\nCULTURAL TRANSLATION CONFUSION MATRIX:")
        report_lines.append("Not available (gtranslate file - no cultural translations)")
    
    # Add Per-Class Accuracy Section
    report_lines.append("\n" + "="*60)
    report_lines.append("PER-CLASS ACCURACY (via sklearn.recall_score)")
    report_lines.append("="*60)
    
    # Literal Translation Per-Class Accuracy
    report_lines.append("\nLITERAL TRANSLATION PER-CLASS ACCURACY:")
    report_lines.append("Accuracy for each individual class")
    report_lines.append("\n")
    
    literal_per_class_acc = eval_metrics['literal_per_class_accuracy']
    for class_label in [0, 1, 2, 3]:
        accuracy = literal_per_class_acc[class_label]
        report_lines.append(f"Class {class_label}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Cultural Translation Per-Class Accuracy (if available)
    if eval_metrics['has_cultural']:
        report_lines.append("\nCULTURAL TRANSLATION PER-CLASS ACCURACY:")
        report_lines.append("Accuracy for each individual class")
        report_lines.append("\n")
        
        cultural_per_class_acc = eval_metrics['cultural_per_class_accuracy']
        for class_label in [0, 1, 2, 3]:
            accuracy = cultural_per_class_acc[class_label]
            report_lines.append(f"Class {class_label}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    else:
        report_lines.append("\nCULTURAL TRANSLATION PER-CLASS ACCURACY:")
        report_lines.append("Not available (gtranslate file - no cultural translations)")
    
    report_lines.append("\n")
    
    # 3. INTER-ANNOTATOR AGREEMENT
    report_lines.append(section_header("INTER-ANNOTATOR AGREEMENT"))
    report_lines.append("Agreement among GPT, Claude, and DeepSeek\n")
    
    # Literal annotations
    report_lines.append(section_header("LITERAL TRANSLATION ANNOTATIONS"))
    
    # Overall metrics
    overall_df = pd.DataFrame(
        {
            "value": [lit_fleiss, lit_alpha],
            "items_used": [len(lit_complete), len(lit_with_miss)],
        },
        index=["Fleiss_kappa", "Krippendorff_alpha"],
    )
    report_lines.append(format_metrics_table(overall_df))
    
    # Pairwise kappas
    if not lit_pairwise.empty:
        report_lines.append(format_pairwise_table(lit_pairwise))
    
    report_lines.append("\n")
    
    # Cultural annotations (if available)
    if not cult_complete.empty:
        report_lines.append(section_header("CULTURAL TRANSLATION ANNOTATIONS"))
        
        # Overall metrics
        overall_df = pd.DataFrame(
            {
                "value": [cult_fleiss, cult_alpha],
                "items_used": [len(cult_complete), len(cult_with_miss)],
            },
            index=["Fleiss_kappa", "Krippendorff_alpha"],
        )
        report_lines.append(format_metrics_table(overall_df))
        
        # Pairwise kappas
        if not cult_pairwise.empty:
            report_lines.append(format_pairwise_table(cult_pairwise))
        
        report_lines.append("\n")
    else:
        report_lines.append("CULTURAL TRANSLATION ANNOTATIONS: Not available (gtranslate file)\n\n")
    
    # 4. INTERPRETATION GUIDE
    report_lines.append(section_header("INTERPRETATION GUIDE"))
    report_lines.append("Guideline bins (rough / Landis & Koch):\n")
    report_lines.append("  < 0.00: Poor agreement\n")
    report_lines.append("  0.00 - 0.20: Slight agreement\n")
    report_lines.append("  0.21 - 0.40: Fair agreement\n")
    report_lines.append("  0.41 - 0.60: Moderate agreement\n")
    report_lines.append("  0.61 - 0.80: Substantial agreement\n")
    report_lines.append("  0.81 - 1.00: Almost perfect agreement\n\n")
    
    report_lines.append("MCC Scale: 0.00-0.30 (Weak), 0.31-0.50 (Weak-Moderate)\n")
    report_lines.append("           0.51-0.70 (Moderate-Strong), 0.71-1.00 (Strong)\n")
    report_lines.append("KL Divergence: Lower values indicate more similar distributions\n")
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nComprehensive report saved to: {OUTPUT_TXT}")
    
    # Console preview
    print("\n" + section_header("SUMMARY"))
    print(report_text[:1000] + ("...\n" if len(report_text) > 1000 else "\n"))
    
    return report_text

if __name__ == "__main__":
    process_all_annotation_files()
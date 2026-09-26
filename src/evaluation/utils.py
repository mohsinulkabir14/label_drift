import os
import pandas as pd
import numpy as np
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def extract_dataset_info(filename):
    """Extract dataset and model information from filename"""
    # Example: Annotation_deptweet_bengali_gpt_4.1_mini_full_translations.csv
    # Extract: dataset=deptweet, language=bengali, model=gpt_4.1_mini
    
    basename = os.path.basename(filename)
    parts = basename.replace('.csv', '').split('_')
    
    # Find dataset name (usually after "Annotation_")
    if 'deptweet' in basename.lower():
        dataset = 'deptweet'
    elif 'irony' in basename.lower():
        dataset = 'irony'
    else:
        dataset = 'unknown'
    
    # Find language
    if 'bengali' in basename.lower():
        language = 'bengali'
    elif 'greek' in basename.lower():
        language = 'greek'
    else:
        language = 'unknown'
    
    # Find model name
    model_keywords = ['gpt', 'claude', 'deepseek', 'llama', 'gtrans', 'gtranslate', 'nllb']
    model = 'unknown'
    for keyword in model_keywords:
        if keyword in basename.lower():
            if 'gpt' in basename.lower():
                if '4.1' in basename.lower() and 'mini' in basename.lower():
                    model = 'gpt_4.1_mini'
                elif '5' in basename.lower():
                    model = 'gpt_5'
                else:
                    model = 'gpt'
            elif 'claude' in basename.lower():
                if 'sonnet' in basename.lower():
                    model = 'claude_sonnet'
                elif 'haiku' in basename.lower():
                    model = 'claude_haiku'
                else:
                    model = 'claude'
            elif 'deepseek' in basename.lower():
                model = 'deepseek'
            elif 'llama' in basename.lower():
                model = 'llama'
            elif 'gtrans' in basename.lower() or 'gtranslate' in basename.lower():
                model = 'gtranslate'
            elif 'nllb' in basename.lower():
                model = 'nllb'
            break
    
    return dataset, language, model

def generate_output_filename(dataset, language, model):
    """Generate output filename based on dataset and model"""
    # Clean up model names for better readability
    model_clean = model.replace('_', '-').replace('gtranslate', 'gt')
    
    # Format: dataset_lang_model_eval.txt
    return f"{dataset}_{language}_{model_clean}_eval.txt"

def get_dataset_file(dataset):
    """Get the appropriate dataset file based on dataset type"""
    if dataset == 'deptweet':
        return os.path.join(DATA_DIR, "deptweet_main.csv")
    elif dataset == 'irony':
        return os.path.join(DATA_DIR, "irony_main.csv")
    else:
        return os.path.join(DATA_DIR, "deptweet_main.csv")  # default fallback

def get_class_mapping(dataset):
    """Get class mapping based on dataset type"""
    if dataset == 'deptweet':
        return {0: 'non-depressed', 1: 'mild', 2: 'moderate', 3: 'severe'}
    elif dataset == 'irony':
        return {0: 'non-ironic', 1: 'ironic-by-clash', 2: 'situational-irony', 3: 'other-irony'}
    else:
        return {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3'}

def get_dataset_display_name(dataset):
    """Get human-readable dataset name"""
    if dataset == 'deptweet':
        return 'Deptweet'
    elif dataset == 'irony':
        return 'Irony Detection'
    else:
        return dataset.title()

def get_model_display_name(model):
    """Get human-readable model name"""
    if model == 'gpt_4.1_mini':
        return 'GPT 4.1 Mini'
    elif model == 'claude':
        return 'Claude'
    elif model == 'deepseek':
        return 'DeepSeek'
    elif model == 'llama':
        return 'Llama'
    elif model == 'gtranslate':
        return 'Gtranslate'
    elif model == 'nllb':
        return 'NLLB'
    else:
        return model.replace('_', ' ').title()

def clean_labels(df):
    """Clean labels to extract only numeric values"""
    
    # Function to extract numeric part from labels
    def extract_numeric(label):
        if pd.isna(label):
            return label
        
        # If already a number, return as is
        if isinstance(label, (int, float)) and not pd.isna(label):
            return label
        
        # Convert to string and extract first number
        label_str = str(label)
        match = re.search(r'(\d+)', label_str)
        if match:
            return int(match.group(1))
        else:
            return label
    
    # Clean literal labels
    if 'gpt_label_literal' in df.columns:
        df['gpt_label_literal'] = df['gpt_label_literal'].apply(extract_numeric)
    if 'deepseek_label_literal' in df.columns:
        df['deepseek_label_literal'] = df['deepseek_label_literal'].apply(extract_numeric)
    if 'claude_label_literal' in df.columns:
        df['claude_label_literal'] = df['claude_label_literal'].apply(extract_numeric)
    if 'literal_machine_label' in df.columns:
        df['literal_machine_label'] = df['literal_machine_label'].apply(extract_numeric)
    
    # Clean cultural labels
    if 'gpt_label_cultural' in df.columns:
        df['gpt_label_cultural'] = df['gpt_label_cultural'].apply(extract_numeric)
    if 'deepseek_label_cultural' in df.columns:
        df['deepseek_label_cultural'] = df['deepseek_label_cultural'].apply(extract_numeric)
    if 'claude_label_cultural' in df.columns:
        df['claude_label_cultural'] = df['claude_label_cultural'].apply(extract_numeric)
    if 'cultural_machine_label' in df.columns:
        df['cultural_machine_label'] = df['cultural_machine_label'].apply(extract_numeric)
    
    return df

def load_data(annotation_file):
    """Load annotation results and main dataset"""
    print("Loading data...")
    
    # Load annotation results with error handling for encoding
    try:
        annotations_df = pd.read_csv(annotation_file)
    except UnicodeDecodeError:
        # Try different encodings for problematic files
        try:
            annotations_df = pd.read_csv(annotation_file, encoding='latin-1')
        except:
            annotations_df = pd.read_csv(annotation_file, encoding='cp1252')
    
    # Clean labels to extract numeric values
    annotations_df = clean_labels(annotations_df)
    
    # Extract dataset info to determine which ground truth file to use
    dataset_info = extract_dataset_info(annotation_file)
    dataset_type = dataset_info[0]
    
    # Load appropriate main dataset
    main_file = get_dataset_file(dataset_type)
    main_df = pd.read_csv(main_file)
    
    # Merge by row position (index) to avoid duplicate issues
    df = annotations_df.copy()
    
    # Ensure lengths match between annotation file and main dataset
    annotation_length = len(df)
    main_length = len(main_df)
    
    print(f"Annotation file length: {annotation_length}")
    print(f"Main dataset length: {main_length}")
    
    if dataset_type == 'deptweet':
        if annotation_length <= main_length:
            # Use the annotation file length
            df['target'] = main_df['target'].values[:annotation_length]
            df['label'] = main_df['label'].values[:annotation_length]
        else:
            # Truncate annotation file to match main dataset
            df = df.head(main_length).copy()
            df['target'] = main_df['target'].values
            df['label'] = main_df['label'].values
    elif dataset_type == 'irony':
        if annotation_length <= main_length:
            # Use the annotation file length
            df['target'] = main_df['label'].values[:annotation_length]  # irony uses 'label' column for numeric values
            df['label'] = main_df['class'].values[:annotation_length]   # irony uses 'class' column for text labels
        else:
            # Truncate annotation file to match main dataset
            df = df.head(main_length).copy()
            df['target'] = main_df['label'].values
            df['label'] = main_df['class'].values
    
    print(f"Dataset type: {dataset_type}")
    print(f"Final samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    return df, dataset_type

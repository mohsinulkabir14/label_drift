import pandas as pd

def calculate_refusal_stats(df):
    """Calculate translation refusal statistics"""
    # Check if cultural translation column exists
    has_cultural = 'cultural_translation' in df.columns
    
    if has_cultural:
        # Both literal and cultural translations
        literal_refused = df['literal_translation'].isna()
        cultural_refused = df['cultural_translation'].isna()
        either_refused = literal_refused | cultural_refused
        both_refused = literal_refused & cultural_refused
        
        refusal_stats = {
            'total_samples': len(df),
            'literal_refused_count': literal_refused.sum(),
            'cultural_refused_count': cultural_refused.sum(),
            'both_refused_count': both_refused.sum(),
            'either_refused_count': either_refused.sum()
        }
    else:
        # Only literal translations (gtranslate files)
        literal_refused = df['literal_translation'].isna()
        
        refusal_stats = {
            'total_samples': len(df),
            'literal_refused_count': literal_refused.sum(),
            'cultural_refused_count': 0,
            'both_refused_count': 0,
            'either_refused_count': literal_refused.sum()
        }
    
    return literal_refused, cultural_refused if has_cultural else pd.Series([False] * len(df)), refusal_stats

def per_class_refusal_stats(df, literal_refused, cultural_refused, class_mapping):
    """Calculate per-class refusal statistics"""
    class_stats = []
    
    # Check if cultural translation column exists
    has_cultural = 'cultural_translation' in df.columns
    
    for target_val, label_name in class_mapping.items():
        class_mask = df['target'] == target_val
        class_total = class_mask.sum()
        
        if class_total > 0:
            literal_class_refused = (literal_refused & class_mask).sum()
            
            if has_cultural:
                cultural_class_refused = (cultural_refused & class_mask).sum()
                both_class_refused = (literal_refused & cultural_refused & class_mask).sum()
                either_class_refused = ((literal_refused | cultural_refused) & class_mask).sum()
            else:
                cultural_class_refused = 0
                both_class_refused = 0
                either_class_refused = literal_class_refused
            
            class_stats.append({
                'class': label_name,
                'target': target_val,
                'total': class_total,
                'literal_refused': literal_class_refused,
                'cultural_refused': cultural_class_refused,
                'both_refused': both_class_refused,
                'either_refused': either_class_refused
            })
    
    return class_stats

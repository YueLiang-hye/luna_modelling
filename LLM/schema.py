from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

def build_feature_schema(
    version: str,
    TX_features: List[str],     # time-varying columns from panel table
    T0_features: List[str],    # time-invariant columns from cross-sectional table
    computed_features: List[str],  # computed features from panel table
    cat_features: Optional[List[str]] = None,  # categorical features across both sets
    overrides: Optional[Dict[str, Dict]] = None,  # per-feature metadata overrides
    model_meta: Optional[Dict] = None           # catboost params, metrics etc.
) -> Dict:
    cat_features = cat_features or []
    overrides = overrides or {}
    model_meta = model_meta or {}

    features = {}

    # defaults for panel features
    for f in TX_features:
        meta = {
            "type": "numeric",            # you can change per feature in overrides
            "source": "panel",
            "temporal": True,
            "ask_frequency": "biweekly",  # every 2 weeks
            "computed": False,
            "reverify_days": 14,          # the agent may re-ask if older than this
            "description": "time-varying feature collected in each two-week session"
        }
        if f in cat_features:
            meta["type"] = "category"
        if f in overrides:
            meta.update(overrides[f])
        features[f] = meta

    # defaults for static features
    for f in T0_features:
        meta = {
            "type": "numeric",
            "source": "static",
            "temporal": False,
            "ask_frequency": "once",
            "computed": False,
            "reverify_days": None,        # only once; can override to annual check
            "description": "background; big five style feature"
        }
        if f in cat_features:
            meta["type"] = "category"
        if f in overrides:
            meta.update(overrides[f])
        features[f] = meta
    
    # computed features
    for f in computed_features:
        meta = {
            "type": "numeric",
            "source": "panel",
            "temporal": True,
            "ask_frequency": "never",    # never ask user
            "computed": True,
            "reverify_days": None,
            "description": "Derived feature; do not ask user."
        }
        if f in overrides:
            meta.update(overrides[f])
        features[f] = meta

    schema = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "target": "next_event",
        "model_type": "catboost_dropout_predictor",
        "features": features,
        "cat_features": cat_features,
        "expected_input_shape": [1, len(features)],
        "model_meta": model_meta 
    }
    return schema


T0_data = pd.read_csv("data/T0/merged_T0.csv", nrows=1)
T0_data.columns.tolist()
## drop studentID
T0_features = [c for c in T0_data.columns if c != 'studentID']
T1_data = pd.read_csv('data/TX/tx_long_preprocessed.csv')
raw_TX_features = [c for c in T1_data.columns if c not in ['studentID', 'meas', 'event', 'next_event', 'meas_prev', 'drpopout']]

computed_features = [c for c in raw_TX_features if (
    ('_lag' in c) or                 
    ('_roll_mean_' in c) or            
    c in ['weeks_since_prev', 'weeks_since_first', 'obs_count']  
)]

TX_scales = pd.read_csv('data/TX/tx_scale.csv')
T0_scales = pd.read_csv('data/T0/T0_scales.csv')
TX_features = [c for c in TX_scales['Columns'] if c not in ['studentID', 'meas', 'event', 'Summe']]
prompt_TX = {
    row['Columns']: row['prompt_in_questionnaire'] 
    for _, row in TX_scales.iterrows() 
    if row['Columns'] in TX_features
}

range_TX = {
    row['Columns']: row['range']
    for _, row in TX_scales.iterrows() 
    if row['Columns'] in TX_features
}

general_name_T0 = {
    row['Columns']: row['general_name']
    for _, row in T0_scales.iterrows() 
    if row['Columns'] in T0_features
}

range_T0 = {
    row['Columns']: row['range']
    for _, row in T0_scales.iterrows()
    if row['Columns'] in T0_features
}

# overrides (German)
conversation_overrides = {}
for feature in TX_features:
    if feature in prompt_TX:
        range_info = range_TX.get(feature, "1 to 4") 
        if "to" in str(range_info):
            parts = str(range_info).strip().split(" to ")
            if len(parts) == 2:
                min_val = int(parts[0])
                max_val = int(parts[1])
                validation_range = [min_val, max_val]
            else:
                validation_range = [1, 7]  
        else:
            validation_range = [1, 7]  # fallback

        qtext = prompt_TX[feature]
        min_val, max_val = validation_range[0], validation_range[1]
        conversation_overrides[feature] = {
            "description": f"Selbsteinschätzung (Skala {min_val}-{max_val}) zu '{qtext}' in den letzten zwei Wochen.",
            "prompt_template_de": f"In den letzten zwei Wochen, wie stark stimmen Sie der Aussage zu: '{qtext}' (Skala {min_val}–{max_val})?",
            "prompt_template_en": f"In the past two weeks, how strongly do you agree with the statement: \"{qtext}\" ({min_val}–{max_val})?",
            "validation_range": validation_range,
            "response_format_de": f"Bitte geben Sie eine Zahl zwischen {min_val} und {max_val} an",
            "response_format_en": f"Please provide a number between {min_val} and {max_val}",
            "scale_type": "likert",
            "original_prompt": qtext,
            "range_format": range_info
        }

# T0 features using general_name_T0 and range_T0
for feature in T0_features:
    if feature not in conversation_overrides:
        # Get general name and range info
        general_name = general_name_T0.get(feature, feature)
        range_info = range_T0.get(feature, "")
        
        # Handle range parsing
        if pd.isna(range_info) or str(range_info).strip() == "" or str(range_info).lower() == "nan":
            # Missing range - ask directly
            conversation_overrides[feature] = {
                "description": f"Bitte geben Sie Ihre {general_name} Information an",
                "prompt_template_de": f"Bitte teilen Sie uns Ihre {general_name} mit",
                "prompt_template_en": f"Please provide your {general_name}",
                "scale_type": "open_response"
            }
        elif "to" in str(range_info):
            # Range format like "x to y" - handle like TX features
            try:
                parts = str(range_info).strip().split(" to ")
                if len(parts) == 2:
                    min_val = int(parts[0])
                    max_val = int(parts[1])
                    validation_range = [min_val, max_val]
                else:
                    validation_range = None
            except:
                validation_range = None
                
            if validation_range:
                min_val, max_val = validation_range[0], validation_range[1]
                conversation_overrides[feature] = {
                    "description": f"Selbsteinschätzung (Skala {min_val}-{max_val}) zu '{general_name}'.",
                    "prompt_template_de": f"Wie würden Sie Ihre {general_name} einschätzen? (Skala {min_val}–{max_val})",
                    "prompt_template_en": f"How would you rate your {general_name}? ({min_val}–{max_val})",
                    "validation_range": validation_range,
                    "response_format_de": f"Bitte geben Sie eine Zahl zwischen {min_val} und {max_val} an",
                    "response_format_en": f"Please provide a number between {min_val} and {max_val}",
                    "scale_type": "likert",
                    "range_format": range_info
                }
            else:
                # Fallback for invalid range format
                conversation_overrides[feature] = {
                    "description": f"Bitte geben Sie Ihre {general_name} Information an",
                    "prompt_template_de": f"Bitte teilen Sie uns Ihre {general_name} mit",
                    "prompt_template_en": f"Please provide your {general_name}",
                    "scale_type": "open_response"
                }
        else:
            # Other range formats - ask directly
            conversation_overrides[feature] = {
                "description": f"Bitte geben Sie Ihre {general_name} Information an (Format: {range_info})",
                "prompt_template_de": f"Bitte teilen Sie uns Ihre {general_name} mit (Format: {range_info})",
                "prompt_template_en": f"Please provide your {general_name} (Format: {range_info})",
                "scale_type": "formatted_response",
                "expected_format": str(range_info)
            }

print(f" generate {len(conversation_overrides)} features' conversation configurations ")
print(" TX features (German):")
for i, (k, v) in enumerate(list(conversation_overrides.items())[:3]):
    if k in TX_features:
        print(f"  {k}: {v.get('prompt_template_de', v.get('prompt_template', 'N/A'))}")

model_meta = {
    "catboost_params": {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 8},
    "validation_metrics": {
        "auc_mean": 0.759, "auc_std": 0.065,
        "f1_mean": 0.300, "f1_std": 0.046,
        "prauc_mean": 0.186, "prauc_std": 0.039
    }
}

feature_schema = build_feature_schema(
    version="v1.0",
    TX_features=TX_features,
    T0_features=T0_features,
    computed_features=computed_features,
    overrides=conversation_overrides,  
    model_meta=model_meta
)

# save the complete conversation schema for the agent
import json
import os
os.makedirs('models', exist_ok=True)

with open('models/conversation_schema.json', 'w', encoding='utf-8') as f:
    json.dump(feature_schema, f, indent=2, ensure_ascii=False)

print("\n Schema summary:")
print(f"  total features: {len(feature_schema['features'])}")
ask_features = [f for f, meta in feature_schema['features'].items() 
               if not meta['computed'] and meta['ask_frequency'] != 'never']
computed = [f for f, meta in feature_schema['features'].items() if meta['computed']]
print(f"  Agent : {len(ask_features)}")
print(f"  automatically computed features: {len(computed)}")
print(f"  save to: models/conversation_schema.json")

# show some examples of ask features (German version)
print(f"\n Agent will ask {len(ask_features)} features:")
for f in ask_features[:5]:
    meta = feature_schema['features'][f]
    prompt = meta.get('prompt_template_de', meta.get('prompt_template', meta['description']))
    print(f"  {f}: {prompt}")

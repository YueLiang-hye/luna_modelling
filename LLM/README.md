# Luna Modelling - Dropout Prediction with LLM Agent Integration

A comprehensive machine learning pipeline for predicting student dropout using CatBoost with conversational AI interface for data collection.

##  Project Overview

This project implements a production-ready dropout prediction system that combines:
- **Longitudinal Data Analysis**: Student×time panel data preprocessing
- **CatBoost Modeling**: Gradient boosting for binary dropout prediction 
- **LLM Agent Integration**: Conversational interface for student data collection
- **German Language Support**: Questionnaire-based prompts in German

##  Key Features

- **Temporal Feature Engineering**: Lag features, rolling windows, time deltas
- **Cross-sectional Integration**: Merge T0 (background) with TX (longitudinal) data
- **Production Pipeline**: Model versioning, schema generation, hot-reload capability
- **Imbalanced Data Handling**: Class weighting and threshold optimization
- **Conversational Schema**: Natural language prompts for student interviews

##  Project Structure

```
├── data/
│   ├── T0/                    # Cross-sectional (background) data
│   │   ├── merged_T0.csv      # Combined T0 features
│   │   └── *_scale.csv        # Feature metadata & prompts
│   └── TX/                    # Longitudinal (time-varying) data
│       ├── tx_long.csv        # Raw panel data
│       ├── tx_long_preprocessed.csv
│       └── tx_scale.csv       # TX feature metadata
├── models/                    # Production model artifacts
│   ├── conversation_schema.json
│   └── *.cbm                  # CatBoost model files (excluded)
├── preprocess.ipynb           # Data preprocessing pipeline
├── main.py                    # Production model interface
└── schema.py                  # Feature schema generation
```


##  Pipeline Overview

### Data Preprocessing (`preprocess.ipynb`)
1. **Missing Value Handling**: Remove features >50% missing
2. **Target Construction**: Create t→t+1 dropout prediction target
3. **Feature Engineering**:
   - Lag features (1,2,3 periods)
   - Rolling means (2,3,5 windows) 
   - Time deltas (weeks since previous/first measurement)
4. **T0 Integration**: Merge background data into panel format
5. **Leakage Prevention**: Strict temporal ordering in feature construction

### Model Training
- **Algorithm**: CatBoost (depth=5, learning_rate=0.05)
- **Cross-validation**: 5-fold stratified
- **Class Balancing**: Balanced class weights
- **Metrics**: AUC=0.759±0.065, F1=0.300±0.046

### LLM Agent Schema (`schema.py`)
- **German Prompts**: Natural questionnaire-based conversations
- **Range Validation**: Automatic parsing of "1 to X" response ranges
- **Feature Categorization**: TX (biweekly) vs T0 (once) vs computed (never ask)
- **Production Ready**: JSON schema for conversational AI integration

##  LLM Agent Integration

The system generates conversation templates for student data collection:

**TX Features (Biweekly Collection)**:
```json
{
  "prompt_template_de": "In den letzten zwei Wochen, wie stark stimmen Sie der Aussage zu: 'Ich mochte die Inhalte' (Skala 1–4)?",
  "validation_range": [1, 4],
  "ask_frequency": "biweekly"
}
```

**T0 Features (One-time Collection)**:
```json
{
  "prompt_template_de": "Wie würden Sie Ihre Abiturnote einschätzen? (Skala 1–4)",
  "validation_range": [1, 4], 
  "ask_frequency": "once"
}
```

##  Model Performance

| Metric | TX Only | TX + T0 |
|--------|---------|---------|
| AUC | 0.732±0.071 | 0.759±0.065 |
| F1 Score | 0.284±0.052 | 0.300±0.046 |
| PR-AUC | 0.171±0.043 | 0.186±0.039 |

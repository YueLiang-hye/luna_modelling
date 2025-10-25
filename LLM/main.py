import json
import joblib
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from catboost import CatBoostClassifier
from CatBoost import CatBoost_combine
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight


def save_production_model(
    model: CatBoostClassifier, 
    feature_cols: List[str],
    cat_features: List[str],
    metrics: Dict[str, float],
    version: Optional[str] = None,
    model_dir: str = "models"
) -> str:

    if version is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # save model file
    model_path = os.path.join(model_dir, f"cat_model_{version}.cbm")
    model.save_model(model_path)
    
    # save feature schema
    feature_schema = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "num_features": len(feature_cols),
        "target": "next_event",
        "model_type": "catboost_dropout_predictor",
        "validation_metrics": metrics,
        "expected_input_shape": (1, len(feature_cols)),
        "catboost_params": {
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 8,
            "random_strength": 1.0
        }
    }
    
    schema_path = os.path.join(model_dir, f"feature_schema_{version}.json")
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(feature_schema, f, indent=2, ensure_ascii=False)
    
    # 3) 保存 "current" 软链接 (用于热更新)
    current_model_path = os.path.join(model_dir, "cat_model_current.cbm")
    current_schema_path = os.path.join(model_dir, "feature_schema_current.json")
    
    # Windows-safe copy 
    import shutil
    shutil.copy2(model_path, current_model_path)
    shutil.copy2(schema_path, current_schema_path)
    
    print(f" Saved production model {version}")
    print(f"   Model: {model_path}")
    print(f"   Schema: {schema_path}")
    print(f"   Metrics: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}")
    
    return version


def load_production_model(
    model_dir: str = "models",
    version: str = "current"
) -> tuple[CatBoostClassifier, Dict]:
    """
    load production model + metadata
    
    Args:
        version: e.g., "v20241024_143022"
    
    Returns:
        (model, schema_dict)
    """
    model_path = os.path.join(model_dir, f"cat_model_{version}.cbm")
    schema_path = os.path.join(model_dir, f"feature_schema_{version}.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = CatBoostClassifier()
    model.load_model(model_path)
    
    # 加载 schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    print(f" oaded model {schema['version']} (trained {schema['timestamp']})")
    return model, schema


def predict_dropout(
    student_features: Dict[str, Any],
    model_dir: str = "models",
    return_explanation: bool = True
) -> Dict[str, Any]:
    """
    LLM Agent 调用的核心预测函数
    
    Args:
        student_features: 学生特征字典 (可能包含缺失值)
        return_explanation: 是否返回解释信息
    
    Returns:
        {
            "dropout_probability": float,
            "risk_level": str,
            "prediction": int,
            "confidence": float,
            "missing_features": List[str],
            "explanation": str (if requested)
        }
    """
    try:
        # 加载当前模型
        model, schema = load_production_model(model_dir)
        
        # 构造特征向量 (按 schema 顺序)
        feature_vector = []
        missing_features = []
        
        for col in schema["feature_cols"]:
            if col in student_features and student_features[col] is not None:
                feature_vector.append(student_features[col])
            else:
                # 使用默认值 (这里简化为 0，实际应该用训练时的 imputation 逻辑)
                feature_vector.append(0)
                missing_features.append(col)
        
        X = np.array([feature_vector])
        prob = model.predict_proba(X)[0, 1]  
        
        # for imbalanced data, adjust threshold
        pred = int(prob > 0.1)  
        
        # level of risk assessment
        if prob < 0.05:        
            risk_level = "extremely low risk"
        elif prob < 0.15:     
            risk_level = "low risk"
        elif prob < 0.30:     
            risk_level = "moderate risk"
        elif prob < 0.50:   
            risk_level = "high risk"
        else:                 
            risk_level = "extremely high risk"
        
        result = {
            "dropout_probability": float(prob),
            "risk_level": risk_level,
            "prediction": pred,
            "confidence": max(prob, 1-prob),  
            "missing_features": missing_features,
            "model_version": schema["version"]
        }
        
        if return_explanation:
            explanation = f"based on {len(schema['feature_cols'])} " \
                         f"features, the predicted dropout probability is {prob:.1%}, " \
            
            if prob >= 0.30:
                explanation += "suggest immediate intervention and close monitoring. "
            elif prob >= 0.15:
                explanation += "suggest enhanced attention and additional support. "
            elif prob >= 0.05:
                explanation += "suggest regular monitoring of academic performance. "
            else:
                explanation += "the student is at very low risk of dropout. "
                
            if missing_features:
                explanation += f" pay attention: {len(missing_features)} missing features may affect accuracy."
            result["explanation"] = explanation
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "dropout_probability": None,
            "risk_level": "cannot assess",
            "prediction": None
        }


def retrain_catboost(
    intake_buffer_path: str = "intake_buffer.parquet",
    base_data_path: str = "data/TX/tx_long_preprocessed_with_T0.csv",
    model_dir: str = "models"
) -> str:
    """
    定期重训练函数 (Nightly/Weekly)
    
    Args:
        intake_buffer_path: 新收集的对话数据
        base_data_path: 历史训练数据
        
    Returns:
        str: 新模型版本号
    """
    print("Starting model retraining...")
    
    # combine new intake data with base data
    base_data = pd.read_csv(base_data_path)
    
    if os.path.exists(intake_buffer_path):
        new_data = pd.read_parquet(intake_buffer_path)
        # 这里需要根据你的 intake 格式进行预处理
        # combined_data = preprocess_and_merge(base_data, new_data)
        combined_data = base_data  # 临时简化
        print(f"   Merged {len(base_data)} historical + {len(new_data)} new samples")
    else:
        combined_data = base_data
        print(f"   No new data found, using {len(base_data)} historical samples")
    
    # retrain CatBoost model
    feature_cols = [c for c in combined_data.columns 
                   if c not in ['studentID', 'meas', 'event', 'next_event', 'meas_prev', 'drpopout']]
    cat_features = [c for c in combined_data.select_dtypes(include=['object', 'category']).columns 
                   if c in feature_cols]
    
    X = combined_data[feature_cols]
    y = combined_data['next_event']
    weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model, metrics = CatBoost_combine(X, y, weights, kf)
    
    # save production model
    version = save_production_model(model, feature_cols, cat_features, metrics, model_dir=model_dir)
    
    print(f"Retraining complete: {version}")
    return version

def get_agent_tools():
    """
    返回 LLM Agent 可用的工具函数列表
    """
    tools = [
        {
            "name": "predict_dropout",
            "description": "预测学生下期退学概率并给出风险评估",
            "function": predict_dropout,
            "parameters": {
                "student_features": "Dict[str, Any] - 学生特征字典",
                "return_explanation": "bool - 是否返回详细解释 (默认 True)"
            }
        },
        {
            "name": "retrain_model",
            "description": "使用最新数据重新训练模型",
            "function": retrain_catboost,
            "parameters": {
                "intake_buffer_path": "str - 新收集数据路径",
                "base_data_path": "str - 历史数据路径"
            }
        }
    ]
    return tools


if __name__ == "__main__":

    print("Training and saving production model")

    data = pd.read_csv('data/TX/tx_long_preprocessed_with_t0.csv')
    feature_cols = [c for c in data.columns if c not in ['studentID', 'meas', 'event', 'next_event', 'meas_prev', 'drpopout']]
    X = data[feature_cols]
    y = data['next_event']
    weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model, metrics = CatBoost_combine(X, y, weights, kf)
    cat_features = [c for c in data.select_dtypes(include=['object', 'category']).columns if c in feature_cols]
    version = save_production_model(model, feature_cols, cat_features, metrics)
    
    # test loading and prediction
    print("\n Testing model loading and prediction...")
    test_features = {
        "age_centered": -2.5,
        "stress_lag1": 3.2,
        "gpa_roll_mean_3": 2.8,
        # ... 其他特征
    }
    
    result = predict_dropout(test_features)
    print(f"Prediction result: {result}")
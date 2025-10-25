from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, average_precision_score
from catboost import CatBoostClassifier, Pool
import numpy as np

def evaluate_model_cv(X, y, weights, kf):

    cv_auc_scores = []
    cv_f1_scores = []  
    cv_accuracy_scores = []
    cv_pr_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx])
        val_pool   = Pool(X.iloc[val_idx], y.iloc[val_idx])

        model = CatBoostClassifier(
            depth=5, learning_rate=0.05, iterations=500,
            l2_leaf_reg=8, class_weights=weights, random_strength=1.0, verbose=0
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        y_prob = model.predict_proba(val_pool)[:,1]
        y_valid = y.iloc[val_idx]
        prec, rec, thr = precision_recall_curve(y_valid, y_prob)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        best_t = thr[np.argmax(f1)]

        y_pred = (y_prob >= best_t).astype(int)
        auc = roc_auc_score(y_valid, y_prob)
        f1_val = f1_score(y_valid, y_pred)
        accuracy = accuracy_score(y_valid, y_pred)
        pr_auc = average_precision_score(y_valid, y_prob)
        
        cv_auc_scores.append(auc)
        cv_f1_scores.append(f1_val)
        cv_accuracy_scores.append(accuracy)
        cv_pr_auc_scores.append(pr_auc)
        
        print(f"   Fold {fold}: AUC={auc:.3f}, F1={f1_val:.3f}")
    
    metrics = {
        'auc': np.mean(cv_auc_scores),
        'auc_std': np.std(cv_auc_scores),
        'f1': np.mean(cv_f1_scores), 
        'f1_std': np.std(cv_f1_scores),
        'accuracy': np.mean(cv_accuracy_scores),
        'accuracy_std': np.std(cv_accuracy_scores),
        'pr_auc': np.mean(cv_pr_auc_scores),
        'pr_auc_std': np.std(cv_pr_auc_scores)
    }
    
    print(f"\n Cross-Validation Results:")
    print(f"   AUC: {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}")
    print(f"   F1: {metrics['f1']:.3f} ± {metrics['f1_std']:.3f}")
    print(f"   Accuracy: {metrics['accuracy']:.3f} ± {metrics['accuracy_std']:.3f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.3f} ± {metrics['pr_auc_std']:.3f}")
    
    return metrics


def train_production_model(X, y, weights):
    
    production_model = CatBoostClassifier(
        depth=5, learning_rate=0.05, iterations=500,
        l2_leaf_reg=8, class_weights=weights, random_strength=1.0, verbose=0
    )
    production_pool = Pool(X, y)
    production_model.fit(production_pool)
    return production_model


def CatBoost_combine(X, y, weights, kf):

    metrics = evaluate_model_cv(X, y, weights, kf)
    production_model = train_production_model(X, y, weights)
    
    return production_model, metrics

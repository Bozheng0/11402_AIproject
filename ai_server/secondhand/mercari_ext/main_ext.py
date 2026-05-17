import sys
import os
import joblib
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from mercari.main_helpers import merge_predictions
from sklearn.linear_model import Lasso
from mercari.mercari_io import load_train_validation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONWARNINGS"] = "ignore"

original_repo_path = os.path.abspath("../mercari-solution")
sys.path.append(original_repo_path)

from mercari.main_tf import define_models_1, define_models_2, define_models_3
from mercari.main_helpers import fit_transform_vectorizer, fit_models, predict_models, rmsle
from mercari.config import logger
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger().handlers.clear()
logging.disable(logging.CRITICAL)


def calculate_all_metrics(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    
    metrics = {
        "RMSLE": rmsle(y_pred, y_true),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics

def run_round_ext(round_num, arg_map):
    logger.info(f"==== Starting Extended Round {round_num} ====")
    models, vectorizer = arg_map[round_num]
    
    X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    
    fitted_models = fit_models(X_tr, y_tr, models)
    
    y_va_preds = predict_models(X_va, fitted_models)
    
    avg_preds = y_va_preds.mean(axis=1)
    metrics = calculate_all_metrics(y_va, avg_preds)
    
    print(f"\n[Round {round_num} Validation Metrics]")
    for m, val in metrics.items():
        print(f"{m}: {val:.4f}")
        
    save_dir = "models_exported"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    with open(f"{save_dir}/round_{round_num}_vectorizer.pkl", 'wb') as f:
        pickle.dump(fitted_vectorizer, f)
    
    all_model_data = []
    for m in fitted_models:
        with m._graph.as_default():
            weights = m._session.run({v.name: v for v in tf.trainable_variables()})
            exclude_keys = ['_session', '_graph', '_xs', '_ys', '_output', '_loss', '_train_op', '_lr', '_w_1', '_hidden']
            state = {k: v for k, v in m.__dict__.items() if k not in exclude_keys}
            
            all_model_data.append({
                'weights': weights,
                'state': state
            })
            
    with open(f"{save_dir}/round_{round_num}_weights.pkl", 'wb') as f:
        pickle.dump(all_model_data, f)

    meta_data = {
        "metrics": metrics,
        "va_preds": y_va_preds,
    }
    with open(f"{save_dir}/round_{round_num}_meta.pkl", 'wb') as f:
        pickle.dump(meta_data, f)
        
    if not os.path.exists(f"{save_dir}/y_va.pkl"):
        with open(f"{save_dir}/y_va.pkl", 'wb') as f:
            pickle.dump(y_va, f)
        
    print(f"Model components saved to {save_dir} (Round {round_num})\n")

def run_merge_ext():
    logger.info("==== Starting Extended Merge (Stacking) ====")
    va_preds_list = []  
    save_dir = "models_exported"
    
    for r in [1, 2, 3]:
        path = f"{save_dir}/round_{r}_meta.pkl"
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skip.")
            continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        va_preds_list.append(data["va_preds"])
        
    if not va_preds_list:
        print("No prediction files found to merge.")
        return

    y_va_path = f"{save_dir}/y_va.pkl"
    if not os.path.exists(y_va_path):
        _, df_va = load_train_validation()
        y_va = df_va.price.values
        with open(y_va_path, 'wb') as f:
            pickle.dump(y_va, f)
    else:
        with open(y_va_path, 'rb') as f:
            y_va = pickle.load(f)

    X_va_stack = np.hstack(va_preds_list)
    print("Fitting Lasso stacker...")
    est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                positive=True, random_state=9999, selection='random')
    
    va_preds_merged, _ = merge_predictions(X_tr=X_va_stack, y_tr=y_va, est=est)
    
    with open(f"{save_dir}/stacker.pkl", 'wb') as f:
        pickle.dump(est, f)
    
    metrics = calculate_all_metrics(y_va, va_preds_merged)
    
    print("\n[Final Merged Stacking Metrics (Lasso)]")
    for m, val in metrics.items():
        print(f"{m}: {val:.4f}")
    print(f"Stacker model saved to {save_dir}/stacker.pkl")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_ext.py [1|2|3|all|merge]")
        sys.exit(1)

    arg_map = {
        1: define_models_1(n_jobs=4, seed=1),
        2: define_models_2(n_jobs=4, seed=2),
        3: define_models_3(n_jobs=4, seed=3),
    }

    action = sys.argv[1]
    if action == "all":
        for r in [1, 2, 3]:
            run_round_ext(r, arg_map)
        run_merge_ext()
    elif action in ["1", "2", "3"]:
        run_round_ext(int(action), arg_map)
    elif action == "merge":
        run_merge_ext()
    else:
        print("Invalid action. Use 1, 2, 3, all, or merge.")

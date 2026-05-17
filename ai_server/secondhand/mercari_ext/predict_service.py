import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

current_dir = os.path.dirname(os.path.abspath(__file__))
original_repo_path = os.path.abspath(os.path.join(current_dir, "../mercari-solution"))
sys.path.append(original_repo_path)

from mercari.main_tf import define_models_1, define_models_2, define_models_3
from mercari.main_helpers import predict_models, merge_predictions
from sklearn.linear_model import Lasso
from mercari.tf_sparse import SparseMatrix

class MercariPredictor:
    def __init__(self, models_dir="models_exported"):
        self.models_dir = os.path.join(current_dir, models_dir)
        self.rounds_data = {}
        self.stacker = None
        
    def load_round(self, round_num):
        
        print(f"Loading components for Round {round_num}...")
        
        vec_path = os.path.join(self.models_dir, f"round_{round_num}_vectorizer.pkl")
        with open(vec_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        weights_path = os.path.join(self.models_dir, f"round_{round_num}_weights.pkl")
        with open(weights_path, 'rb') as f:
            all_model_data = pickle.load(f)
            
        if round_num == 1:
            models_templates, _ = define_models_1(n_jobs=1, seed=1)
        elif round_num == 2:
            models_templates, _ = define_models_2(n_jobs=1, seed=1)
        elif round_num == 3:
            models_templates, _ = define_models_3(n_jobs=1, seed=1)
        else:
            raise ValueError(f"Unsupported round: {round_num}")
            
        self.rounds_data[round_num] = {
            'vectorizer': vectorizer,
            'templates': models_templates,
            'data': all_model_data
        }
        print(f"Round {round_num} metadata loaded.")

    def _initialize_model_tf(self, model, weights, n_features):
        model._graph = tf.Graph()
        config = tf.ConfigProto(device_count={'GPU': 0})
        model._session = tf.Session(graph=model._graph, config=config)
        
        with model._graph.as_default():
            model._xs = SparseMatrix()
            model._lr = tf.placeholder(tf.float32, shape=[])
            model._build_model(n_features)
            model._session.run(tf.global_variables_initializer())
            
            for var in tf.trainable_variables():
                if var.name in weights:
                    val = weights[var.name]
                    var.load(val, model._session)
        model.is_fitted = True

    def _get_stacker(self):
        if self.stacker is not None:
            return self.stacker
            
        stacker_path = os.path.join(self.models_dir, "stacker.pkl")
        if os.path.exists(stacker_path):
            print("Loading pre-trained Lasso stacker...")
            with open(stacker_path, 'rb') as f:
                self.stacker = pickle.load(f)
            return self.stacker
            
        print("Pre-trained stacker not found. Attempting to fit on the fly...")
        va_preds_list = []
        y_va_path = os.path.join(self.models_dir, "y_va.pkl")
        
        if not os.path.exists(y_va_path):
            print("Warning: y_va.pkl not found. Cannot fit Lasso stacker.")
            return None
            
        with open(y_va_path, 'rb') as f:
            y_va = pickle.load(f)
            
        for r in sorted(self.rounds_data.keys()):
            meta_path = os.path.join(self.models_dir, f"round_{r}_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                va_preds_list.append(meta["va_preds"])
            else:
                print(f"Warning: meta file for round {r} not found. Stacker might be inaccurate.")
        
        if not va_preds_list:
            return None
            
        X_va_stack = np.hstack(va_preds_list)
        self.stacker = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                            positive=True, random_state=9999, selection='random')

        merge_predictions(X_tr=X_va_stack, y_tr=y_va, est=self.stacker)
        print("Lasso stacker fitted successfully.")
        return self.stacker

    def predict(self, df, single_round=None):

        bundle_preds = {}

        rounds = [single_round] if single_round else sorted(self.rounds_data.keys())

        for round_num in rounds:
            bundle = self.rounds_data[round_num]
            print(f"Executing Prediction for Round {round_num}...")

            vectorizer = bundle['vectorizer']
            X = vectorizer.transform(df)
            n_features = X.shape[1]

            if 'fitted_models' not in bundle:
                bundle['fitted_models'] = []
                for i, m_data in enumerate(bundle['data']):
                    m_template = bundle['templates'][i]

                    for k, v in m_data['state'].items():
                        setattr(m_template, k, v)

                    self._initialize_model_tf(m_template, m_data['weights'], n_features)
                    bundle['fitted_models'].append(m_template)

            y_preds = predict_models(X, bundle['fitted_models'])
            bundle_preds[round_num] = y_preds


        if single_round is not None:
            print(f"Single model mode (Round {single_round})")
            return bundle_preds[single_round].mean(axis=1)


        all_model_preds = [bundle_preds[r] for r in sorted(bundle_preds.keys())]

        X_te_stack = np.hstack(all_model_preds)

        stacker = self._get_stacker()

        if stacker is not None:
            # Check if feature count matches
            if X_te_stack.shape[1] == stacker.coef_.shape[0]:
                print("Using Lasso Stacking...")
                return np.expm1(stacker.predict(np.log1p(X_te_stack)))
            else:
                print(f"Warning: Stacker expects {stacker.coef_.shape[0]} features, but got {X_te_stack.shape[1]}.")

        print("Falling back to mean ensemble...")
        return X_te_stack.mean(axis=1)

def main():
    parser = argparse.ArgumentParser(description="Mercari Price Predictor Service")
    parser.add_argument("--input", type=str, required=True, help="Path to input TSV file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to output CSV file")
    parser.add_argument("--rounds", type=str, default="1", help="Comma separated rounds to use (e.g., 1,2,3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Reading input data: {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    
    predictor = MercariPredictor()
    round_list = [int(r) for r in args.rounds.split(',')]
    
    for r in round_list:
        try:
            predictor.load_round(r)
        except FileNotFoundError:
            print(f"Warning: Model files for Round {r} not found. Skipping.")

    if not predictor.rounds_data:
        print("Error: No models loaded. Please train models first using main_ext.py.")
        return

    print("Starting prediction...")
    prices = predictor.predict(df)
    
    output_df = pd.DataFrame({
        'test_id': df.get('test_id', range(len(prices))),
        'price': prices
    })
    output_df.to_csv(args.output, index=False)
    print(f"Done! Predictions saved to {args.output}")

if __name__ == "__main__":
    main()

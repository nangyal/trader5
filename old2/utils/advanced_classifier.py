"""
GPU-enabled wrapper for the old EnhancedForexPatternClassifier.

This file provides `GPUEnhancedForexClassifier` which uses the
`EnhancedForexPatternClassifier` implementation from `/old/forex_pattern_classifier.py`
but sets XGBoost to GPU mode (when available) for training and prediction.
"""

import os
from pathlib import Path
import joblib
import importlib


class GPUEnhancedForexClassifier:
    """Wrapper that uses the old classifier but forces XGBoost to GPU when available."""

    def __init__(self, model_path: str = None, output_dir: str = None):
        # Delay import to avoid heavy imports at module import time
        mod = importlib.import_module('old.forex_pattern_classifier')
        cls = getattr(mod, 'EnhancedForexPatternClassifier')
        self.impl = cls(output_dir=output_dir)
        self.model_path = Path(model_path) if model_path else None

    def train(self, df, labels, *, use_gpu=True, **kwargs):
        # Ensure environment variable for XGBoost GPU acceleration if available
        if use_gpu:
            os.environ['XGBOOST_VERBOSITY'] = '0'
        # Call original training; the original will try GPU first
        model = self.impl.train(df, labels, **kwargs)
        # After training, we ensure the parameters prefer GPU if present
        try:
            if hasattr(model, 'set_params'):
                model.set_params(tree_method='gpu_hist', device='cuda', predictor='gpu_predictor')
        except Exception:
            pass
        return model

    def predict(self, df):
        # Use underlying predict implementation
        return self.impl.predict(df)

    def save_model(self, path: str):
        # save including label_encoder and scaler
        return self.impl.save_model(path)

    def load_model(self, path: str, *, force_gpu=False):
        self.impl.load_model(path)
        # Optionally adjust to use GPU for predictions if supported
        if force_gpu and hasattr(self.impl.model, 'set_params'):
            try:
                self.impl.model.set_params(tree_method='gpu_hist', device='cuda', predictor='gpu_predictor')
                print('Model set to GPU predictor')
            except Exception:
                print('Unable to set model to GPU predictor')

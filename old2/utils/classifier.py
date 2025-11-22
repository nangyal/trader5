import joblib
import numpy as np
import pandas as pd


class SimpleClassifier:
    """Minimal, self-contained classifier wrapper.

    - Can load a saved model saved via the older workflow (joblib with keys 'model', 'label_encoder', 'scaler', 'feature_names').
    - If feature columns aren't present in the passed DataFrame it will return 'no_pattern' labels.
    This avoids importing the Monolithic pattern code from `/old` while still allowing the model file to be used if present.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None

    def load_model(self, filepath, **kwargs):
        # Accept optional kwargs (e.g., force_gpu) to be compatible with GPU wrapper
        return self._load_model(filepath)

    def _load_model(self, filepath):
        try:
            data = joblib.load(str(filepath))
            self.model = data.get('model')
            self.label_encoder = data.get('label_encoder')
            self.scaler = data.get('scaler')
            self.feature_names = data.get('feature_names')
            print('SimpleClassifier: model loaded from', filepath)
            return True
        except Exception as e:
            print('SimpleClassifier could not load model:', e)
            return False


    def predict(self, df: pd.DataFrame):
        # No model => return no_pattern
        if self.model is None or self.feature_names is None:
            labels = ['no_pattern'] * len(df)
            probs = np.zeros((len(df), 1))
            return labels, probs

        # Check if required features are present in df
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            print('SimpleClassifier: missing features (will return no_pattern):', missing[:5])
            labels = ['no_pattern'] * len(df)
            probs = np.zeros((len(df), 1))
            return labels, probs

        # Prepare input
        X = df[self.feature_names].astype(float).fillna(0.0)
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                X = X.values

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        # decode labels if needed
        try:
            labels = self.label_encoder.inverse_transform(preds)
        except Exception:
            labels = preds.astype(str)

        return labels, probs

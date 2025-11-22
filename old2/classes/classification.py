# Enhanced Forex Pattern Classifier
# Copied from old/forex_pattern_classifier.py with GPU optimization
import os
os.environ['XGBOOST_VERBOSITY'] = '0'

import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import warnings
import logging
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from pathlib import Path

logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class EnhancedFeatureExtractor:
    """Enhanced feature extraction with professional technical indicators"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features = {}
        
    def add_advanced_price_features(self):
        """Advanced price action features with professional metrics"""
        df = self.df
        
        # Basic price features with safe calculations
        self.features['price_range'] = df['high'] - df['low']
        self.features['body_size'] = abs(df['close'] - df['open'])
        self.features['body_to_range_ratio'] = self._safe_divide(
            self.features['body_size'], self.features['price_range'], 0.5
        )
        
        # Professional candlestick patterns
        self.features['upper_shadow_ratio'] = self._safe_divide(
            df['high'] - df[['open', 'close']].max(axis=1), self.features['price_range'], 0
        )
        self.features['lower_shadow_ratio'] = self._safe_divide(
            df[['open', 'close']].min(axis=1) - df['low'], self.features['price_range'], 0
        )
        
        # Price position features
        self.features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        return self
        
    def add_professional_technical_indicators(self):
        """Professional-grade technical indicators"""
        df = self.df
        close, high, low = df['close'].values, df['high'].values, df['low'].values
        
        # Trend indicators
        self.features['sma_10'] = talib.SMA(close, timeperiod=10)
        self.features['sma_20'] = talib.SMA(close, timeperiod=20)
        self.features['sma_50'] = talib.SMA(close, timeperiod=50)
        self.features['ema_12'] = talib.EMA(close, timeperiod=12)
        self.features['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum indicators
        self.features['rsi'] = talib.RSI(close, timeperiod=14)
        self.features['stoch_k'], self.features['stoch_d'] = talib.STOCH(high, low, close)
        self.features['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Volume-based indicators (if volume available)
        if 'volume' in df.columns:
            self.features['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            self.features['volume_ratio'] = self._safe_divide(
                df['volume'], self.features['volume_sma'], 1.0
            )
            self.features['obv'] = talib.OBV(close, df['volume'])
        
        # Volatility indicators
        self.features['atr'] = talib.ATR(high, low, close, timeperiod=14)
        self.features['natr'] = talib.NATR(high, low, close, timeperiod=14)
        self.features['volatility'] = talib.STDDEV(close, timeperiod=20, nbdev=1)
        
        # Advanced indicators
        macd, macd_signal, macd_hist = talib.MACD(close)
        self.features['macd'] = macd
        self.features['macd_signal'] = macd_signal
        self.features['macd_hist'] = macd_hist
        
        # Bollinger Bands with position
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.features['bb_upper'] = bb_upper
        self.features['bb_lower'] = bb_lower
        self.features['bb_position'] = self._safe_divide(
            close - bb_lower, bb_upper - bb_lower, 0.5
        )
        self.features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        return self
        
    def add_pattern_specific_features(self, window=30):
        """Pattern-specific features based on professional definitions"""
        df = self.df
        close, high, low = df['close'].values, df['high'].values, df['low'].values
        
        # Support and resistance features
        self.features['resistance_level'] = talib.MAX(high, timeperiod=window)
        self.features['support_level'] = talib.MIN(low, timeperiod=window)
        self.features['price_vs_resistance'] = self._safe_divide(
            self.features['resistance_level'] - close, self.features['resistance_level'], 0
        )
        self.features['price_vs_support'] = self._safe_divide(
            close - self.features['support_level'], close, 0
        )
        
        # Trend strength and direction
        self.features['trend_strength'] = talib.LINEARREG_SLOPE(close, timeperiod=window)
        self.features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Pattern probability features (compatibility with old model)
        self.features['triangle_probability'] = np.zeros(len(close))
        self.features['reversal_probability'] = np.zeros(len(close))
        self.features['consolidation_strength'] = np.zeros(len(close))
        
        return self
        
    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division avoiding division by zero"""
        return np.where(denominator == 0, default, numerator / denominator)
        
    def get_features_df(self):
        """Get final features DataFrame with robust data cleaning"""
        features_df = pd.DataFrame(self.features)
        
        # Robust data cleaning
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values strategically
        for col in features_df.columns:
            if features_df[col].isna().any():
                features_df[col] = features_df[col].ffill().bfill()
        
        # Final cleanup of any remaining NaN
        features_df = features_df.fillna(0)
        
        # Remove any remaining infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Optimize data types for GPU processing
        for col in features_df.select_dtypes(include=['float64']).columns:
            features_df[col] = features_df[col].astype(np.float32)
            
        return features_df


class EnhancedForexPatternClassifier:
    """Enhanced classifier with GPU optimization and professional features"""
    
    def __init__(self, output_dir=None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.output_dir = output_dir if output_dir else '.'
        
    def prepare_data(self, df: pd.DataFrame, pattern_labels: pd.Series):
        """Prepare data with enhanced feature extraction"""
        print("Extracting advanced features...")
        
        # Enhanced feature extraction
        extractor = (EnhancedFeatureExtractor(df)
                    .add_advanced_price_features()
                    .add_professional_technical_indicators()
                    .add_pattern_specific_features())
        
        features_df = extractor.get_features_df()
        self.feature_names = features_df.columns.tolist()
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(pattern_labels)
        
        return features_df, encoded_labels
        
    def train(self, df: pd.DataFrame, pattern_labels: pd.Series, test_size=0.2, optimize_hyperparams=True):
        """Train with GPU optimization and hyperparameter tuning"""
        # Prepare data
        X, y = self.prepare_data(df, pattern_labels)
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            best_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        # GPU-optimized XGBoost model (XGBoost 2.0+ uses device='cuda')
        try:
            self.model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='mlogloss',
                device='cuda:0'  # XGBoost 2.0+: auto-selects GPU algorithm
            )
            print("✅ Training model with GPU acceleration (CUDA)...")
        except Exception as e:
            print(f"⚠️  GPU not available ({e}), using CPU...")
            self.model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric='mlogloss',
                device='cpu'
            )
        
        print("Training model...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=True
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return self.model
        
    def _optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters using randomized search"""
        param_dist = {
            'n_estimators': [300, 500, 800],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
        }
        
        try:
            model = xgb.XGBClassifier(
                random_state=42,
                device='cuda:0'  # XGBoost 2.0+
            )
            print("🔍 Hyperparameter search using GPU...")
        except:
            model = xgb.XGBClassifier(
                random_state=42,
                device='cpu'
            )
            print("🔍 Hyperparameter search using CPU...")
        
        search = RandomizedSearchCV(
            model, param_dist, n_iter=10, cv=3, scoring='accuracy',
            random_state=42, n_jobs=1, verbose=1
        )
        
        search.fit(X, y)
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")
        
        return search.best_params_
        
    def predict(self, df: pd.DataFrame):
        """Predict patterns with GPU acceleration"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        print(f"🔮 Starting prediction on {len(df)} samples...")
        
        # CRITICAL: Force GPU before each prediction (XGBoost resets to CPU)
        if hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                booster.set_param('device', 'cuda:0')
                print(f"   🎯 GPU device set for prediction")
            except:
                pass
            
        # Extract features
        extractor = (EnhancedFeatureExtractor(df)
                    .add_advanced_price_features()
                    .add_professional_technical_indicators()
                    .add_pattern_specific_features())
        
        features_df = extractor.get_features_df()
        print(f"   Features extracted: {features_df.shape}")
        
        features_scaled = self.scaler.transform(features_df)
        
        # Predict with GPU
        print(f"   Running GPU prediction...")
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        print(f"   ✅ Prediction complete!")
        
        return predicted_labels, probabilities
        
    def save_model(self, filepath: str):
        """Save model with all components"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str, force_gpu: bool = False):
        """Load model with all components"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        # Check current device setting
        current_device = None
        if hasattr(self.model, 'get_xgb_params'):
            params = self.model.get_xgb_params()
            current_device = params.get('device', 'cpu')
        
        print(f"Model loaded from {filepath} (device: {current_device})")
        
        # Force GPU mode if requested
        if force_gpu and hasattr(self.model, 'set_params'):
            try:
                # XGBoost 2.0+: Just set device='cuda', tree_method is automatic
                self.model.set_params(device='cuda:0')
                
                # Configure booster directly
                booster = self.model.get_booster()
                booster.set_param('device', 'cuda:0')
                
                print("✅ GPU mode enabled (CUDA:0)")
            except Exception as e:
                print(f"⚠️  Could not enable GPU: {e}")
                try:
                    self.model.set_params(device='cpu')
                    print(f"Fallback to CPU mode")
                except:
                    pass
        else:
            # Default to CPU to avoid GPU/CPU mismatch
            if hasattr(self.model, 'set_params'):
                try:
                    self.model.set_params(device='cpu')
                    print("CPU mode")
                except:
                    pass


class Classifier:
    """Wrapper class for backward compatibility"""
    
    def __init__(self, model_path: str = None):
        self.model = EnhancedForexPatternClassifier()
        self.model_path = model_path

    def load_model_if_exists(self, path: Path, force_gpu: bool = True):
        try:
            if path and path.exists():
                self.model.load_model(str(path), force_gpu=force_gpu)
                print('Classifier loaded from', path)
                return True
            else:
                print('No saved model found at', path)
                return False
        except Exception as e:
            print('Error loading classifier model:', e)
            return False

    def predict(self, df):
        return self.model.predict(df)

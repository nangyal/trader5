# Classifier Integrálása

## Mit csináltam

A teljes `EnhancedForexPatternClassifier` osztályt bemásoltam az `old/forex_pattern_classifier.py` fájlból a `classes/classification.py` fájlba.

## Változások

### Előtte
- `classes/classification.py` külső függőségeket használt (`utils.classifier`, `utils.advanced_classifier`)
- Importálta az `old` könyvtárból a classifiert

### Utána
- **Teljes önállóság**: Az összes classifier kód most a `classes/classification.py`-ban van
- **Nincs külső függőség**: Nem importál az `old/` könyvtárból
- **GPU támogatás**: Teljes XGBoost GPU gyorsítás (gpu_hist, CUDA)

## Beintegrált komponensek

1. **EnhancedFeatureExtractor**
   - Professional technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
   - Advanced price features (candlestick patterns, shadows)
   - Pattern-specific features (support/resistance, trend strength)

2. **EnhancedForexPatternClassifier**
   - GPU-optimized XGBoost training
   - Hyperparameter optimization (RandomizedSearchCV)
   - Feature extraction and scaling
   - Model save/load with GPU mode support

3. **Classifier wrapper**
   - Backward kompatibilitás
   - `load_model_if_exists()` force_gpu paraméterrel
   - `predict()` metódus

## Használat

```python
from classes.classification import Classifier

# Létrehozás
classifier = Classifier(model_path='models/enhanced_forex_pattern_model.pkl')

# Modell betöltése GPU módban
classifier.load_model_if_exists(Path('models/enhanced_forex_pattern_model.pkl'), force_gpu=True)

# Predikció
predictions, probabilities = classifier.predict(df)
```

## GPU Mód

A classifier automatikusan GPU-t használ ha elérhető:
- **Training**: `tree_method='gpu_hist'`, `device='cuda'`
- **Prediction**: `predictor='gpu_predictor'`
- **Fallback**: Automatikusan CPU-ra vált ha GPU nem elérhető

## Tesztelés

```bash
# Import teszt
python -c "from classes.classification import Classifier; print('✅ OK')"

# Teljes rendszer teszt
python start.py
```

## Kimenet példa

```
Model loaded from /home/nangyal/Desktop/v4/models/enhanced_forex_pattern_model.pkl (device: GPU)
Model set to GPU predictor
Classifier loaded from /home/nangyal/Desktop/v4/models/enhanced_forex_pattern_model.pkl
BacktestRunner: running for coins: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
```

## Előnyök

✅ **Önálló modul**: Nincs függőség az `old/` könyvtártól
✅ **GPU gyorsítás**: Teljes XGBoost GPU támogatás
✅ **Professional features**: 40+ technical indicator
✅ **Backward compatible**: Működik a meglévő kóddal
✅ **Robust**: Error handling, fallback mechanizmusok

## Megjegyzések

- Az `utils/classifier.py` és `utils/advanced_classifier.py` fájlok most már nem szükségesek
- A `config.USE_ADVANCED_CLASSIFIER` flag is elhagyható
- A teljes classifier logika most egy helyen van: `classes/classification.py`

# run_ensemble.py
"""
Запуск комбинированной модели прогнозирования (Random Forest + LSTM)

Использование:
    python run_ensemble.py

Описание:
    Модель использует исторические данные о загруженности парковки
    и прогнозирует будущие значения на основе комбинации:
    - Random Forest (краткосрочный прогноз)
    - LSTM (долгосрочный прогноз)
    - Ансамбль (взвешенное усреднение)
"""

import os
import sys

print("="*70)
print("🚗 ЗАПУСК КОМБИНИРОВАННОЙ МОДЕЛИ ПРОГНОЗИРОВАНИЯ")
print("Random Forest + LSTM")
print("="*70)

# Проверка наличия необходимых пакетов
try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.ensemble import RandomForestRegressor
    print("\n✓ Все необходимые пакеты найдены")
    print(f"  TensorFlow версия: {tf.__version__}")
except ImportError as e:
    print(f"\n❌ Ошибка импорта: {e}")
    print("\nУстановите необходимые пакеты:")
    print("  pip install tensorflow scikit-learn pandas numpy matplotlib")
    sys.exit(1)

# Запуск модели
from ensemble_rf_lstm import main

if __name__ == "__main__":
    main()
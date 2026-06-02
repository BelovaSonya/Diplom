"""
Запуск комбинированной модели (RF + LSTM)
"""

import os
import sys

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
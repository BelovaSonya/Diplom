# run_ensemble_sarima_fixed.py
"""
Запуск комбинированной модели (RF + SARIMA) - ИСПРАВЛЕННАЯ
"""

import os
import sys

print("="*70)
print("🚗 ЗАПУСК МОДЕЛИ RF + SARIMA")
print("="*70)

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    print("\n✓ Пакеты загружены")
except ImportError as e:
    print(f"\n❌ Ошибка: {e}")
    sys.exit(1)

from ensemble_rf_sarima_fixed import main

if __name__ == "__main__":
    main()
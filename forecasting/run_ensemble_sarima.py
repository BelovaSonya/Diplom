"""
Запуск комбинированной модели (RF + SARIMA)
"""

import os
import sys

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    print("\n✓ Пакеты загружены")
except ImportError as e:
    print(f"\n❌ Ошибка: {e}")
    sys.exit(1)

from ensemble_rf_sarima import main

if __name__ == "__main__":
    main()
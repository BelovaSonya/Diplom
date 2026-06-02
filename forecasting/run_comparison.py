"""
Запуск сравнения моделей RF+LSTM vs RF+SARIMA
"""

import os
import sys

# Проверка пакетов
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("\n✓ Все пакеты найдены")
except ImportError as e:
    print(f"\n❌ Ошибка: {e}")
    sys.exit(1)

# Запуск сравнения
from compare_ensembles import main

if __name__ == "__main__":
    main()
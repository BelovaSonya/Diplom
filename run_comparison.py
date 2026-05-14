# run_comparison.py
"""
Запуск сравнения моделей RF+LSTM vs RF+SARIMA

Использование:
    python run_comparison.py

Вывод:
    - Сравнение метрик (MAE, RMSE, MAPE, R²)
    - Графики производительности
    - Радарная диаграмма
    - Итоговый отчет с победителем
"""

import os
import sys

print("="*70)
print("🚗 ЗАПУСК СРАВНЕНИЯ МОДЕЛЕЙ")
print("RF+LSTM vs RF+SARIMA")
print("="*70)

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
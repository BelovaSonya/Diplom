# run_advanced.py
"""
Запуск продвинутого визуализатора парковки Москвы
"""

import os
import sys

print("="*60)
print("🚗 ПАРКОВКИ МОСКВЫ - ПРОДВИНУТАЯ СИСТЕМА МОНИТОРИНГА")
print("="*60)

# Проверка наличия моделей
models_dir = os.path.join(os.path.dirname(__file__), "models")
models_found = False

if os.path.exists(models_dir):
    models = os.listdir(models_dir)
    if models:
        models_found = True
        print(f"\n✓ Найдены модели: {', '.join(models[:3])}")
    else:
        print("\n⚠ Модели не найдены, работа в демо-режиме")
else:
    print("\n⚠ Папка models не найдена, работа в демо-режиме")

print("\n" + "="*60)
print("ЗАПУСК ВИЗУАЛИЗАТОРА...")
print("="*60)
print("\nОсобенности интерфейса:")
print("  • Современный дизайн в светло-зеленых тонах")
print("  • Интерактивная карта парковки Москвы")
print("  • График динамики загрузки")
print("  • Прогноз освобождения мест")
print("  • ⏰ ПРОГНОЗ ПОЛНОГО ОСВОБОЖДЕНИЯ ПАРКОВКИ")
print("  • 📊 ПОЧАСОВОЙ ПРОГНОЗ ЗАГРУЗКИ")
print("  • Интеллектуальные рекомендации")
print("  • Автообновление каждые 10 секунд")
print("\n" + "="*60)

# Запуск
from parking_visualizer_advanced import AdvancedParkingVisualizer

if __name__ == "__main__":
    app = AdvancedParkingVisualizer()
    app.run()
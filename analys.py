# analyze_data.py
import os
import cv2
from pathlib import Path

BASE_PATH = r"C:\Users\Администратор\Desktop\Учеба\ДИПЛОМ\pklot_data"

print("="*60)
print("АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
print("="*60)

# Проверяем наличие файлов
if os.path.exists(BASE_PATH):
    print(f"\nПапка существует: {BASE_PATH}")
    
    # Ищем все файлы
    all_files = []
    for root, dirs, files in os.walk(BASE_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))
    
    print(f"\nНайдено изображений: {len(all_files)}")
    
    if len(all_files) > 0:
        print("\nПримеры имен файлов (первые 10):")
        for i, file in enumerate(all_files[:10]):
            rel_path = os.path.relpath(file, BASE_PATH)
            print(f"  {i+1}. {rel_path}")
        
        # Проверяем имена файлов на наличие информации о классе
        print("\nАнализ имен файлов:")
        has_occupied = any('occupied' in f.lower() for f in all_files)
        has_free = any('free' in f.lower() for f in all_files)
        
        if has_occupied or has_free:
            print("  ✓ В именах файлов есть слова 'occupied' или 'free'")
            if has_occupied:
                occupied_count = sum(1 for f in all_files if 'occupied' in f.lower())
                print(f"    - 'occupied' найдено в {occupied_count} файлах")
            if has_free:
                free_count = sum(1 for f in all_files if 'free' in f.lower())
                print(f"    - 'free' найдено в {free_count} файлах")
        else:
            print("  ✗ В именах файлов нет слов 'occupied' или 'free'")
        
        # Ищем CSV файлы
        csv_files = [f for f in os.listdir(BASE_PATH) if f.endswith('.csv')]
        if csv_files:
            print(f"\nНайдены CSV файлы: {csv_files}")
        else:
            print("\nCSV файлы не найдены")
            
        # Проверяем размеры изображений
        print("\nПроверка размеров изображений (первые 3):")
        for i, file in enumerate(all_files[:3]):
            img = cv2.imread(file)
            if img is not None:
                h, w = img.shape[:2]
                print(f"  {i+1}. {os.path.basename(file)} - размер: {w}x{h}")
            else:
                print(f"  {i+1}. Не удалось загрузить")
else:
    print(f"Ошибка: папка {BASE_PATH} не найдена")
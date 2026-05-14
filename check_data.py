# check_data.py
"""
Проверка наличия и структуры данных
"""

import os

DATA_PATH = r"C:\Users\Администратор\Desktop\Учеба\ДИПЛОМ\pklot_project\data"

print("="*60)
print("ПРОВЕРКА ДАННЫХ")
print("="*60)

if os.path.exists(DATA_PATH):
    print(f"✓ Папка data существует: {DATA_PATH}")
    
    # Считаем изображения
    images = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✓ Найдено изображений: {len(images)}")
    
    # Показываем примеры
    if images:
        print("\nПримеры имен файлов:")
        for img in images[:10]:
            print(f"  {img}")
        
        # Проверяем формат имен
        print("\nАнализ формата имен:")
        sample = images[0]
        parts = sample.split('_')
        print(f"  Имя файла: {sample}")
        print(f"  Количество частей: {len(parts)}")
        
        if len(parts) >= 6:
            print("  Формат: YYYY-MM-DD_HH_MM_SS.jpg ✓")
        else:
            print("  Формат не соответствует ожидаемому")
    
else:
    print(f"❌ Папка data не найдена: {DATA_PATH}")
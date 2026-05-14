# data_loader.py
import os
import numpy as np
import cv2
from tqdm import tqdm
import config
import shutil

class PKLotDataLoader:
    """Класс для загрузки и предобработки данных PKLot"""
    
    def __init__(self, base_path=config.ORIGINAL_BASE_PATH, img_size=config.IMG_SIZE):
        self.original_base_path = base_path
        self.img_size = img_size
        self.data = []
        self.labels = []
        
    def copy_images_to_temp(self):
        """Копируем изображения во временную папку с английским путем"""
        temp_images_path = config.IMAGES_PATH
        
        if os.path.exists(temp_images_path):
            print(f"Временная папка уже существует: {temp_images_path}")
            return temp_images_path
        
        original_images_path = os.path.join(self.original_base_path, "images")
        
        if not os.path.exists(original_images_path):
            print(f"Ошибка: Папка {original_images_path} не найдена")
            return None
        
        print(f"\nКопирование изображений во временную папку...")
        print(f"Из: {original_images_path}")
        print(f"В: {temp_images_path}")
        
        # Создаем временную папку
        os.makedirs(temp_images_path, exist_ok=True)
        
        # Копируем все изображения
        images = [f for f in os.listdir(original_images_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(images, desc="Копирование"):
            src = os.path.join(original_images_path, img_name)
            dst = os.path.join(temp_images_path, img_name)
            shutil.copy2(src, dst)
        
        print(f"Скопировано {len(images)} изображений")
        return temp_images_path
    
    def load_data_from_structure(self):
        """Загрузка данных из временной папки"""
        print("="*50)
        print("Загрузка данных PKLot")
        print("="*50)
        
        # Копируем изображения во временную папку
        temp_path = self.copy_images_to_temp()
        if temp_path is None:
            return np.array([]), np.array([])
        
        # Получаем список всех изображений
        all_images = [f for f in os.listdir(temp_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nНайдено изображений: {len(all_images)}")
        
        if len(all_images) == 0:
            print("Изображения не найдены!")
            return np.array([]), np.array([])
        
        # Для теста возьмем только первые 1000 изображений (чтобы не ждать долго)
        # Если хотите все, уберите [:1000]
        sample_images = all_images
        print(f"Загружаем {len(sample_images)} изображений для обучения...")
        
        # Загружаем изображения
        for img_name in tqdm(sample_images, desc="Загрузка изображений"):
            img_path = os.path.join(temp_path, img_name)
            img = self.load_and_preprocess_image(img_path)
            if img is not None:
                self.data.append(img)
                # Временно присваиваем случайные метки для теста
                # В реальности нужно определять по времени или другим признакам
                self.labels.append(np.random.randint(0, 2))
        
        print(f"\nЗагружено изображений: {len(self.data)}")
        print(f"Занятых: {sum(self.labels)}")
        print(f"Свободных: {len(self.labels) - sum(self.labels)}")
        
        return np.array(self.data), np.array(self.labels)
    
    def load_and_preprocess_image(self, image_path):
        """Загрузка и предобработка изображения"""
        try:
            # Пробуем загрузить с помощью cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0
            return img
        except Exception as e:
            return None
    
    def extract_features_from_images(self):
        """Извлечение признаков из изображений"""
        if len(self.data) == 0:
            print("Нет данных для извлечения признаков")
            return np.array([])
            
        print("\nИзвлечение признаков из изображений...")
        features = []
        
        for img in tqdm(self.data, desc="Извлечение признаков"):
            # Статистические признаки
            mean_rgb = np.mean(img, axis=(0, 1))
            std_rgb = np.std(img, axis=(0, 1))
            
            # Гистограммные признаки
            hist_features = []
            for i in range(3):
                hist = np.histogram(img[:,:,i], bins=10, range=(0, 1))[0]
                hist_features.extend(hist)
            
            # Текстурные признаки
            gray = np.mean(img, axis=2)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            features.append(np.concatenate([
                mean_rgb, std_rgb, 
                hist_features,
                [np.mean(gradient_magnitude), np.std(gradient_magnitude)]
            ]))
        
        return np.array(features)
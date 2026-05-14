# cnn_models.py
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ПРАВИЛЬНЫЕ импорты - используем tensorflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import config

# Отключение предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

class ParkingCNNModels:
    """Класс для CNN моделей"""
    
    def __init__(self, img_size=config.IMG_SIZE):
        self.img_size = img_size
        self.models = {}
        self.history = {}
        print(f"ParkingCNNModels инициализирован")
        
    def create_simple_cnn(self):
        """Создание простой CNN"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_mobilenet(self):
        """Создание модели MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X_images, y):
        """Обучение CNN моделей"""
        print("\n" + "="*50)
        print("Обучение CNN моделей")
        print("="*50)
        
        if len(X_images) == 0:
            print("Ошибка: Нет данных для обучения")
            return None, None, {}
        
        print(f"Всего изображений: {len(X_images)}")
        print(f"Занятых: {sum(y)}, Свободных: {len(y)-sum(y)}")
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_images, y, test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        print(f"Обучающая выборка: {len(X_train)}")
        print(f"Тестовая выборка: {len(X_test)}")
        
        # Создание генератора с аугментацией
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = datagen.flow(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            subset='training'
        )
        
        val_generator = datagen.flow(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            subset='validation'
        )
        
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=2),
            ModelCheckpoint(
                os.path.join(config.MODELS_DIR, 'best_cnn.h5'),
                save_best_only=True
            )
        ]
        
        results = {}
        
        # Обучение Simple CNN
        print("\n1. Обучение Simple CNN...")
        try:
            cnn_model = self.create_simple_cnn()
            print("Модель создана, начинаем обучение...")
            
            history_cnn = cnn_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=min(config.EPOCHS, 3),  # 3 эпохи для быстрого теста
                callbacks=callbacks,
                verbose=1
            )
            
            self.models['Simple CNN'] = cnn_model
            self.history['Simple CNN'] = history_cnn
            
            # Оценка на тестовых данных
            loss, accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
            results['Simple CNN'] = {'Accuracy': accuracy, 'Loss': loss}
            print(f"\n✓ Simple CNN обучена!")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"✗ Ошибка при обучении Simple CNN: {e}")
            import traceback
            traceback.print_exc()
        
        return X_test, y_test, results
    
    def evaluate_models(self, X_test, y_test):
        """Оценка CNN моделей"""
        print("\n" + "="*50)
        print("Оценка CNN моделей")
        print("="*50)
        
        results = {}
        for name, model in self.models.items():
            try:
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                results[name] = {'Accuracy': accuracy, 'Loss': loss}
                print(f"\n{name}:")
                print(f"  Test Accuracy: {accuracy:.4f}")
                print(f"  Test Loss: {loss:.4f}")
            except Exception as e:
                print(f"\n{name}: Ошибка оценки - {e}")
                results[name] = {'Accuracy': 0, 'Loss': 0}
        
        return results
    
    def save_models(self):
        """Сохранение CNN моделей"""
        for name, model in self.models.items():
            try:
                filename = os.path.join(config.MODELS_DIR, f"{name.replace(' ', '_')}.h5")
                model.save(filename)
                print(f"Сохранена модель: {filename}")
            except Exception as e:
                print(f"Ошибка сохранения модели {name}: {e}")

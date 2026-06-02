import numpy as np
import os
import cv2
import joblib
from keras.models import load_model
import config

class ParkingPredictor:
    """Класс для прогнозирования на новых изображениях"""
    
    def __init__(self, traditional_model_path=None, cnn_model_path=None):
        self.traditional_model = None
        self.cnn_model = None
        self.scaler = None
        
        if traditional_model_path:
            self.traditional_model = joblib.load(traditional_model_path)
            self.scaler = joblib.load(os.path.join(config.MODELS_DIR, 'scaler.pkl'))
        
        if cnn_model_path:
            self.cnn_model = load_model(cnn_model_path)
    
    def load_and_preprocess_image(self, image_path):
        """Загрузка и предобработка изображения"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, config.IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        return img
    
    def extract_features(self, img):
        """Извлечение признаков из изображения"""
        mean_rgb = np.mean(img, axis=(0, 1))
        std_rgb = np.std(img, axis=(0, 1))
        
        hist_features = []
        for i in range(3):
            hist = np.histogram(img[:,:,i], bins=10, range=(0, 1))[0]
            hist_features.extend(hist)
        
        gray = np.mean(img, axis=2)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return np.concatenate([
            mean_rgb, std_rgb, 
            hist_features,
            [np.mean(gradient_magnitude), np.std(gradient_magnitude)]
        ])
    
    def predict(self, image_path, method='ensemble'):
        """Прогнозирование для одного изображения"""
        img = self.load_and_preprocess_image(image_path)
        if img is None:
            return None
        
        predictions = {}
        
        # Традиционная модель
        if method in ['traditional', 'ensemble'] and self.traditional_model:
            features = self.extract_features(img)
            features_scaled = self.scaler.transform([features])
            pred = self.traditional_model.predict(features_scaled)[0]
            prob = self.traditional_model.predict_proba(features_scaled)[0][1]
            predictions['traditional'] = {
                'class': int(pred),
                'probability': float(prob),
                'label': 'Занято' if pred == 1 else 'Свободно'
            }
        
        # CNN модель
        if method in ['cnn', 'ensemble'] and self.cnn_model:
            img_batch = np.expand_dims(img, axis=0)
            prob = self.cnn_model.predict(img_batch, verbose=0)[0][0]
            pred = 1 if prob > 0.5 else 0
            predictions['cnn'] = {
                'class': pred,
                'probability': float(prob),
                'label': 'Занято' if pred == 1 else 'Свободно'
            }
        
        # Ансамбль
        if method == 'ensemble' and len(predictions) == 2:
            ensemble_prob = (predictions['traditional']['probability'] + 
                           predictions['cnn']['probability']) / 2
            ensemble_class = 1 if ensemble_prob > 0.5 else 0
            predictions['ensemble'] = {
                'class': ensemble_class,
                'probability': ensemble_prob,
                'label': 'Занято' if ensemble_class == 1 else 'Свободно'
            }
        
        return predictions
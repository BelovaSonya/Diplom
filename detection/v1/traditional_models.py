import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import config
import joblib
import os

class ParkingTraditionalModels:
    """Класс для традиционных моделей ML"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def train_models(self, X, y):
        """Обучение традиционных моделей"""
        print("\n" + "="*50)
        print("Обучение традиционных моделей ML")
        print("="*50)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest Classifier
        print("\n1. Обучение Random Forest Classifier...")
        rf_clf = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
        rf_clf.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_clf
        
        # Random Forest Regressor
        print("2. Обучение Random Forest Regressor...")
        rf_reg = RandomForestRegressor(
            n_estimators=100, max_depth=10,
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
        rf_reg.fit(X_train_scaled, y_train)
        self.models['Random Forest Regressor'] = rf_reg
        
        # Оценка
        self.evaluate_models(X_test_scaled, y_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_models(self, X_test, y_test):
        """Оценка производительности"""
        print("\n" + "="*50)
        print("Оценка моделей")
        print("="*50)
        
        results = {}
        for name, model in self.models.items():
            if 'Regressor' in name:
                y_pred = model.predict(X_test)
                y_pred_class = (y_pred > 0.5).astype(int)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred_class)
                results[name] = {'MSE': mse, 'R2': r2, 'Accuracy': acc}
                print(f"\n{name}:")
                print(f"  MSE: {mse:.4f}")
                print(f"  R2: {r2:.4f}")
                print(f"  Accuracy: {acc:.4f}")
            else:
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[name] = {'Accuracy': acc}
                print(f"\n{name}:")
                print(f"  Accuracy: {acc:.4f}")
                print(classification_report(y_test, y_pred, 
                                          target_names=['Свободно', 'Занято']))
        
        return results
    
    def save_models(self):
        """Сохранение моделей"""
        for name, model in self.models.items():
            filename = os.path.join(config.MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
            joblib.dump(model, filename)
            print(f"Сохранена модель: {filename}")
        
        # Сохранение скейлера
        scaler_path = os.path.join(config.MODELS_DIR, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Сохранен скейлер: {scaler_path}")
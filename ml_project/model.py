from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

class MLModel:
    def __init__(self, algorithm='rf'):
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        
        # 定义多个算法选项
        self.algorithms = {
            'rf': RandomForestClassifier(random_state=42),
            'gb': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42)
        }
        
        # 为每个算法定义参数网格
        self.param_grids = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gb': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        self.model = self.algorithms[algorithm]
        
    def train(self, X, y, optimize=True, cv=5):
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        if optimize:
            # 使用网格搜索找到最佳参数
            grid_search = GridSearchCV(
                self.model,
                self.param_grids[self.algorithm],
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        else:
            # 普通训练
            self.model.fit(X_scaled, y)
        
        # 执行交叉验证
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        print(f"交叉验证分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        print("\n分类报告:")
        print(classification_report(y, y_pred))
        
        print("\n混淆矩阵:")
        print(confusion_matrix(y, y_pred))
        
    def predict(self, X, return_proba=False):
        X_scaled = self.scaler.transform(X)
        if return_proba and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return self.model.predict(X_scaled)
    
    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) 
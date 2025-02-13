from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import joblib
import os
from model import MLModel
import pandas as pd

def prepare_data():
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    # 加载示例数据集（乳腺癌数据集）
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # 保存特征名称
    pd.DataFrame({'feature_names': feature_names}).to_csv('data/feature_names.csv', index=False)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化并训练模型
    model = MLModel()
    model.train(X_train, y_train)
    
    # 评估模型
    train_score = model.model.score(X_train, y_train)
    test_score = model.model.score(X_test, y_test)
    
    # 生成预测概率
    y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    y_pred = model.model.predict(X_test)
    
    # 保存数据
    joblib.dump(model.model, 'data/best_model.joblib')
    joblib.dump(model.scaler, 'data/best_scaler.joblib')
    
    # 保存评估结果
    results = {
        'train_score': train_score,
        'test_score': test_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    joblib.dump(results, 'data/evaluation_results.joblib')
    
    print("数据准备完成！模型已训练并保存。")

if __name__ == '__main__':
    prepare_data()

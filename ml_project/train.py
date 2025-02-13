import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from model import MLModel
from src.visualization import MLVisualizer
from src.data_analysis import DataAnalyzer
from src.utils import MLUtils

def main():
    # 初始化工具
    visualizer = MLVisualizer()
    analyzer = DataAnalyzer()
    
    # 加载数据集
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # 数据分析
    print("\n=== 开始数据分析 ===")
    analyzer.analyze_features(X, data.feature_names)
    analyzer.analyze_correlations(X, data.feature_names)
    analyzer.detect_outliers(X, data.feature_names)
    
    # 数据可视化
    print("\n=== 生成数据可视化 ===")
    visualizer.plot_feature_distribution(X.values, y, data.feature_names)
    visualizer.plot_tsne(X.values, y)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练不同模型
    algorithms = ['rf', 'gb', 'svm']
    best_score = 0
    best_model = None
    results = {}
    
    for algo in algorithms:
        print(f"\n=== 训练 {algo} 模型 ===")
        model = MLModel(algorithm=algo)
        model.train(X_train, y_train, optimize=True, cv=5)
        
        # 评估模型
        print("\n模型评估:")
        model.evaluate(X_test, y_test)
        
        # 保存结果
        current_score = model.model.score(model.scaler.transform(X_test), y_test)
        results[algo] = {
            'test_score': current_score,
            'parameters': model.model.get_params()
        }
        
        if current_score > best_score:
            best_score = current_score
            best_model = model
    
    # 为最佳模型生成可视化
    if hasattr(best_model.model, 'feature_importances_'):
        visualizer.plot_feature_importance(best_model.model, data.feature_names)
    
    y_pred_proba = best_model.predict(X_test, return_proba=True)
    if y_pred_proba is not None:
        visualizer.plot_roc_curve(y_test, y_pred_proba[:, 1])
    
    # 保存实验结果
    MLUtils.save_experiment_results(results)
    
    print(f"\n最佳模型使用算法: {best_model.algorithm}")
    print(f"最佳测试集分数: {best_score:.4f}")
    
    # 保存最佳模型
    best_model.save_model("data/best_model.joblib", "data/best_scaler.joblib")

if __name__ == "__main__":
    main() 
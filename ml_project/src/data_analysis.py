import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self):
        self.summary = {}
    
    def analyze_features(self, X, feature_names):
        """分析特征统计信息"""
        df = pd.DataFrame(X, columns=feature_names)
        
        analysis = {
            '基础统计量': df.describe(),
            '缺失值': df.isnull().sum(),
            '偏度': df.skew(),
            '峰度': df.kurtosis()
        }
        
        print("\n=== 数据分析报告 ===")
        for name, stats in analysis.items():
            print(f"\n{name}:\n", stats)
        
        return analysis
    
    def analyze_correlations(self, X, feature_names, threshold=0.7):
        """分析特征相关性"""
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr()
        
        # 找出高相关性特征
        high_corr = np.where(np.abs(corr_matrix) > threshold)
        high_corr = [(feature_names[i], feature_names[j], corr_matrix.iloc[i, j])
                     for i, j in zip(*high_corr) if i < j]
        
        print("\n=== 特征相关性分析 ===")
        print(f"\n相关系数大于{threshold}的特征对:")
        for feat1, feat2, corr in high_corr:
            print(f"{feat1} - {feat2}: {corr:.3f}")
        
        return corr_matrix, high_corr
    
    def detect_outliers(self, X, feature_names, threshold=3):
        """检测异常值"""
        df = pd.DataFrame(X, columns=feature_names)
        outliers = {}
        
        for feature in feature_names:
            z_scores = np.abs(stats.zscore(df[feature]))
            outliers[feature] = np.sum(z_scores > threshold)
        
        print("\n=== 异常值检测 ===")
        print(f"使用Z分数方法 (阈值: {threshold})")
        for feature, count in outliers.items():
            print(f"{feature}: {count}个异常值")
        
        return outliers
    
    def feature_importance_analysis(self, model, feature_names):
        """分析特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feat_imp = pd.DataFrame({
                '特征': feature_names,
                '重要性': importance
            }).sort_values('重要性', ascending=False)
            
            print("\n=== 特征重要性分析 ===")
            print(feat_imp)
            
            return feat_imp
        return None 
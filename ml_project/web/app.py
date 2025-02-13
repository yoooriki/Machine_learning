from flask import Flask, render_template
import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import MLVisualizer
from model import MLModel
import pandas as pd
import numpy as np
import json
import plotly
import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

app = Flask(__name__)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive model performance metrics"""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # 计算混淆矩阵相关指标
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)  # Negative Predictive Value
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'npv': npv,
        'confusion_matrix': cm
    }

def load_data():
    """Load model, results and calculate comprehensive metrics"""
    try:
        # 获取项目根目录的路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Project root: {project_root}")
        
        # 加载模型和结果
        model = MLModel()
        model_path = os.path.join(project_root, 'data/best_model.joblib')
        scaler_path = os.path.join(project_root, 'data/best_scaler.joblib')
        
        print(f"Loading model from: {model_path}")
        model.model = joblib.load(model_path)
        print(f"Loading scaler from: {scaler_path}")
        model.scaler = joblib.load(scaler_path)
        
        # 加载评估结果
        results_path = os.path.join(project_root, 'data/evaluation_results.joblib')
        print(f"Loading results from: {results_path}")
        results = joblib.load(results_path)
        
        # 加载特征名称
        features_path = os.path.join(project_root, 'data/feature_names.csv')
        print(f"Loading feature names from: {features_path}")
        feature_names = pd.read_csv(features_path)['feature'].tolist()
        
        # 计算指标
        print("Calculating metrics...")
        metrics = calculate_metrics(results['y_test'], results['y_pred'], results['y_pred_proba'])
        
        return model, results, feature_names, metrics
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise

@app.route('/')
def index():
    try:
        # 加载数据和计算指标
        print("Loading data...")
        model, results, feature_names, metrics = load_data()
        print("Data loaded successfully")
        
        # 初始化可视化器和空的JSON数据
        print("Initializing visualizer...")
        visualizer = MLVisualizer('visualizations')
        importance_json = "{}"
        roc_json = "{}"
        cm_json = "{}"
        
        # 生成特征重要性图
        print("Generating feature importance plot...")
        try:
            if model.model is not None and feature_names:
                fig_importance = visualizer.plot_feature_importance(model.model, feature_names)
                if fig_importance:
                    importance_json = json.dumps(fig_importance, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error generating importance plot: {str(e)}")
        
        # 生成ROC曲线
        print("Generating ROC curve...")
        try:
            if 'y_test' in results and 'y_pred_proba' in results:
                fig_roc = visualizer.plot_roc_curve(results['y_test'], results['y_pred_proba'])
                if fig_roc:
                    roc_json = json.dumps(fig_roc, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error generating ROC plot: {str(e)}")
        
        # 生成混淆矩阵
        print("Generating confusion matrix...")
        try:
            if 'confusion_matrix' in metrics:
                fig_cm = visualizer.plot_confusion_matrix(metrics['confusion_matrix'])
                if fig_cm:
                    cm_json = json.dumps(fig_cm, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error generating confusion matrix plot: {str(e)}")
        
        # 格式化指标为百分比，添加默认值
        print("Formatting metrics...")
        formatted_metrics = {
            'train_accuracy': f"{results.get('train_score', 0)*100:.1f}",
            'test_accuracy': f"{results.get('test_score', 0)*100:.1f}",
            'precision': f"{metrics.get('precision', 0)*100:.1f}",
            'recall': f"{metrics.get('recall', 0)*100:.1f}",
            'f1': f"{metrics.get('f1', 0)*100:.1f}",
            'auc_score': f"{metrics.get('auc', 0):.3f}",
            'specificity': f"{metrics.get('specificity', 0)*100:.1f}",
            'npv': f"{metrics.get('npv', 0)*100:.1f}",
            'current_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("Rendering template...")
        return render_template('index.html',
                             importance_plot=importance_json,
                             roc_plot=roc_json,
                             cm_plot=cm_json,
                             **formatted_metrics)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return render_template('error.html', error=str(e))

def find_free_port():
    """Find a free port to run the application"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    port = find_free_port()
    print(f"\n* 应用将在以下地址运行: http://localhost:{port}")
    app.run(debug=True, port=port, host='0.0.0.0')

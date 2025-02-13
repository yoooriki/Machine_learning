import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os

class MLVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # 定义基本颜色
        self.colors = {
            'blue': '#1f77b4',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'background': '#ffffff',
            'grid': '#eeeeee',
            'text': '#333333'
        }
        
    def plot_feature_importance(self, model, feature_names, title="特征重要性分析"):
        """绘制特征重要性图并返回Plotly图形对象"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            data = pd.DataFrame({
                '特征': [feature_names[i] for i in indices],
                '重要性': importance[indices]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data['重要性'],
                y=data['特征'],
                orientation='h',
                marker=dict(
                    color=self.colors['blue'],
                    opacity=0.8
                )
            ))
            
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=20, color=self.colors['text'])
                ),
                xaxis=dict(
                    title='重要性得分',
                    gridcolor=self.colors['grid'],
                    showgrid=True
                ),
                yaxis=dict(
                    title='特征名称',
                    gridcolor=self.colors['grid'],
                    showgrid=True
                ),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                height=max(600, len(feature_names) * 25),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
        
    def plot_roc_curve(self, y_true, y_pred_proba, title="ROC曲线分析"):
        """绘制ROC曲线并返回Plotly图形对象"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # 添加ROC曲线
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'ROC曲线 (AUC = {roc_auc:.3f})',
            mode='lines',
            line=dict(
                color=self.colors['blue'],
                width=2
            ),
            fill='tozeroy',
            fillcolor=f'rgba(31, 119, 180, 0.1)'  # blue with alpha
        ))
        
        # 添加对角线
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='随机猜测',
            mode='lines',
            line=dict(
                color=self.colors['text'],
                width=1,
                dash='dash'
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=self.colors['text'])
            ),
            xaxis=dict(
                title='假阳性率',
                gridcolor=self.colors['grid'],
                showgrid=True
            ),
            yaxis=dict(
                title='真阳性率',
                gridcolor=self.colors['grid'],
                showgrid=True
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm, classes=['良性', '恶性'], title="混淆矩阵分析"):
        """绘制混淆矩阵热力图并返回Plotly图形对象"""
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        # 创建注释文本
        annotations = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                annotations.append(dict(
                    x=classes[j],
                    y=classes[i],
                    text=f'数量: {cm[i, j]}<br>占比: {cm_percent[i, j]:.1f}%',
                    font=dict(color='white' if cm[i, j] > cm.max()/2 else 'black', size=14),
                    showarrow=False
                ))
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale=[
                [0, 'rgb(240,240,240)'],
                [1, self.colors['blue']]
            ],
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='样本数量',
                    font=dict(size=12)
                )
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color=self.colors['text'])
            ),
            xaxis=dict(
                title='预测类别',
                gridcolor=self.colors['grid'],
                showgrid=True
            ),
            yaxis=dict(
                title='真实类别',
                gridcolor=self.colors['grid'],
                showgrid=True
            ),
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            annotations=annotations,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig

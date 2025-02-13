# 机器学习模型分析系统

这是一个综合的机器学习模型分析系统，提供模型训练、评估和可视化功能，包括交互式Web界面展示模型性能指标和各类分析图表。

## 安装和环境配置

### 依赖要求
- Python >= 3.7
- pandas
- numpy
- scikit-learn
- Flask
- plotly
- joblib

### 安装步骤
1. 克隆项目到本地：
```bash
git clone https://github.com/your-username/ml_project.git
cd ml_project
```

2. 创建并激活虚拟环境（可选）：
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 主要功能

### 1. 模型训练与评估
- 自动数据预处理和特征缩放
- 模型训练与保存
- 多指标模型评估（准确率、精确度、召回率、F1分数、AUC等）

### 2. 可视化分析
- 特征重要性分析
- ROC曲线分析
- 混淆矩阵可视化
- 学习曲线分析

### 3. Web界面
- 交互式模型性能展示
- 动态图表更新
- 响应式设计

## 项目结构
```
ml_project/
├── data/               # 数据文件
│   ├── best_model.joblib   # 训练好的模型
│   ├── best_scaler.joblib  # 特征缩放器
│   ├── evaluation_results.joblib  # 评估结果
│   └── feature_names.csv  # 特征名称
├── src/                # 源代码
│   └── visualization.py  # 可视化工具
├── web/                # Web应用
│   ├── app.py            # Flask应用
│   ├── templates/        # HTML模板
│   └── static/           # 静态资源
├── model.py            # 模型定义和训练
├── predict.py          # 预测功能
├── prepare_data.py      # 数据准备
├── requirements.txt    # 项目依赖
└── README.md           # 项目文档
```

## 使用方法

### 1. 启动Web应用
```bash
python web/app.py
```
访问 `http://localhost:8888` 查看模型分析报告。

### 2. 训练新模型
```bash
python model.py
```

### 3. 进行预测
```bash
python predict.py
```

## 注意事项
- 首次运行前请确保安装了所有依赖包
- Web应用默认运行在8888端口，如果端口被占用可以在app.py中修改
- 所有数据文件都应该存放在data/目录下

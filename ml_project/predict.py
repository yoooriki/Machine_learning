import numpy as np
import os
from model import MLModel

def main():
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载模型
    model = MLModel()
    model.load_model(
        os.path.join(current_dir, "data/best_model.joblib"),
        os.path.join(current_dir, "data/best_scaler.joblib")
    )
    
    # 示例数据 (使用乳腺癌数据集的格式)
    sample_data = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 
                            0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                            0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                            0.2654, 0.4601, 0.1189]])
    
    # 进行预测
    prediction = model.predict(sample_data)
    probabilities = model.predict(sample_data, return_proba=True)
    
    # 输出预测结果
    print(f"预测类别: {prediction[0]} ({'良性' if prediction[0] == 0 else '恶性'})")
    if probabilities is not None:
        print(f"预测概率: 良性: {probabilities[0][0]:.4f}, 恶性: {probabilities[0][1]:.4f}")

if __name__ == "__main__":
    main() 
import os
import json
import numpy as np
from datetime import datetime

class MLUtils:
    @staticmethod
    def save_experiment_results(results, filename='experiment_results.json'):
        """保存实验结果"""
        results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 转换numpy数组为列表
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
        
        os.makedirs('data', exist_ok=True)
        with open(f'data/{filename}', 'w') as f:
            json.dump(results, f, indent=4)
    
    @staticmethod
    def load_experiment_results(filename='experiment_results.json'):
        """加载实验结果"""
        try:
            with open(f'data/{filename}', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None 
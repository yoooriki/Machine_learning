#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_dependencies():
    """检查是否已安装所需的依赖"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import plotly
        return True
    except ImportError as e:
        print(f"缺少必要的依赖: {str(e)}")
        return False

def install_dependencies():
    """安装项目依赖"""
    print("正在安装必要的依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def prepare_model():
    """准备模型和数据"""
    if not os.path.exists("data/best_model.joblib"):
        print("正在训练模型...")
        subprocess.run([sys.executable, "prepare_data.py"])
    else:
        print("模型文件已存在，跳过训练步骤")

def start_server():
    """启动Web服务器"""
    print("正在启动可视化服务器...")
    print("请稍等，浏览器将自动打开...")
    
    # 启动Flask应用
    process = subprocess.Popen([sys.executable, "web/app.py"])
    
    # 等待服务器启动
    sleep(2)
    
    # 在浏览器中打开页面
    webbrowser.open('http://localhost:3000')
    
    return process

def main():
    # 检查环境
    if not check_dependencies():
        print("是否要安装必要的依赖？(y/n)")
        if input().lower() == 'y':
            install_dependencies()
        else:
            print("无法继续，请先安装必要的依赖")
            return
    
    # 准备模型
    prepare_model()
    
    # 启动服务器
    server_process = start_server()
    
    print("\n可视化界面已启动！")
    print("请在浏览器中访问: http://localhost:3000")
    print("按Ctrl+C退出程序")
    
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n正在关闭服务器...")
        server_process.terminate()
        server_process.wait()
        print("服务器已关闭")

if __name__ == "__main__":
    main()

import MNN
import MNN.numpy as np

onnx_path = "/home/wtbu/agibot_x1/agibot_x1_train/logs/x1_dh_stand/exported_onnx/2025-10-17_11-44-07/x1_policy.onnx"
mnn_path = "/home/wtbu/agibot_x1/agibot_x1_train/logs/x1_dh_stand/exported_onnx/2025-10-17_11-44-07/x1_policy.mnn"

def convert_with_python_api(onnx_path, mnn_path):
    """使用MNN Python API进行转换"""
    
    try:
        # 方法1: 使用express接口
        print("尝试使用Express接口转换...")
        
        # 定义输入（明确指定形状和名称）
        input_var = MNN.express.placeholder([1, 47], MNN.express.NCHW)
        input_var.setName("input")
        
        # 加载ONNX并转换
        model = MNN.express.load([input_var], onnx_path)
        if model:
            MNN.express.save(model, mnn_path)
            print(f"✓ 转换成功: {mnn_path}")
            return True
            
    except Exception as e:
        print(f"Express转换失败: {e}")
    
    try:
        # 方法2: 使用传统的转换方式
        print("尝试传统转换方式...")
        
        # 这里需要根据你的MNN版本调整
        # 不同版本的MNN Python API可能有所不同
        
        print("⚠️ 请检查你的MNN版本和API文档")
        return False
        
    except Exception as e:
        print(f"传统转换失败: {e}")
        return False

# 尝试转换
convert_with_python_api(onnx_path, mnn_path)
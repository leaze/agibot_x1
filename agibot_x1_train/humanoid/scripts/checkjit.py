import torch
import torch.jit

def check_jit_model(jit_path):
    """检查原始JIT模型的输入输出维度"""
    
    # 加载JIT模型
    print("加载JIT模型...", jit_path)
    jit_model = torch.jit.load(jit_path)
    print("✓ JIT模型加载成功")
    
    # 打印模型结构
    print(jit_model.graph)
    
    # 测试输入输出维度
    # 测试帧堆叠输入
    test_input_3102 = torch.randn(1, 3102)
    output_3102 = jit_model(test_input_3102)
    print(f"输入[1,3102] -> 输出形状: {output_3102.shape}")
    
    # 测试单帧输入
    test_input_47 = torch.randn(1, 47)
    try:
        output_47 = jit_model(test_input_47)
        print(f"输入[1,47] -> 输出形状: {output_47.shape}")
    except:
        print("模型不接受[1,47]输入")
    
    return jit_model

# 检查JIT模型
jit_path = "/home/wtbu/agibot_x1/agibot_x1_train/logs/x1_dh_stand/exported_policies/2025-10-17_11-27-00/policy_dh.jit"
check_jit_model(jit_path)

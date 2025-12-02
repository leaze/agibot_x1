# 在Python中验证ONNX模型
import onnx
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
model_path = "/home/wtbu/agibot_x1/agibot_x1_train/logs/x1_dh_stand/exported_onnx/2025-10-17_11-44-07/x1_policy.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# 打印模型信息
print("===== ONNX模型信息 =====")
print("输入信息:")
for input in onnx_model.graph.input:
    print(f"  Name: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")
    
print("输出信息:")  
for output in onnx_model.graph.output:
    print(f"  Name: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")

# 测试推理
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"实际输入名称: {input_name}")
print(f"实际输出名称: {output_name}")
print(f"实际输出形状: {session.get_outputs()[0].shape}")


# 用零输入测试
test_input = np.zeros((1, 47), dtype=np.float32)
result = session.run([output_name], {input_name: test_input})
print(f"输出形状: {result[0].shape}")


def check_onnx_model(onnx_path):
    """详细检查ONNX模型"""
    
    try:
        session = ort.InferenceSession(onnx_path)
        
        print("=== ONNX模型信息 ===")
        
        # 输入信息
        inputs = session.get_inputs()
        print("输入信息:")
        for i, inp in enumerate(inputs):
            print(f"  [{i}] 名称: {inp.name}")
            print(f"      形状: {inp.shape}")
            print(f"      类型: {inp.type}")
        
        # 输出信息
        outputs = session.get_outputs()
        print("输出信息:")
        for i, out in enumerate(outputs):
            print(f"  [{i}] 名称: {out.name}")
            print(f"      形状: {out.shape}")
            print(f"      类型: {out.type}")
            
        # 测试推理
        if inputs and outputs:
            # 创建测试输入
            input_shape = inputs[0].shape
            test_shape = []

            for dim in input_shape:
                if dim is None or dim == -1 or isinstance(dim, str):
                    test_shape.append(1)
                else:
                    test_shape.append(int(dim))

            print(f"测试输入形状: {test_shape}")
            test_input = np.random.randn(*test_shape).astype(np.float32)

            print(f"\n测试推理:")
            print(f"输入形状: {test_input.shape}")
            
            try:
                result = session.run([outputs[0].name], {inputs[0].name: test_input})
                print(f"输出形状: {result[0].shape}")
                print(f"输出范围: [{result[0].min():.3f}, {result[0].max():.3f}]")
                
                if result[0].size == 0:
                    print("⚠️ 警告: ONNX模型输出为空!")
                else:
                    print("✓ ONNX模型推理正常")
                    
            except Exception as e:
                print(f"❌ ONNX推理失败: {e}")                
            
        return True
        
    except Exception as e:
        print(f"❌ 加载ONNX模型失败: {e}")
        return False

# 检查ONNX模型
check_onnx_model(model_path)
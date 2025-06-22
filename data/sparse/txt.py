import os
import pycolmap

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sparse_path = script_dir  # 直接使用当前目录

# 验证路径是否存在
if not os.path.exists(sparse_path):
    print(f"错误：路径 {sparse_path} 不存在！")
else:
    # 检查所需的.bin文件是否都存在
    required_files = ['cameras.bin', 'images.bin', 'points3D.bin']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(sparse_path, f))]

    if missing_files:
        print(f"错误：缺少必要的文件：{', '.join(missing_files)}")
    else:
        # 加载COLMAP模型
        model = pycolmap.Reconstruction(sparse_path)
        print(f"成功加载模型！相机数量：{len(model.cameras)}，图像数量：{len(model.images)}")

        # 保存为文本格式（可选）
        output_dir = os.path.join(script_dir, "sparse_txt")
        os.makedirs(output_dir, exist_ok=True)
        model.write_text(output_dir)
        print(f"已将模型转换为文本格式，保存至：{output_dir}")
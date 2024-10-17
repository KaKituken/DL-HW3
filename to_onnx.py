import torch
from model import VAE
from models.DCGAN import Discriminator, Generator
import onnx


img_shape = (1, 28, 28)
z_dim = 100  
fearure_dim = 64
dim_factor = [1, 2, 4]
discriminator = Discriminator(img_shape, fearure_dim, dim_factor)
generator = Generator(z_dim, img_shape, fearure_dim, dim_factor)
dummy_input_d = torch.randn(1, *img_shape)
dummy_input_g = torch.randn(1, z_dim, 1, 1)

model_name = "dis.onnx"

torch.onnx.export(
    discriminator,                  # 要导出的 PyTorch 模型
    dummy_input_d,            # 示例输入
    model_name,           # 导出的文件名
    export_params=True,     # 保存模型的训练参数
    opset_version=11,       # ONNX 的 opset 版本，通常选择 11 或更高
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=['input'],  # 输入名，用于标注输入
    output_names=['output'] # 输出名，用于标注输出
)


onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)
print("ONNX has been verified!")

model_name = "gen.onnx"

torch.onnx.export(
    generator,                  # 要导出的 PyTorch 模型
    dummy_input_g,            # 示例输入
    model_name,           # 导出的文件名
    export_params=True,     # 保存模型的训练参数
    opset_version=11,       # ONNX 的 opset 版本，通常选择 11 或更高
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=['input'],  # 输入名，用于标注输入
    output_names=['output'] # 输出名，用于标注输出
)

onnx_model = onnx.load(model_name)
onnx.checker.check_model(onnx_model)

print("ONNX has been verified!")



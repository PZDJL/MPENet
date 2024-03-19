# -*- coding utf-8 -*-
# author： navigation 4052
# function:


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt
from paddle.vision.transforms import Compose, Resize, ToTensor
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']

class Gaussian_laplacian_Pyramid(nn.Layer):
    def __init__(self, levels=4, kernel_size=5):
        super(Gaussian_laplacian_Pyramid, self).__init__()
        self.levels = levels
        self.kernel = kernel_size
        self.gaussian = nn.Conv2D(3, 3, kernel_size=self.kernel, padding=2, bias_attr=False, groups=3)

        gaussian_kernel = self.get_gaussian_kernel()  # 获取高斯核
        self.gaussian.weight.set_value(gaussian_kernel)  # 将高斯核设置为权重
        self.gaussian.weight.stop_gradient = True  # 固定权重

        self.downsample = nn.MaxPool2D(kernel_size=2, stride=2)


    def get_gaussian_kernel(self, sigma=1.0):  #default == 1.0
        kernel_shape = (self.kernel, self.kernel)
        kernel = paddle.zeros(kernel_shape, dtype='float32')
        center = self.kernel // 2
        for i in range(self.kernel):
            for j in range(self.kernel):
                x = i - center
                y = j - center
                kernel[i, j] = paddle.exp(-paddle.to_tensor((x ** 2 + y ** 2) / (2 * sigma ** 2)))
        kernel =kernel / kernel.sum()
        kernel = paddle.to_tensor(kernel, dtype='float32')  # 转换为 PaddlePaddle 张量
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 在两个维度上添加尺寸为1的维度

        # 使用 tile 操作来重复张量
        kernel = paddle.tile(kernel, [3, 1, 1, 1])  # 重复3次，每次沿指定维度重复1次
        return kernel  # Normalize the kernel

    def forward(self, x):

        #构建高斯金字塔
        g_pyramids = []
        current_level = x
        g_pyramids.append(current_level)                           #添加第一个图像为原始图像
        for _ in range(self.levels):
            smoothed = self.gaussian(current_level)
            current_level = self.downsample(smoothed)
            g_pyramids.append(current_level)

        #构建拉普拉斯金字塔
        l_pyramids = []
        for i in range(len(g_pyramids) - 1):
            higher_level = g_pyramids[i]
            lower_level = g_pyramids[i + 1]
            # 上采样操作
            upsampled_lower_level = F.interpolate(lower_level, size=higher_level.shape[2:], mode='bilinear',
                                                  align_corners=False)
            upsampled_lower_level = self.gaussian(upsampled_lower_level)
            laplacian = higher_level - upsampled_lower_level
            l_pyramids.append(laplacian)
        return g_pyramids, l_pyramids




# 对拉普拉斯金字塔进行边缘检测
class EdgeDetection(nn.Layer):
    def __init__(self):
        super(EdgeDetection, self).__init__()
        self.conv_edge = nn.Conv2D(3, 3, kernel_size=1, bias_attr=False)
        self.spatial_attention = SpatialAttention()
        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 3)
        self.spatial_conv = nn.Sequential(
            nn.Conv2D(32, 3, kernel_size=1, bias_attr=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2D(3, 32, kernel_size=1, bias_attr=False)
        )
        self.fusion = nn.Conv2D(6, 3, kernel_size=1)

        self.sobel_x = nn.Conv2D(3, 3, kernel_size=3, padding=1, bias_attr=False, groups=3)
        self.sobel_y = nn.Conv2D(3, 3, kernel_size=3, padding=1, bias_attr=False, groups=3)
        # 使用 paddle.tile 来重复权重张量
        self.sobel_x.weight.data = paddle.tile(paddle.to_tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32').unsqueeze(0).unsqueeze(0), [3, 1, 1, 1])
        self.sobel_y.weight.data = paddle.transpose(self.sobel_x.weight.data, perm=[0, 1, 3, 2])

        # Set requires_grad to False so the parameters are not learned
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False



    def forward(self, level):

        gradient_x = self.sobel_x(level)
        gradient_y = self.sobel_y(level)
        edge_magnitude = gradient_x * 0.5 + gradient_y * 0.5
        res_input = self.res1(level)
        spatial_conv_out = self.res2(self.spatial_conv(self.spatial_attention(res_input))+res_input)
        out_edges = self.fusion(paddle.concat([self.conv_edge(edge_magnitude)+level, spatial_conv_out], axis=1))

        return out_edges


class ConvolutionOnPyramid(nn.Layer):
    def __init__(self, num_levels):
        super(ConvolutionOnPyramid, self).__init__()
        # 卷积核不共享
        self.conv_layers = paddle.nn.LayerList([
            EdgeDetection() for i in range(num_levels)
        ])
        #共享卷积核
        # self.conv_layers = EdgeDetection()
    def forward(self, pyramids):
        convolved_pyramids = []
        for i, level in enumerate(pyramids):
            #卷积核不共享
            convolved_level = self.conv_layers[i](level)
            #共享卷积核
            # convolved_level = self.conv_layers(level)
            convolved_pyramids.append(convolved_level)
        return convolved_pyramids


class ResidualBlock(nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2D(in_features, out_features, 3, padding=1)

        self.block = nn.Sequential(
            nn.Conv2D(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2D(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))



class SpatialAttention(nn.Layer):          #  penent-DEYOLO
    def __init__(self, in_channels=32):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2D(in_channels, 1, kernel_size=1, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class Lowpass_moudle(nn.Layer):
    def __init__(self):
        super(Lowpass_moudle, self).__init__()
        # 共享卷积核
        # self.low_pass = Trans_low()
        # 卷积核不共享
        self.low_pass = paddle.nn.LayerList([
            Trans_low() for i in range(4)
        ])

    def forward(self, pyramids):
        out_low_pass = []
        for i, level in enumerate(pyramids):
            # 共享卷积核
            # low_pass_outputs = self.low_pass(level)
            # 卷积核不共享
            low_pass_outputs = self.low_pass[i](level)
            out_low_pass.append(low_pass_outputs)

        return out_low_pass

class SpatialAttention_Trans(nn.Layer):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        self.conv = nn.Conv2D(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = paddle.mean(x, axis=1, keepdim=True)
        maxout = paddle.max(x, axis=1, keepdim=True)
        attention = paddle.concat([avgout, maxout], axis=1)
        attention = self.conv(attention)

        return self.sigmoid(attention) * x




class Trans_guide(nn.Layer):
    def __init__(self, ch=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2D(6, ch, 3, padding=1),
            nn.LeakyReLU(True),
            SpatialAttention_Trans(3),
            nn.Conv2D(ch, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)


class Trans_low(nn.Layer):
    def __init__(self, ch_blocks=64, ch_mask=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2D(3, 16, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2D(16, ch_blocks, 3, padding=1),
            nn.LeakyReLU(True)
        )

        self.mm1 = nn.Conv2D(ch_blocks, ch_blocks // 4, kernel_size=1, padding=0)
        self.mm2 = nn.Conv2D(ch_blocks, ch_blocks // 4, kernel_size=3, padding=1)
        self.mm3 = nn.Conv2D(ch_blocks, ch_blocks // 4, kernel_size=5, padding=2)
        self.mm4 = nn.Conv2D(ch_blocks, ch_blocks // 4, kernel_size=7, padding=3)


        self.decoder = nn.Sequential(
            nn.Conv2D(ch_blocks, 16, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2D(16, 3, 3, padding=1)
        )
        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = paddle.concat([self.mm1(x1), self.mm2(x1), self.mm3(x1), self.mm4(x1)], axis=1)
        out = self.decoder(x1)+x
        out = F.relu(out)
        return out



class RecoveryModule(nn.Layer):
    def __init__(self):
        super(RecoveryModule, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, pyramids):
        recovered_image = pyramids[-1]  # Start with the highest level
        for level in reversed(range(len(pyramids) - 1)):
            upsampled = self.upsample(recovered_image)
            recovered_image = upsampled + pyramids[level]  # Add the corresponding level
        return recovered_image


class FeatureFusionModule(nn.Layer):
    def __init__(self):
        super(FeatureFusionModule, self).__init__()
        self.edges = ConvolutionOnPyramid(4)
        self.fusion = nn.Conv2D(6, 3, kernel_size=1)

        self.low_pass = Lowpass_moudle()
        self.pyramids = Gaussian_laplacian_Pyramid()
        self.recover = RecoveryModule()


    def forward(self, x):
        fusion = []
        gaussian_pyramids, laplacian_pyramids = self.pyramids(x)
        # out_low_pass = self.low_pass(laplacian_pyramids)
        # out_edges = self.edges(gaussian_pyramids[:-1])
        out_low_pass = self.low_pass(gaussian_pyramids[:-1])
        out_edges = self.edges(laplacian_pyramids)
        for i in range(len(out_edges)):
            fusion.append(self.fusion(paddle.concat([out_low_pass[i], out_edges[i]], axis=1)))

        out_image = self.recover(fusion)

        return out_image

# if __name__ == '__main__':
#     # 设置设备为GPU（如果可用），否则使用CPU
#     device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
#
#     # 定义目标大小
#     target_size = (640, 640)
#
#     # 加载输入图像并进行预处理
#     image_path = r'C:\Users\dd\Desktop\ultralytics-main\img.png'
#     input_image = Image.open(image_path).convert("RGB")
#     transform = Compose([Resize(target_size), ToTensor()])
#     input_data = transform(input_image).unsqueeze(0)
#
#     # 将输入数据移动到所选设备（GPU或CPU）
#     # input_data = input_data.to('gpu')
#
#     # 创建并加载模型
#     model = FeatureFusionModule()  # 请替换为您的模型名称
#     model = model.to(device)
#
#     # 打印模型结构
#     print(model)
#
#     # 进行推断
#     output = model(input_data)
#
#     for i, pyramid_level in enumerate(output):
#         print(f"边缘检测金字塔层 {i}: {pyramid_level.shape}")
#
#         # 使用'transpose'重新排列张量的维度
#         pyramid_image = pyramid_level.squeeze(0).transpose((1, 2, 0)).cpu().numpy()
#         pyramid_image = (pyramid_image - pyramid_image.min()) / (pyramid_image.max() - pyramid_image.min())
#
#         plt.figure()
#         plt.imshow(pyramid_image)
#         plt.title(f"边缘检测金字塔层 {i}")
#         plt.axis('off')
#         plt.show()

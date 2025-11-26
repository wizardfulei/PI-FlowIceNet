import torch
import torch.nn as nn
import torch.nn.functional as F

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLSTMCell(nn.Module):
    """单个 ConvLSTMCell"""

    def __init__(self, input_channels, hidden_channels, kernel_size, bottleneck_ratio=4):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        padding = kernel_size // 2
        bottleneck_channels = max(4, hidden_channels // bottleneck_ratio)

        print(f"创建 ConvLSTMCell: input_channels={input_channels}, hidden_channels={hidden_channels}")
        print(f"  bottleneck 输入通道: {input_channels + hidden_channels}")
        print(f"  bottleneck 输出通道: {bottleneck_channels}")

        self.bottleneck = nn.Conv2d(
            input_channels + hidden_channels,
            bottleneck_channels,
            kernel_size=1
        )
        self.conv = nn.Conv2d(
            bottleneck_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.bottleneck(combined)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class FlowConvLSTM(nn.Module):
    """编码器-解码器 ConvLSTM，输入亮温，只输出光流"""
    def __init__(self, input_channels=1, hidden_channels=[64, 32, 32],
                 kernel_size=3, output_steps=5):
        super(FlowConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_steps = output_steps

        # 编码器 ConvLSTM
        # 关键修改：确保正确传递输入通道数
        self.convlstm1 = ConvLSTMCell(input_channels, hidden_channels[0], kernel_size)
        self.convlstm2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1], kernel_size)
        self.convlstm3 = ConvLSTMCell(hidden_channels[1], hidden_channels[2], kernel_size)

        # 解码器输出头 - 只保留光流输出
        self.output_flow = nn.Sequential(
            nn.Conv2d(hidden_channels[2], hidden_channels[2] // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[2] // 2, 2, 3, padding=1)  # 输出光流 2 通道 (dx, dy)
        )

    def forward(self, x, teacher_forcing_ratio=0.5, output_steps=None):
        """
        x: [B, T_in, C, H, W] 输入亮温
        teacher_forcing_ratio: 保留参数但不使用（为兼容性）
        return: dict {
            "flow": [B, T_out, 2, H, W]
        }
        """
        batch_size, T_in, C, H, W = x.size()

        output_steps = output_steps or self.output_steps

        # 初始化隐藏状态
        h1, c1 = self.init_hidden(batch_size, H, W, self.hidden_channels[0])
        h2, c2 = self.init_hidden(batch_size, H, W, self.hidden_channels[1])
        h3, c3 = self.init_hidden(batch_size, H, W, self.hidden_channels[2])

        # 编码器：处理历史亮温
        for t in range(T_in):
            frame = x[:, t]

            h1, c1 = self.convlstm1(frame, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
            h3, c3 = self.convlstm3(h2, (h3, c3))

        # 解码器：逐步预测
        predictions_flow = []

        # 第一步输入初始化为 0
        last_input = torch.zeros(batch_size, self.input_channels, H, W, device=x.device)

        for t in range(output_steps):
            h1, c1 = self.convlstm1(last_input, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
            h3, c3 = self.convlstm3(h2, (h3, c3))

            pred_flow = torch.tanh(self.output_flow(h3))  # 光流输出范围 [-1,1]
            predictions_flow.append(pred_flow)

            # 构建下一步的输入：保持为零
            last_input = torch.zeros(batch_size, self.input_channels, H, W, device=x.device)

        return {
            "flow": torch.stack(predictions_flow, dim=1)
        }

    def init_hidden(self, batch_size, H, W, hidden_channels):
        return (torch.zeros(batch_size, hidden_channels, H, W, device=device),
                torch.zeros(batch_size, hidden_channels, H, W, device=device))
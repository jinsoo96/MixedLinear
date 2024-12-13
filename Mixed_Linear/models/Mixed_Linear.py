import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """
#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)
#         return x

# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """
#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x)
#         res = x - moving_mean
#         return res, moving_mean
    
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # 패딩을 AvgPool1d에서 처리하도록 변경
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MultiScaleDecomp(nn.Module):
    """
    Multi-scale series decomposition block
    """
    def __init__(self, configs):
        super(MultiScaleDecomp, self).__init__()
        self.decomps = nn.ModuleList(
            [series_decomp(kernel_size=kernel_size) for kernel_size in configs.decomp_kernel_sizes]
        )

    def forward(self, x):
        seasonal_list = []
        trend_list = []
        for decomp in self.decomps:
            seasonal, trend = decomp(x)
            seasonal_list.append(seasonal)
            trend_list.append(trend)
        return seasonal_list, trend_list
    
    
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.decomp_kernel_sizes = configs.decomp_kernel_sizes
        # MultiScaleDecomp 정의
        self.multiscale_decomp = MultiScaleDecomp(configs)

        # MultiScaleDecomp 정의 필요

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.ModuleList(
                    [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
                ))
                self.Linear_Trend.append(nn.ModuleList(
                    [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
                ))
        else:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
            )

    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        batch_size, _, _ = x.size()
        seasonal_list, trend_list = self.multiscale_decomp(x)  # multiscale_decomp 구현 필요

        seasonal_output = torch.zeros((batch_size, self.pred_len, self.channels), 
                                      dtype=x.dtype, device=x.device)
        trend_output = torch.zeros((batch_size, self.pred_len, self.channels), 
                                   dtype=x.dtype, device=x.device)

        for j, (seasonal, trend) in enumerate(zip(seasonal_list, trend_list)):
            if self.individual:
                for i in range(self.channels):
                    s = seasonal[:, :, i].transpose(1, 0)  # [seq_len, batch_size]
                    t = trend[:, :, i].transpose(1, 0)     # [seq_len, batch_size]
                    seasonal_output[:, :, i] += self.Linear_Seasonal[i][j](s).transpose(1, 0)
                    trend_output[:, :, i] += self.Linear_Trend[i][j](t).transpose(1, 0)
            else:
                s = seasonal.transpose(1, 2)  # [batch_size, channels, seq_len]
                t = trend.transpose(1, 2)     # [batch_size, channels, seq_len]
                seasonal_output += self.Linear_Seasonal[j](s).transpose(1, 2)
                trend_output += self.Linear_Trend[j](t).transpose(1, 2)

        x = seasonal_output + trend_output
        return x  # Output shape: [batch_size, pred_len, channels]    
    
    
    
    
    


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.channels = configs.enc_in
#         self.individual = configs.individual
#         self.decomp_kernel_sizes = configs.decomp_kernel_sizes

#         self.multiscale_decomp = MultiScaleDecomp(configs)

#         if self.individual:
#             self.Linear_Seasonal = nn.ModuleList()
#             self.Linear_Trend = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear_Seasonal.append(nn.ModuleList(
#                     [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
#                 ))
#                 self.Linear_Trend.append(nn.ModuleList(
#                     [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
#                 ))
#         else:
#             self.Linear_Seasonal = nn.ModuleList(
#                 [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
#             )
#             self.Linear_Trend = nn.ModuleList(
#                 [nn.Linear(self.seq_len, self.pred_len) for _ in range(len(self.decomp_kernel_sizes))]
#             )

#     def forward(self, x):
#         seasonal_list, trend_list = self.multiscale_decomp(x)

#         seasonal_output = torch.zeros([x.size(0), self.pred_len, self.channels], dtype=x.dtype).to(x.device)
#         trend_output = torch.zeros([x.size(0), self.pred_len, self.channels], dtype=x.dtype).to(x.device)

#         for j, (seasonal, trend) in enumerate(zip(seasonal_list, trend_list)):
#             if self.individual:
#                 for i in range(self.channels):
#                     seasonal_init = seasonal[:, i, :].unsqueeze(1)
#                     trend_init = trend[:, i, :].unsqueeze(1)
#                     seasonal_output[:, :, i] += self.Linear_Seasonal[i][j](seasonal_init).squeeze(1)
#                     trend_output[:, :, i] += self.Linear_Trend[i][j](trend_init).squeeze(1)
#             else:
#                 seasonal_init = seasonal.permute(0, 2, 1)
#                 trend_init = trend.permute(0, 2, 1)
#                 seasonal_output += self.Linear_Seasonal[j](seasonal_init)
#                 trend_output += self.Linear_Trend[j](trend_init)

#         x = seasonal_output + trend_output
        # return x
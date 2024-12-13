import torch
import torch.nn as nn

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MultiScaleDecomp(nn.Module):
    def __init__(self, configs):
        super(MultiScaleDecomp, self).__init__()
        self.decomps = nn.ModuleList(
            [series_decomp(kernel_size=kernel_size) for kernel_size in configs.decomp_kernel_sizes]
        )
        self.pool = nn.AdaptiveAvgPool1d(configs.seq_len)

    def forward(self, x):
        seasonal_list = []
        trend_list = []
        for decomp in self.decomps:
            seasonal, trend = decomp(x)
            seasonal = self.pool(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend = self.pool(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_list.append(seasonal)
            trend_list.append(trend)
            x = trend
        return seasonal_list, trend_list

class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Conv1d(configs.channels, configs.channels, kernel_size=3, padding=1)
                for _ in range(len(configs.decomp_kernel_sizes) - 1)
            ]
        )
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.ConvTranspose1d(configs.channels, configs.channels, kernel_size=3, padding=1)
                for _ in range(len(configs.decomp_kernel_sizes) - 1)
            ]
        )

    def forward(self, seasonal_list):
        out_season_list = [seasonal_list[0]]

        for i in range(len(seasonal_list) - 1):
            out_low_res = self.down_sampling_layers[i](seasonal_list[i].transpose(1, 2))
            out_low_res = self.up_sampling_layers[i](out_low_res).transpose(1, 2)

            if out_low_res.size(1) != seasonal_list[i + 1].size(1):
                out_low_res = out_low_res[:, :seasonal_list[i + 1].size(1), :]

            out_low = seasonal_list[i + 1] + out_low_res
            out_season_list.append(out_low)

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.ConvTranspose1d(configs.channels, configs.channels, kernel_size=3, padding=1)
                for _ in range(len(configs.decomp_kernel_sizes) - 1)
            ]
        )

    def forward(self, trend_list):
        out_trend_list = [trend_list[-1]]

        for i in reversed(range(len(trend_list) - 1)):
            out_high_res = self.up_sampling_layers[i](trend_list[i + 1].transpose(1, 2)).transpose(1, 2)

            if out_high_res.size(1) != trend_list[i].size(1):
                out_high_res = out_high_res[:, :trend_list[i].size(1), :]

            out_high = trend_list[i] + out_high_res
            out_trend_list.insert(0, out_high)

        return out_trend_list

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.decomp_kernel_sizes = configs.decomp_kernel_sizes

        self.multiscale_decomp = MultiScaleDecomp(configs)
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

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
                [nn.Linear(self.seq_len * self.channels, self.pred_len * self.channels) for _ in range(len(self.decomp_kernel_sizes))]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(self.seq_len * self.channels, self.pred_len * self.channels) for _ in range(len(self.decomp_kernel_sizes))]
            )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        seasonal_list, trend_list = self.multiscale_decomp(x)
        
        seasonal_mixed = self.mixing_multi_scale_season(seasonal_list)
        trend_mixed = self.mixing_multi_scale_trend(trend_list)

        seasonal_output = torch.zeros((batch_size, self.pred_len, self.channels), dtype=x.dtype, device=x.device)
        trend_output = torch.zeros((batch_size, self.pred_len, self.channels), dtype=x.dtype, device=x.device)

        for j, (seasonal, trend) in enumerate(zip(seasonal_mixed, trend_mixed)):
            if self.individual:
                for i in range(self.channels):
                    s = seasonal[:, :, i]  # [batch_size, seq_len]
                    t = trend[:, :, i]  # [batch_size, seq_len]
                    seasonal_output[:, :, i] += self.Linear_Seasonal[i][j](s)  # [batch_size, pred_len]
                    trend_output[:, :, i] += self.Linear_Trend[i][j](t)  # [batch_size, pred_len]
            else:
                s = seasonal.reshape(batch_size, -1)  # [batch_size, seq_len * channels]
                t = trend.reshape(batch_size, -1)  # [batch_size, seq_len * channels]
                s_out = self.Linear_Seasonal[j](s).reshape(batch_size, self.pred_len, -1)  # [batch_size, pred_len, channels]
                t_out = self.Linear_Trend[j](t).reshape(batch_size, self.pred_len, -1)  # [batch_size, pred_len, channels]
                seasonal_output += s_out
                trend_output += t_out

        x = seasonal_output + trend_output
        return x
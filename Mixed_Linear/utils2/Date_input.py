import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

class DateFilter:
    def __init__(self, df, date_column='date'):
        self.df = df
        self.date_column = date_column
        self.convert_to_datetime()

    def convert_to_datetime(self):
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])

    def filter_by_date(self, start_date, end_date):
        mask = (self.df[self.date_column] >= start_date) & (self.df[self.date_column] <= end_date)
        return self.df.loc[mask]
    
    def filter_by_date_inclusive(self, start_date, end_date):
        start_datetime = pd.to_datetime(start_date).normalize()
        end_datetime = pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.DateOffset(minutes=5)
        mask = (self.df[self.date_column] >= start_datetime) & (self.df[self.date_column] < end_datetime)
        return self.df.loc[mask]

    def get_data_for_year(self, year):
        return self.filter_by_date(f'{year}-01-01', f'{year}-12-31')

    def get_data_for_half_year(self, year, half):
        if half == 1:
            return self.filter_by_date(f'{year}-01-01', f'{year}-06-30')
        elif half == 2:
            return self.filter_by_date(f'{year}-07-01', f'{year}-12-31')

    def get_data_for_quarter(self, year, quarter):
        if quarter == 1:
            return self.filter_by_date(f'{year}-01-01', f'{year}-03-31')
        elif quarter == 2:
            return self.filter_by_date(f'{year}-04-01', f'{year}-06-30')
        elif quarter == 3:
            return self.filter_by_date(f'{year}-07-01', f'{year}-09-30')
        elif quarter == 4:
            return self.filter_by_date(f'{year}-10-01', f'{year}-12-31')

            return self.filter_by_date(f'{year}-07-01', f'{year}-09-30')
        elif quarter == 4:
            return self.filter_by_date(f'{year}-10-01', f'{year}-12-31')
        
class WeightedAverageCalculator:
    def __init__(self, weight_file, data_file, drop_column):
        # Read weight data
        self.weight_df = pd.read_csv(weight_file)
        self.weights = self.weight_df.drop(columns=[drop_column]).mean()  # Drop column and compute mean

        # MinMax Scaling
        scaler = MinMaxScaler()
        self.weights_scaled = scaler.fit_transform(self.weights.values.reshape(-1, 1)).flatten()

        # Read data
        self.data_df = pd.read_csv(data_file)

        self.locations = ['서울', '인천', '원주', '수원', '청주', '대전', '포항', '대구', '전주', '울산', '창원', '광주', '부산', '여수', '제주', '천안']
        self.attributes = ['기온(°C)', '누적강수량(mm)', '풍향(deg)', '풍속(m/s)', '현지기압(hPa)', '해면기압(hPa)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)']

    def calculate_weighted_average(self):
        for attribute in self.attributes:
            total = pd.Series(0, index=self.data_df.index)
            total_weight = 0
            for i, location in enumerate(self.locations):
                column = location + '_' + attribute
                if column in self.data_df.columns:
                    total += self.data_df[column].fillna(0) * self.weights_scaled[i]
                    total_weight += self.weights_scaled[i]
            self.data_df['전체지역_' + attribute + '_가중평균'] = total / total_weight

        return self.data_df

def apply_pca(data_frame, threshold):
    pca = PCA()
    df_pca = pca.fit_transform(data_frame)
    total_variance = np.cumsum(pca.explained_variance_ratio_)
    n_over_80 = len(total_variance[total_variance >= threshold])
    df_pca = df_pca[:, :n_over_80] 
    result_df = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(n_over_80)])

    return result_df
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


def load_and_agg(path):
    # 1. 读取 CSV，并解析 DateTime
    df = pd.read_csv(
        path,
        encoding='utf-8',
        parse_dates=['DateTime'],
        low_memory=False
    )

    # 2. 定义所有需要做数值聚合的列
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]

    # 3. 强制转换成数值，不可解析的变成 NaN
    # for col in numeric_cols:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    #
    # # 4. 过滤掉任何包含 NaN 的“异常”行
    # mask_valid = df[numeric_cols].notnull().all(axis=1)
    # dropped = len(df) - mask_valid.sum()
    # if dropped > 0:
    #     print(f"过滤掉 {dropped} 条异常记录")
    # df = df.loc[mask_valid].reset_index(drop=True)

    # 5. 构造天级索引
    df['date'] = df['DateTime'].dt.date

    # 6. 按天聚合
    agg_funcs = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first',
    }
    daily = df.groupby('date', as_index=False).agg(agg_funcs)

    return daily

def load_and_agg_new(path):
    # 1. 读取 CSV，并解析 DateTime
    df = pd.read_csv(
        path,
        encoding='utf-8',
        parse_dates=['DateTime'],
        low_memory=False
    )

    # 2. 定义所有需要做数值聚合的列
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]

    # 3. 强制转换成数值，不可解析的变成 NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. 过滤掉任何包含 NaN 的“异常”行
    mask_valid = df[numeric_cols].notnull().all(axis=1)
    dropped = len(df) - mask_valid.sum()
    if dropped > 0:
        print(f"过滤掉 {dropped} 条异常记录")

    # 5. 提取有效数据
    valid_df = df.loc[mask_valid].reset_index(drop=True)

    # 6. 保存有效数据到新 CSV
    valid_df.to_csv(f'new_{path}', index=False, encoding='utf-8')
    print(f"已保存有效数据至 new_{path}")

    # 7. 构造天级索引
    valid_df['date'] = valid_df['DateTime'].dt.date

    # 8. 按天聚合
    agg_funcs = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first',
    }
    daily = valid_df.groupby('date', as_index=False).agg(agg_funcs)

    return daily


class PowerDataset(Dataset):
    def __init__(self, df, out_len, input_win):
        self.out_len = out_len
        self.dates = df['date'].values  # 所有日期顺序
        data = df.drop(columns=['date']).values.astype(float)
        self.scaler = MinMaxScaler()
        data = self.scaler.fit_transform(data)

        Xs, Ys, Dates = [], [], []
        L = len(data)
        for i in range(L - input_win - out_len + 1):
            x = data[i:i + input_win]
            y = data[i + input_win:i + input_win + out_len, 0]  # 目标列

            Xs.append(x)
            Ys.append(y)
            # 预测起点日期 = 输入窗口最后一天 + 1
            predict_start_date = self.dates[i + input_win]
            Dates.append(predict_start_date)

        self.X = torch.from_numpy(np.stack(Xs)).float()
        self.Y = torch.from_numpy(np.stack(Ys)).float()
        self.date_starts = np.array(Dates)  # 保存日期序列

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        date_str = str(self.date_starts[idx])  # 转为 'YYYY-MM-DD' 格式
        return self.X[idx], self.Y[idx], date_str


if __name__ == '__main__':
    load_and_agg_new('test.csv')
    load_and_agg_new('train.csv')
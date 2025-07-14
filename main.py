from dataset import load_and_agg, PowerDataset
from models import LSTMForecaster, TransformerForecaster, AdditiveTransformerUQ, BayesianTransformerQuantile, \
    quantile_loss
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using {DEVICE}')
assert DEVICE.type == 'cuda'
BATCH_SIZE = 512
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 30
INPUT_WINDOW = 90
drawing = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_nll_loss(mean, logvar, target):
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (mean - target) ** 2 * inv_var).mean()


def train_and_eval(train_loader, val_loader, seed, out_len, model_type='lstm', load_model_path=None):
    set_seed(seed)
    if model_type == 'lstm':
        model = LSTMForecaster(
            input_size=train_loader.dataset.X.shape[-1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            out_len=out_len
        )
    elif model_type == 'transformer':
        model = TransformerForecaster(
            input_size=train_loader.dataset.X.shape[-1],
            d_model=HIDDEN_SIZE,
            nhead=8,
            num_layers=NUM_LAYERS,
            out_len=out_len
        )
    elif model_type == 'additive':
        model = AdditiveTransformerUQ(
            input_size=train_loader.dataset.X.shape[-1],
            d_model=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS, out_len=out_len
        ).to(DEVICE)
    elif model_type == 'bayes':
        model = BayesianTransformerQuantile(
            input_size=train_loader.dataset.X.shape[-1],  # 输入特征维度
            d_model=HIDDEN_SIZE,
            nhead=8,
            num_layers=NUM_LAYERS,
            out_len=out_len,  # 预测长度
            quantiles=[0.1, 0.5, 0.9],
            dim_feedforward=256,
            dropout=0.1
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    if load_model_path is not None and os.path.exists(load_model_path):
        model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))
        print(f"Loaded model from {load_model_path}")
    else:
        # 训练阶段
        model.train()
        for epoch in range(EPOCHS):
            for x, y, _ in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if model_type in ['lstm', 'transformer']:
                    pred = model(x)
                    loss = criterion_mse(pred, y)
                elif model_type == 'additive':
                    mean, logvar = model(x)
                    loss = gaussian_nll_loss(mean, logvar, y)
                elif model_type == 'bayes':
                    pred = model(x)
                    loss = quantile_loss(pred, y, model.quantiles)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # --------- 保存模型 ----------
        os.makedirs("models", exist_ok=True)
        save_path = f"models/model_{model_type}_out{out_len}_seed{seed}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    # 验证阶段
    model.eval()
    mses, maes = [], []
    true_list = []
    pred_list = []
    date_list = []  # 新增日期列表
    print(f'start to validate')

    with torch.no_grad():
        for x, y, date_start in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            date_start = pd.to_datetime(date_start)

            if model_type in ['lstm', 'transformer']:
                y_pred = model(x)
                mses.append(criterion_mse(y_pred, y).item())
                maes.append(criterion_mae(y_pred, y).item())
            elif model_type == 'additive':
                mean, _ = model(x)
                y_pred = mean
                mses.append(criterion_mse(mean, y).item())
                maes.append(criterion_mae(mean, y).item())
            elif model_type == 'bayes':
                y_pred = model(x)
                mses.append(criterion_mse(y_pred[:, :, 1], y).item())
                maes.append(criterion_mae(y_pred[:, :, 1], y).item())
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # 收集数据用于绘图

            # 保存用于绘图
            for i in range(y.shape[0]):
                true_list.append(y[i].cpu().numpy().flatten())
                if model_type == 'bayes':
                    pred_list.append(y_pred[i][:, 1].cpu().numpy().flatten())
                else:
                    pred_list.append(y_pred[i].cpu().numpy().flatten())
                date_list.append(date_start[i])

    if not drawing:
        return np.mean(mses), np.mean(maes)
    # 绘图：每个样本单独创建 x 轴长度，避免 shape mismatch
    # 所有样本的 Ground Truth 和预测
    true_array = np.array(true_list)  # shape (N_samples, out_len)
    out_len = true_array.shape[1]

    plt.figure(figsize=(14, 6))

    for i, (yp, start_date) in enumerate(zip(pred_list, date_list)):
        dates = pd.date_range(start=start_date, periods=len(yp), freq='D')
        plt.plot(dates, yp * 1000, color='red', alpha=0.05, label='Prediction' if i == 0 else "")

    val_ds = val_loader.dataset
    all_dates = pd.to_datetime(val_ds.dates).normalize()
    date_index = pd.date_range(start=min(all_dates), end=max(all_dates), freq='D')

    # 构建 date → gt 映射
    gt_map = {}
    for i in range(len(val_ds)):
        _, y, _date = val_ds[i]
        _date = pd.to_datetime(_date)
        for j in range(len(y)):
            d = (_date + pd.Timedelta(days=j)).normalize()
            if d not in gt_map:  # 只保留第一次出现的gt
                gt_map[d] = y[j].item()

    # 构造 full_gt
    dates = sorted(gt_map.keys())
    values = [gt_map[d] for d in dates]

    plt.plot(dates, np.array(values) * 1000, color='blue', linewidth=2, label='Ground Truth')

    plt.xlabel('Date')
    plt.ylabel('Global Active Power (W)')
    plt.title(f'All Samples: {model_type.upper()} Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/pred_vs_gt_{model_type}_out{out_len}_seed{seed}.png", dpi=300)
    plt.close()

    # 取第一个验证样本
    gt = np.array(true_list[0])
    pred = np.array(pred_list[0])
    start_date = pd.to_datetime(date_list[0])
    date_index = pd.date_range(start=start_date, periods=len(gt), freq='D')

    plt.figure(figsize=(10, 5))
    plt.plot(date_index, gt * 1000, label="Ground Truth", color='blue')
    plt.plot(date_index, pred * 1000, label="Prediction", color='red')

    # 设置x轴日期格式
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.xlabel('Date')
    plt.ylabel('Global Active Power (W)')
    plt.title(f'{model_type} Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{model_type}_{out_len}_{seed}_first_pred_vs_gt.png", dpi=300)
    plt.close()

    return np.mean(mses), np.mean(maes)


def main(out_len):
    # 数据加载与 DataLoader 构建
    train_df = load_and_agg('new_train.csv')
    test_df = load_and_agg('new_test.csv')
    train_ds = PowerDataset(train_df, out_len, INPUT_WINDOW)
    test_ds = PowerDataset(test_df, out_len, INPUT_WINDOW)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 多轮实验
    # results = {'lstm': [], 'transformer': [], 'additive': [], 'bayes': []}
    results = {'lstm': [], 'transformer': [], 'bayes': []}

    for model_type in results:
        for seed in [37, 42, 654321, 123456, 888, 666]:
            mse, mae = train_and_eval(train_loader, test_loader, seed, out_len, model_type, load_model_path=f'models/model_{model_type}_out{out_len}.pth')
            print(f"{model_type.upper()} Out={out_len} Seed={seed} → MSE={mse:.4f}, MAE={mae:.4f}")
            results[model_type].append((mse, mae))

        arr = np.array(results[model_type])
        print(f"\n--- Summary {model_type.upper()} out_len={out_len} ---")
        print(f"MSE  mean={arr[:, 0].mean():.4f}, std={arr[:, 0].std():.4f}")
        print(f"MAE  mean={arr[:, 1].mean():.4f}, std={arr[:, 1].std():.4f}\n")


if __name__ == '__main__':
    print("=== Short-term UQ Transformer (90→90) ===")
    main(out_len=90)
    print("\n=== Long-term UQ Transformer (90→365) ===")
    main(out_len=365)

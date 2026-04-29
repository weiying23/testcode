import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('dispersion_data.csv')

plt.figure(figsize=(10, 6))

# 绘制精确解
plt.plot(df['kh'], df['Exact'], 'k--', label='Exact (k*=k)', linewidth=2)

# 绘制各阶格式
colors = ['blue', 'green', 'orange', 'red']
orders = [4, 8, 12, 14]

for i, order in enumerate(orders):
    col_name = f'Order_{order}'
    plt.plot(df['kh'], df[col_name], label=f'{order}-Order Explicit', color=colors[i], alpha=0.8)

plt.xlabel(r'Normalized Wavenumber $k\Delta x$ (rad)')
plt.ylabel(r'Modified Wavenumber $k^*\Delta x$')
plt.title('Dispersion Analysis: Explicit Central Schemes')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlim([0, np.pi])
plt.ylim([0, np.pi])

plt.show()

# 打印分辨率统计
print("\n--- Spectral Resolution (1% Phase Error Limit) ---")
for order in orders:
    col_name = f'Order_{order}'
    # 计算相对误差
    diff = np.abs(df[col_name] - df['Exact'])
    rel_error = diff / df['kh']
    
    # 找到相对误差超过 1% 的第一个点（跳过 kh 太小的点）
    mask = (df['kh'] > 0.1) & (rel_error > 0.01)
    if mask.any():
        limit_kh = df.loc[mask, 'kh'].iloc[0]
    else:
        limit_kh = np.pi
    
    ppw = 2 * np.pi / limit_kh  # points per wave
    print(f"{order}-Order: Accurate up to k*dx = {limit_kh:.4f} (Resolution: {ppw:.2f} points per wave)")

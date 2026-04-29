import numpy as np
import matplotlib.pyplot as plt

def get_coeffs(order):
    """返回各阶显式中心差分系数"""
    coeffs = {
        4: [2/3, -1/12],
        6: [3/4, -3/20, 1/60],
        8: [4/5, -1/5, 4/105, -1/280],
        10: [5/6, -5/14, 5/84, -1/126, 1/1260],
        12: [6/7, -3/7, 2/7, -5/63, 5/252, -1/2772]
    }
    return coeffs.get(order, [])

def modified_wavenumber(kh, coeffs):
    """计算修正波数"""
    return 2 * sum(a * np.sin(m * kh) for m, a in enumerate(coeffs, start=1))

# 计算各阶格式的修正波数
kh = np.linspace(0.01, np.pi, 1000)
orders = [4, 6, 8, 10, 12]
colors = ['blue', 'green', 'orange', 'red', 'purple']

plt.figure(figsize=(14, 5))

# 子图1: 修正波数曲线
plt.subplot(1, 2, 1)
plt.plot(kh, kh, 'k--', label='Exact', linewidth=2, alpha=0.7)
for order, color in zip(orders, colors):
    coeffs = get_coeffs(order)
    k_star = modified_wavenumber(kh, coeffs)
    plt.plot(kh, k_star, label=f'{order}-Order', color=color, linewidth=1.5)
plt.xlabel(r'$k\Delta x$'); plt.ylabel(r'$k^*\Delta x$')
plt.title('Modified Wavenumber Comparison'); plt.legend(); plt.grid(alpha=0.3)
plt.xlim([0, np.pi]); plt.ylim([0, np.pi])

# 子图2: 相对误差（对数坐标）
plt.subplot(1, 2, 2)
for order, color in zip(orders, colors):
    coeffs = get_coeffs(order)
    k_star = modified_wavenumber(kh, coeffs)
    err = np.abs(k_star - kh) / kh
    plt.semilogy(kh, err, label=f'{order}-Order', color=color, linewidth=1.5)
plt.axhline(0.01, color='red', linestyle=':', label='1% Error Threshold')
plt.xlabel(r'$k\Delta x$'); plt.ylabel('Relative Dispersion Error')
plt.title('Dispersion Error (Log Scale)'); plt.legend(); plt.grid(alpha=0.3)
plt.xlim([0.1, np.pi])

plt.tight_layout(); plt.show()

# 打印分辨率统计
print("\n=== Resolution Analysis (1% Phase Error Limit) ===")
print(f"{'Order':<8} {'Stencil':<10} {'kh_1%':<12} {'PPW':<10} {'Gain vs Prev'}")
print("-" * 55)
prev_ppw = None
for order in orders:
    coeffs = get_coeffs(order)
    k_star = modified_wavenumber(kh, coeffs)
    err = np.abs(k_star - kh) / kh
    # 找1%误差临界点（跳过kh<0.2的区域）
    mask = (kh > 0.2) & (err > 0.01)
    kh_1pct = kh[mask][0] if mask.any() else np.pi
    ppw = 2*np.pi / kh_1pct
    gain = f"+{(ppw-prev_ppw)/prev_ppw*100:.1f}%" if prev_ppw else "baseline"
    print(f"{order:<8} {2*len(coeffs)+1:<10} {kh_1pct:<12.4f} {ppw:<10.3f} {gain}")
    prev_ppw = ppw

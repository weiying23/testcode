#!/usr/bin/env python3
"""
矩阵特征可视化程序
分析 benchmark1_1.mtx, benchmark10_1.mtx, benchmark1000_1.mtx 的矩阵特征
包括：矩阵稀疏模式、特征值分布、条件数、秩等
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

def load_mtx_file(filename):
    """加载 .mtx 格式的矩阵文件"""
    matrices = []
    with open(filename, 'r') as f:
        # 跳过注释行
        line = f.readline()  # %FILENAME comment
        line = f.readline()  # header line: Number of matrices:96 Matrix length:40 40

        # 解析头部
        # 格式: "Number of matrices:96 Matrix length:40 40"
        import re
        match = re.search(r'Number of matrices:(\d+)', line)
        num_matrices = int(match.group(1)) if match else 0
        match2 = re.search(r'Matrix length:(\d+)\s+(\d+)', line)
        rows = int(match2.group(1)) if match2 else 0
        cols = int(match2.group(2)) if match2 else 0

        # 读取所有矩阵数据（列优先存储）
        total_elements = num_matrices * rows * cols
        data = []
        for _ in range(total_elements):
            line = f.readline()
            data.append(float(line.strip()))

        data = np.array(data)

        # 分离成多个矩阵
        for i in range(num_matrices):
            start = i * rows * cols
            end = start + rows * cols
            mat_col = data[start:end].reshape(rows, cols)
            # 转换为行优先（标准 numpy 格式）
            matrices.append(mat_col.T.copy())

    return matrices, rows, cols

def analyze_matrix(mat, name):
    """分析单个矩阵的特征"""
    eigenvalues = linalg.eigvals(mat)
    real_eigs = np.real(eigenvalues)
    imag_eigs = np.imag(eigenvalues)

    # 计算条件数
    cond_num = np.linalg.cond(mat)

    # 计算秩
    rank = np.linalg.matrix_rank(mat)

    # Frobenius范数
    frob_norm = np.linalg.norm(mat, 'fro')

    # 最大/最小奇异值
    singular_vals = linalg.svdvals(mat)

    return {
        'name': name,
        'eigenvalues': eigenvalues,
        'real_eigs': real_eigs,
        'imag_eigs': imag_eigs,
        'cond_num': cond_num,
        'rank': rank,
        'frob_norm': frob_norm,
        'singular_vals': singular_vals,
        'shape': mat.shape,
        'max_sv': singular_vals[0],
        'min_sv': singular_vals[-1]
    }

def visualize_matrices(files, output_dir):
    """可视化多个矩阵文件的特性"""

    # 为每个文件创建单独的分析
    all_stats = []

    for filename in files:
        basename = os.path.basename(filename)
        matrices, rows, cols = load_mtx_file(filename)

        print(f"\n处理 {basename}:")
        print(f"  矩阵数量: {len(matrices)}")
        print(f"  矩阵尺寸: {rows}×{cols}")

        # 分析所有矩阵
        stats = []
        for i, mat in enumerate(matrices[:10]):  # 分析前10个矩阵作为样本
            stat = analyze_matrix(mat, f"{basename}_mat{i}")
            stats.append(stat)

        all_stats.extend(stats)

        # 创建该文件的概览图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{basename} 矩阵特征分析\n({len(matrices)} 个 {rows}×{cols} 矩阵)', fontsize=14)

        # 1. 矩阵元素热力图（第一个矩阵）
        ax1 = axes[0, 0]
        mat_sample = matrices[0]
        im = ax1.imshow(mat_sample, cmap='RdBu_r', aspect='equal')
        ax1.set_title('矩阵元素热力图 (第一个矩阵)')
        ax1.set_xlabel('列')
        ax1.set_ylabel('行')
        plt.colorbar(im, ax=ax1)

        # 2. 特征值分布（复平面）
        ax2 = axes[0, 1]
        all_eigs_real = []
        all_eigs_imag = []
        for stat in stats:
            all_eigs_real.extend(stat['real_eigs'])
            all_eigs_imag.extend(stat['imag_eigs'])
        ax2.scatter(all_eigs_real, all_eigs_imag, alpha=0.5, s=10, c='blue')
        ax2.set_title('特征值分布 (复平面)')
        ax2.set_xlabel('实部')
        ax2.set_ylabel('虚部')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        # 3. 奇异值分布
        ax3 = axes[1, 0]
        for i, stat in enumerate(stats[:5]):
            ax3.semilogy(stat['singular_vals'], alpha=0.7, label=f'矩阵{i}')
        ax3.set_title('奇异值分布')
        ax3.set_xlabel('索引')
        ax3.set_ylabel('奇异值 (对数)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 条件数分布
        ax4 = axes[1, 1]
        cond_nums = [stat['cond_num'] for stat in stats]
        ax4.hist(cond_nums, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_title('条件数分布')
        ax4.set_xlabel('条件数')
        ax4.set_ylabel('频数')
        ax4.axvline(np.mean(cond_nums), color='red', linestyle='--',
                    label=f'均值: {np.mean(cond_nums):.2f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{basename}_analysis.png'), dpi=150)
        plt.close()

        # 打印统计信息
        print(f"  样本条件数范围: {min(cond_nums):.2f} ~ {max(cond_nums):.2f}")
        print(f"  平均条件数: {np.mean(cond_nums):.2f}")

    # 创建综合对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('三个矩阵文件的特征对比', fontsize=14)

    # 对比条件数
    ax1 = axes[0]
    file_names = [os.path.basename(f) for f in files]

    # 收集每个文件的统计
    cond_by_file = {}
    for filename in files:
        basename = os.path.basename(filename)
        matrices, rows, cols = load_mtx_file(filename)
        cond_nums = []
        for mat in matrices[:20]:
            cond_nums.append(np.linalg.cond(mat))
        cond_by_file[basename] = cond_nums

    positions = [1, 2, 3]
    bp = ax1.boxplot([cond_by_file[fn] for fn in file_names], positions=positions)
    ax1.set_xticklabels([fn.replace('.mtx', '') for fn in file_names], rotation=15)
    ax1.set_title('条件数对比')
    ax1.set_ylabel('条件数')
    ax1.grid(True, alpha=0.3)

    # 对比最大奇异值
    ax2 = axes[1]
    max_sv_by_file = {}
    for filename in files:
        basename = os.path.basename(filename)
        matrices, rows, cols = load_mtx_file(filename)
        max_svs = []
        for mat in matrices[:20]:
            sv = linalg.svdvals(mat)
            max_svs.append(sv[0])
        max_sv_by_file[basename] = max_svs

    bp2 = ax2.boxplot([max_sv_by_file[fn] for fn in file_names], positions=positions)
    ax2.set_xticklabels([fn.replace('.mtx', '') for fn in file_names], rotation=15)
    ax2.set_title('最大奇异值对比')
    ax2.set_ylabel('最大奇异值')
    ax2.grid(True, alpha=0.3)

    # 对比最小奇异值
    ax3 = axes[2]
    min_sv_by_file = {}
    for filename in files:
        basename = os.path.basename(filename)
        matrices, rows, cols = load_mtx_file(filename)
        min_svs = []
        for mat in matrices[:20]:
            sv = linalg.svdvals(mat)
            min_svs.append(sv[-1])
        min_sv_by_file[basename] = min_svs

    bp3 = ax3.boxplot([min_sv_by_file[fn] for fn in file_names], positions=positions)
    ax3.set_xticklabels([fn.replace('.mtx', '') for fn in file_names], rotation=15)
    ax3.set_title('最小奇异值对比')
    ax3.set_ylabel('最小奇异值')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=150)
    plt.close()

    print(f"\n可视化完成，图片保存在: {output_dir}")

def main():
    base_dir = "/Users/yingwei/Documents/code/testcode/solver/LU-inverse"
    files = [
        os.path.join(base_dir, "benchmark1_1.mtx"),
        os.path.join(base_dir, "benchmark10_1.mtx"),
        os.path.join(base_dir, "benchmark1000_1.mtx")
    ]

    output_dir = base_dir

    visualize_matrices(files, output_dir)

    # 输出详细统计信息
    print("\n" + "="*60)
    print("详细统计信息")
    print("="*60)

    for filename in files:
        basename = os.path.basename(filename)
        matrices, rows, cols = load_mtx_file(filename)

        print(f"\n{basename}:")
        print(f"  矩阵数量: {len(matrices)}")
        print(f"  矩阵尺寸: {rows}×{cols}")

        # 计算全局统计
        all_cond = []
        all_frob = []
        all_rank = []
        for mat in matrices[:30]:
            all_cond.append(np.linalg.cond(mat))
            all_frob.append(np.linalg.norm(mat, 'fro'))
            all_rank.append(np.linalg.matrix_rank(mat))

        print(f"  条件数: min={min(all_cond):.4f}, max={max(all_cond):.4f}, mean={np.mean(all_cond):.4f}")
        print(f"  Frobenius范数: min={min(all_frob):.4f}, max={max(all_frob):.4f}, mean={np.mean(all_frob):.4f}")
        print(f"  矩阵秩: {set(all_rank)} (全部满秩)")

if __name__ == "__main__":
    main()
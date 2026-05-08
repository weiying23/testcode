/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * MTE vs SDMA 跨卡带宽测试 Host 程序
 *
 * 功能说明：
 *   在同节点的两个 NPU 之间测试 MTE 和 SDMA 通信引擎的带宽性能。
 *   通过 Host 端 chrono 计时，避免 profiling 系统的 overflow 问题。
 *
 * 运行方式：
 *   bash run.sh <npu0_id> <npu1_id>
 *   例如: bash run.sh 0 1  (使用 NPU 0 和 NPU 1 测试)
 *
 * 测试流程：
 *   1. 初始化 ACL 和 SHMEM
 *   2. 分配对称内存
 *   3. 运行数据正确性验证（verify）
 *   4. 扫描不同消息大小，测试 MTE 带宽
 *   5. 扫描不同消息大小，测试 SDMA 带宽
 *   6. 输出对比结果
 */

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>    // 用于高精度计时
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "shmem.h"
#include "host/shmem_host_def.h"

// Kernel 启动函数（在 comm_test_kernel.cpp 中定义）
extern "C" void launch_mte_bw(uint32_t, void *, uint8_t *, int64_t, int64_t);
extern "C" void launch_sdma_bw(uint32_t, void *, uint8_t *, int64_t, int64_t);

// 全局 UID 结构（用于 SHMEM 初始化时的进程间通信）
static aclshmemx_uniqueid_t g_uid;

// ============================================================================
// SHMEM 初始化
// ============================================================================
// 设置 SHMEM 运行参数：
//   - my_pe: 本进程的 PE 编号（0 或 1）
//   - n_pes: 总 PE 数量（固定为 2）
//   - local_mem_size: 对称内存大小
//   - engine: 数据操作引擎类型
//
// 注意：这里使用 MTE 引擎初始化（ACLSHMEM_DATA_OP_MTE）
//       因为 SDMA 需要单独的 SIO 链路，而 MTE 可以在 HCCS 上工作
//       实际 kernel 中会分别调用 MTE 和 SDMA API
static bool shmem_init(int pe, int n_pes, const char *ipport)
{
    aclshmemx_init_attr_t attr;
    int ver = (1 << 16) + sizeof(aclshmemx_init_attr_t);

    // 复制 IP 地址到配置结构
    size_t ip_len = std::min(strlen(ipport), (size_t)ACLSHMEM_MAX_IP_PORT_LEN - 1);
    std::copy_n(ipport, ip_len, attr.ip_port);
    attr.ip_port[ip_len] = '\0';

    // 设置 PE 编号和内存大小
    attr.my_pe          = pe;
    attr.n_pes          = n_pes;
    attr.local_mem_size = 512UL * 1024 * 1024;  // 512MB 对称内存

    // 使用 MTE 引擎初始化（SDMA 需要 SIO 链路，MTE 可在 HCCS 上工作）
    attr.option_attr    = {ver, ACLSHMEM_DATA_OP_MTE,
                           DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};

    // 设置 UID（用于进程间通信）
    g_uid.my_pe    = pe;
    g_uid.n_pes    = n_pes;
    attr.comm_args = &g_uid;

    // 禁用配置存储 TLS
    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    // 初始化 SHMEM
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (ret != ACLSHMEM_SUCCESS) {
        std::cerr << "[PE " << pe << "] shmem_init failed: " << ret << "\n";
        return false;
    }
    return true;
}

// ============================================================================
// 测试消息大小配置
// ============================================================================
// 从 4KB 到 16MB 的典型消息大小
static std::vector<size_t> msg_sizes()
{
    return {4096, 16384, 65536, 262144,
            1 << 20, 4 << 20, 8 << 20, 16 << 20};
}

// ============================================================================
// 迭代次数配置
// ============================================================================
// 根据消息大小调整迭代次数（大消息迭代少，小消息迭代多）
static int get_iters(size_t sz)
{
    if (sz <=  65536)  return 1000;   // ≤64KB: 1000 次
    if (sz <= (4<<20)) return 200;    // ≤4MB:  200 次
    return 100;                       // >4MB:  100 次
}

// ============================================================================
// 输出格式化
// ============================================================================
static void print_header(const char *engine)
{
    std::cout << "\n=== " << engine << " ===\n";
    std::cout << std::left
              << std::setw(14) << "MsgSize(B)"
              << std::setw(14) << "Iters"
              << std::setw(18) << "Time/iter(us)"
              << "BW(GB/s)\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout.flush();
}

static void print_row(size_t msg_size, int iters, double time_us, double bw)
{
    std::cout << std::left
              << std::setw(14) << msg_size
              << std::setw(14) << iters
              << std::setw(18) << std::fixed << std::setprecision(2) << time_us
              << std::fixed << std::setprecision(2) << bw << "\n";
    std::cout.flush();
}

// ============================================================================
// 数据正确性验证
// ============================================================================
// 目的：确认数据确实从 Rank 0 传输到了 Rank 1
//
// 方法：
//   1. Rank 0 用 0xAA 填充对称内存
//   2. Rank 1 用 0x55 填充对称内存
//   3. Barrier 同步
//   4. Rank 0 执行一次 PUT（将 0xAA 写到 Rank 1）
//   5. Rank 1 检查自己的内存是否变成了 0xAA
//
// 输出：
//   PASS — 数据正确传输
//   FAIL — 数据未传输（需要排查问题）
static void verify(const char *engine_name, bool use_sdma,
                   int pe, uint8_t *gva, void *stream, uint32_t block_dim)
{
    const size_t verify_sz = 65536;  // 验证用 64KB 数据
    size_t aligned = (verify_sz / block_dim) * block_dim;

    // Rank 0 填充 0xAA，Rank 1 填充 0x55
    uint8_t fill_val = (pe == 0) ? 0xAA : 0x55;
    aclrtMemset(gva, verify_sz, (int32_t)fill_val, verify_sz);
    aclshmem_barrier_all();  // 等待双方都完成填充

    // Rank 0 执行一次 PUT
    if (use_sdma)
        launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, 1);
    else
        launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, 1);
    aclrtSynchronizeStream(stream);  // 等待 kernel 完成

    // Rank 1 验证数据是否正确接收
    if (pe == 1) {
        std::vector<uint8_t> host_buf(aligned, 0);
        aclrtMemcpy(host_buf.data(), aligned, gva, aligned, ACL_MEMCPY_DEVICE_TO_HOST);

        // 检查是否所有字节都是 0xAA（来自 Rank 0）
        bool pass = true;
        for (size_t i = 0; i < aligned; i++) {
            if (host_buf[i] != 0xAA) { pass = false; break; }
        }

        std::cout << "[verify][" << engine_name << "] "
                  << (pass ? "PASS — PE1.gva == 0xAA (数据传输成功)"
                           : "FAIL — PE1.gva != 0xAA (数据未传输)")
                  << "\n";
        std::cout.flush();
    }
    aclshmem_barrier_all();  // 等待验证完成
}

// ============================================================================
// 带宽扫描测试
// ============================================================================
// 对不同的消息大小测试带宽：
//   1. 执行 warmup（预热，不计入测试）
//   2. 记录开始时间
//   3. 执行 iterations 次 PUT
//   4. 记录结束时间
//   5. 计算：time_per_iter = total_time / iterations
//   6. 计算：bandwidth = msg_size / time_per_iter (GB/s)
static void sweep(const char *name, bool use_sdma,
                  int pe, uint8_t *gva, void *stream, uint32_t block_dim)
{
    // 打印表头
    if (pe == 0) print_header(name);

    for (size_t msg_size : msg_sizes()) {
        int iters        = get_iters(msg_size);
        int warmup_iters = 20;  // 预热迭代

        // 对齐消息大小到 block_dim
        size_t aligned = (msg_size / block_dim) * block_dim;
        if (aligned == 0) aligned = block_dim;

        // Lambda: 启动 kernel
        auto launch = [&](int64_t n) {
            if (use_sdma)
                launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, n);
            else
                launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, n);
        };

        // 预热：不计入测试时间
        launch(warmup_iters);
        aclrtSynchronizeStream(stream);

        // ========== 正式测试 ==========
        // 使用 chrono 高精度计时
        auto t0 = std::chrono::high_resolution_clock::now();
        launch(iters);
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        // 只有 Rank 0 输出结果
        if (pe == 0) {
            // 计算时间（微秒）
            double total_us      = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double time_per_iter = total_us / iters;

            // 计算带宽：
            // BW = msg_size(bytes) / time_per_iter(us)
            //    = msg_size / time_per_iter / 1e6 (MB/s)
            //    = msg_size / time_per_iter / 1e9 (GB/s)
            // 简化：BW = msg_size / time_per_iter / 1000 (GB/s)
            double bw = (double)aligned / time_per_iter / 1000.0;

            print_row(aligned, iters, time_per_iter, bw);
        }
    }
    aclshmem_barrier_all();  // 等待双方都完成测试
}

// ============================================================================
// 主函数
// ============================================================================
// 参数：<n_pes> <pe_id> <ipport> <device_id> <peer_device_id>
// 例如：./comm_test 2 0 tcp://127.0.0.1:8898 0 1
int main(int argc, char *argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <device_id> <peer_device_id>\n";
        return 1;
    }

    // 解析参数
    int         n_pes        = std::atoi(argv[1]);   // 总 PE 数（固定为 2）
    int         pe           = std::atoi(argv[2]);   // 本 PE 编号（0 或 1）
    const char *ipport       = argv[3];              // IP 地址和端口
    int         device_id    = std::atoi(argv[4]);   // 本 NPU 设备 ID
    int         peer_device  = std::atoi(argv[5]);   // 对端 NPU 设备 ID
    uint32_t    block_dim    = 32;                   // AI Core 数量

    // ========== ACL 初始化 ==========
    aclInit(nullptr);
    aclrtSetDevice(device_id);

    // ========== P2P 访问设置 ==========
    // 检查并启用 NPU 之间的 P2P（Peer-to-Peer）访问
    // P2P 允许两个 NPU 直接访问对方的内存，无需经过 host
    int can_access = 0;
    aclrtDeviceCanAccessPeer(&can_access, device_id, peer_device);
    std::cout << "[P2P] device " << device_id << " -> " << peer_device
              << ": can_access=" << can_access;
    if (can_access) {
        aclError p2p_err = aclrtDeviceEnablePeerAccess(peer_device, 0);
        std::cout << ", EnablePeerAccess=" << p2p_err;
    }
    std::cout << "\n";
    std::cout.flush();

    // 创建 ACL stream
    void *stream = nullptr;
    aclrtCreateStream(&stream);

    // ========== SHMEM 初始化 ==========
    if (!shmem_init(pe, n_pes, ipport)) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    // ========== 分配对称内存 ==========
    // 对称内存：所有 PE 在相同偏移处有相同大小的内存
    // PE 0 的 gva[0] 和 PE 1 的 gva[0] 是对称的
    // PUT 操作：PE 0 写 PE 1 的 gva，数据直接到达 PE 1 的内存
    uint8_t *gva = (uint8_t *)aclshmem_malloc(16UL << 20);  // 16MB
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed\n";
        aclshmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    // 等待双方都完成初始化
    aclshmem_barrier_all();

    // ========== 运行测试 ==========
    // 同一个 SHMEM session 中运行两种引擎测试

    // 1. MTE 测试
    verify("MTE",  false, pe, gva, stream, block_dim);  // 验证数据传输
    sweep("MTE  (aclshmemx_mte_put_nbi)",  false, pe, gva, stream, block_dim);  // 带宽扫描

    // 2. SDMA 测试
    verify("SDMA", true,  pe, gva, stream, block_dim);  // 验证数据传输
    sweep("SDMA (aclshmemx_sdma_put_nbi)", true,  pe, gva, stream, block_dim);  // 带宽扫描

    // ========== 资源释放 ==========
    aclshmem_free(gva);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    if (pe == 0) std::cout << "\n[DONE]\n";
    return 0;
}
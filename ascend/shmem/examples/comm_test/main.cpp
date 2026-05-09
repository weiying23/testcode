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

// 【为什么需要全局 UID】
// SHMEM 初始化时需要进程间通信来交换 PE 信息：
//   - g_uid: 包含 my_pe 和 n_pes，用于两个进程协商谁是 PE 0，谁是 PE 1
//   - 为什么要全局变量：SHMEM API 要求 comm_args 指针指向的内存必须是全局/静态的
//   - 如果是局部变量：初始化函数返回后，指针指向的内存可能被释放，导致错误
static aclshmemx_uniqueid_t g_uid;

// ============================================================================
// SHMEM 初始化
// ============================================================================
// 【为什么用 MTE 引擎初始化】
// SHMEM 初始化需要指定数据操作引擎：
//   - ACLSHMEM_DATA_OP_MTE：使用 MTE 引擎做底层通信
//   - ACLSHMEM_DATA_OP_SDMA：使用 SDMA 引擎做底层通信
//   - ACLSHMEM_DATA_OP_ROCE：使用 RoCE（RDMA over Ethernet）做跨节点通信
//
// 为什么这里选 MTE？
//   - 原因 1：SDMA 需要单独的 SIO 链路配置，MTE 可以直接在 HCCS 上工作
//   - 原因 2：HCCS 是 Ascend 同节点 NPU 间的高速互连，默认可用
//   - 原因 3：初始化引擎和 kernel 调用的 API 可以不同：
//       - 初始化引擎：建立对称内存映射和通信通道
//       - Kernel API：实际数据搬运时可以选择不同引擎
//
// 即使初始化用 MTE，kernel 中调用 aclshmemx_sdma_put_nbi 仍然使用 SDMA 引擎
static bool shmem_init(int pe, int n_pes, const char *ipport, int engine_type)
{
    aclshmemx_init_attr_t attr;
    int ver = (1 << 16) + sizeof(aclshmemx_init_attr_t);

    size_t ip_len = std::min(strlen(ipport), (size_t)ACLSHMEM_MAX_IP_PORT_LEN - 1);
    std::copy_n(ipport, ip_len, attr.ip_port);
    attr.ip_port[ip_len] = '\0';

    attr.my_pe          = pe;
    attr.n_pes          = n_pes;
    attr.local_mem_size = 512UL * 1024 * 1024;

    attr.option_attr    = {ver, engine_type,
                           DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};

    g_uid.my_pe    = pe;
    g_uid.n_pes    = n_pes;
    attr.comm_args = &g_uid;

    aclshmemx_set_conf_store_tls(false, nullptr, 0);

    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (ret != ACLSHMEM_SUCCESS) {
        std::cerr << "[PE " << pe << "] shmem_init failed (engine=" << engine_type << "): " << ret << "\n";
        return false;
    }
    return true;
}

// ============================================================================
// 测试消息大小配置
// ============================================================================
// 【为什么选择这些大小】
// 消息大小覆盖典型使用场景：
//   - 4KB, 16KB, 64KB：小消息（UB 大小级别，MTE/SDMA 差异明显）
//   - 256KB, 1MB：中等消息（流水线效果开始显现）
//   - 4MB, 8MB, 16MB：大消息（带宽饱和，测峰值吞吐）
//
// 为什么从 4KB 开始？
//   - 太小的消息（如 64B）：单次传输时间太短，计时精度不够
//   - 4KB 是一个合理的起点，可以测出有效数据
static std::vector<size_t> msg_sizes()
{
    return {4096, 16384, 65536, 262144,
            1 << 20, 4 << 20, 8 << 20, 16 << 20};
}

// ============================================================================
// 迭代次数配置
// ============================================================================
// 【为什么根据消息大小调整迭代次数】
// 迭代次数需要平衡测试时间和精度：
//   - 小消息：单次传输快，需要多次迭代才能得到稳定平均值
//   - 大消息：单次传输慢，次数太多会浪费时间
//
// 具体选择：
//   - ≤64KB：1000 次（单次约几微秒，需要大量样本）
//   - ≤4MB：200 次（单次约几百微秒，样本适中）
//   - >4MB：100 次（单次约几毫秒，样本足够）
static int get_iters(size_t sz)
{
    if (sz <=  65536)  return 1000;
    if (sz <= (4<<20)) return 200;
    return 100;
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
// 【为什么需要 verify】
// 带宽测试前必须确认数据确实传输了：
//   - 如果数据没传输：带宽计算会得到 0 或错误值
//   - verify 用小数据（64KB）快速验证通道是否正常
//   - 如果 verify FAIL：说明通信有问题，后续带宽测试不可信
//
// 验证方法：
//   1. Rank 0 填充 0xAA，Rank 1 填充 0x55（不同值以便区分）
//   2. Barrier：确保双方都完成填充
//   3. Rank 0 PUT：把自己的 0xAA 写到 Rank 1 的内存
//   4. Rank 1 检查：自己的内存是否变成了 0xAA
//
// 如果成功：说明 Rank 0 的数据确实到达了 Rank 1
// 如果失败：说明通信通道有问题（可能是初始化错误、P2P 未启用等）
static void verify(const char *engine_name, bool use_sdma,
                   int pe, uint8_t *gva, void *stream, uint32_t block_dim)
{
    const size_t verify_sz = 65536;  // 64KB：足够验证，又不会太慢
    size_t aligned = (verify_sz / block_dim) * block_dim;

    // 【为什么用不同的填充值】
    // 0xAA 和 0x55 是不同的字节值：
    //   - 如果传输成功：Rank 1 的内存从 0x55 变成 0xAA
    //   - 如果传输失败：Rank 1 的内存仍然是 0x55
    //   - 不同值让结果判断简单明确
    uint8_t fill_val = (pe == 0) ? 0xAA : 0x55;
    aclrtMemset(gva, verify_sz, (int32_t)fill_val, verify_sz);
    aclshmem_barrier_all();  // 等待双方都完成填充

    // Rank 0 发送一次
    if (use_sdma)
        launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, 1);
    else
        launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, 1);
    aclrtSynchronizeStream(stream);  // 等待 kernel 完成

    // 【为什么只有 Rank 1 检查】
    // PUT 是单边操作：
    //   - Rank 0 主动发送，它知道数据应该到 Rank 1
    //   - Rank 1 被动接收，只有它能看到自己的内存是否变了
    //   - 如果让 Rank 0 检查，它看不到 Rank 1 的内存
    if (pe == 1) {
        std::vector<uint8_t> host_buf(aligned, 0);
        aclrtMemcpy(host_buf.data(), aligned, gva, aligned, ACL_MEMCPY_DEVICE_TO_HOST);

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
    aclshmem_barrier_all();  // 等待验证完成，再继续
}

// ============================================================================
// 带宽扫描测试
// ============================================================================
// 【为什么用 chrono 计时而不是 profiling】
// Ascend profiling 系统有以下问题：
//   - cycles 累加容易 overflow（int64_t 最大值约 9.2×10^18）
//   - profiling 数据解析复杂，需要处理 block_id、cycle2us 转换等
//   - 之前测试时发现 cycles 显示负值，说明 overflow 问题
//
// chrono 计时的优势：
//   - 高精度：std::chrono::high_resolution_clock 可以达到纳秒级
//   - 简单：直接测 kernel 执行时间，不需要解析 profiling 数据
//   - 可靠：在 host 端测量，不受 device 内存状态影响
//
// 计时方法：
//   1. warmup（预热）：执行若干次，让硬件进入稳定状态
//   2. 记录 t0：测试开始时间
//   3. 执行 N 次传输
//   4. 记录 t1：测试结束时间
//   5. 计算：time_per_iter = (t1 - t0) / N
//   6. 计算：bandwidth = msg_size / time_per_iter
static void sweep(const char *name, bool use_sdma,
                  int pe, uint8_t *gva, void *stream, uint32_t block_dim)
{
    if (pe == 0) print_header(name);

    for (size_t msg_size : msg_sizes()) {
        int iters        = get_iters(msg_size);
        int warmup_iters = 20;

        // 【为什么需要 warmup】
        // 硬件第一次执行时有冷启动开销：
        //   - 缓存预热：数据第一次访问时需要从 DRAM 加载
        //   - DMA 通道初始化：第一次 DMA 可能需要额外时间
        //   - warmup：让硬件执行若干次，进入稳定状态后再正式计时
        //   - 20 次 warmup：足够让硬件稳定，又不会浪费太多时间

        size_t aligned = (msg_size / block_dim) * block_dim;
        if (aligned == 0) aligned = block_dim;

        auto launch = [&](int64_t n) {
            if (use_sdma)
                launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, n);
            else
                launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, n);
        };

        // warmup：不计入测试
        launch(warmup_iters);
        aclrtSynchronizeStream(stream);

        // 【为什么只有 Rank 0 输出结果】
        // 带宽测试测的是发送能力：
        //   - Rank 0 发送，它知道发送了多少数据、花了多少时间
        //   - Rank 1 被动接收，它不知道发送方的 timing
        //   - 只有 Rank 0 能计算出带宽
        auto t0 = std::chrono::high_resolution_clock::now();
        launch(iters);
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (pe == 0) {
            double total_us      = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double time_per_iter = total_us / iters;

            // 【带宽计算公式】
            // BW = msg_size(bytes) / time_per_iter(us)
            //    = msg_size / time_per_iter / 1,000,000 (bytes/s)
            //    = msg_size / time_per_iter / 1,000    (MB/s)
            //    = msg_size / time_per_iter / 1,000    (GB/s，因为 1GB/s = 1000 MB/s)
            //
            // 举例：
            //   msg_size = 1MB = 1048576 bytes
            //   time_per_iter = 1000 us = 1 ms
            //   BW = 1048576 / 1000 / 1000 = 1.05 GB/s
            double bw = (double)aligned / time_per_iter / 1000.0;

            print_row(aligned, iters, time_per_iter, bw);
        }
    }
    aclshmem_barrier_all();
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char *argv[])
{
    // 【参数解析】
    // 参数：<n_pes> <pe_id> <ipport> <device_id> <peer_device_id>
    // 为什么需要 peer_device_id？
    //   - P2P 访问需要知道对端设备 ID
    //   - aclrtDeviceEnablePeerAccess(peer_device) 需要这个参数
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <device_id> <peer_device_id>\n";
        return 1;
    }

    int         n_pes        = std::atoi(argv[1]);
    int         pe           = std::atoi(argv[2]);
    const char *ipport       = argv[3];
    int         device_id    = std::atoi(argv[4]);
    int         peer_device  = std::atoi(argv[5]);
    uint32_t    block_dim    = 32;  // 32 个 AI Core 并行

    // ========== ACL 初始化 ==========
    aclInit(nullptr);
    aclrtSetDevice(device_id);

    // 【为什么需要 P2P 访问设置】
    // P2P (Peer-to-Peer) 访问：两个 NPU 直接访问对方的内存
    //   - 为什么需要：跨卡通信本质就是 P2P 访问
    //   - aclrtDeviceCanAccessPeer：检查硬件是否支持 P2P
    //   - aclrtDeviceEnablePeerAccess：启用 P2P 访问权限
    //
    // 如果不启用 P2P：
    //   - NPU 之间的数据传输需要经过 Host CPU 中转
    //   - 延迟高，带宽低
    //   - SHMEM 的 PUT 操作依赖 P2P
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

    void *stream = nullptr;
    aclrtCreateStream(&stream);

    // ========== 第一轮：MTE 测试（MTE 引擎初始化） ==========
    if (!shmem_init(pe, n_pes, ipport, ACLSHMEM_DATA_OP_MTE)) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    uint8_t *gva = (uint8_t *)aclshmem_malloc(16UL << 20);
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed (MTE)\n";
        aclshmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
    aclshmem_barrier_all();

    verify("MTE",  false, pe, gva, stream, block_dim);
    sweep("MTE  (aclshmemx_mte_put_nbi)",  false, pe, gva, stream, block_dim);

    aclshmem_free(gva);
    aclshmem_finalize();

    // ========== 第二轮：SDMA 测试（SDMA 引擎初始化） ==========
    if (!shmem_init(pe, n_pes, ipport, ACLSHMEM_DATA_OP_SDMA)) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    gva = (uint8_t *)aclshmem_malloc(16UL << 20);
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed (SDMA)\n";
        aclshmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }
    aclshmem_barrier_all();

    verify("SDMA", true,  pe, gva, stream, block_dim);
    sweep("SDMA (aclshmemx_sdma_put_nbi)", true,  pe, gva, stream, block_dim);

    aclshmem_free(gva);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    if (pe == 0) std::cout << "\n[DONE]\n";
    return 0;
}

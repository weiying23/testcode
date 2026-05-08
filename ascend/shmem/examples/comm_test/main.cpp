/**
 * MTE vs SDMA cross-card bandwidth benchmark.
 * Both engines use MTE shmem init (MTE engine works over HCCS).
 * The difference is in the kernel: mte_bw_kernel uses aclshmemx_mte_put_nbi,
 * sdma_bw_kernel uses aclshmemx_sdma_put_nbi. Both run in one shmem session.
 * Args:  <n_pes> <pe_id> <ipport> <device_id>
 * Only Rank 0 prints results.
 */
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "shmem.h"
#include "host/shmem_host_def.h"

extern "C" void launch_mte_bw(uint32_t, void *, uint8_t *, int64_t, int64_t);
extern "C" void launch_sdma_bw(uint32_t, void *, uint8_t *, int64_t, int64_t);

static aclshmemx_uniqueid_t g_uid;

static bool shmem_init(int pe, int n_pes, const char *ipport)
{
    aclshmemx_init_attr_t attr;
    int ver = (1 << 16) + sizeof(aclshmemx_init_attr_t);

    size_t ip_len = std::min(strlen(ipport), (size_t)ACLSHMEM_MAX_IP_PORT_LEN - 1);
    std::copy_n(ipport, ip_len, attr.ip_port);
    attr.ip_port[ip_len] = '\0';

    attr.my_pe          = pe;
    attr.n_pes          = n_pes;
    attr.local_mem_size = 512UL * 1024 * 1024;
    // Always use MTE engine: SDMA_DATA_OP_SDMA requires a separate SIO link,
    // MTE engine works over HCCS which is what both kernels actually use.
    attr.option_attr    = {ver, ACLSHMEM_DATA_OP_MTE,
                           DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    g_uid.my_pe    = pe;
    g_uid.n_pes    = n_pes;
    attr.comm_args = &g_uid;

    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (ret != ACLSHMEM_SUCCESS) {
        std::cerr << "[PE " << pe << "] shmem_init failed: " << ret << "\n";
        return false;
    }
    return true;
}

static std::vector<size_t> msg_sizes()
{
    return {4096, 16384, 65536, 262144,
            1 << 20, 4 << 20, 8 << 20, 16 << 20};
}

static int get_iters(size_t sz)
{
    if (sz <=  65536)  return 1000;
    if (sz <= (4<<20)) return 200;
    return 100;
}

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

static void sweep(const char *name, bool use_sdma,
                  int pe, uint8_t *gva, void *stream, uint32_t block_dim)
{
    if (pe == 0) print_header(name);

    for (size_t msg_size : msg_sizes()) {
        int iters        = get_iters(msg_size);
        int warmup_iters = 20;

        size_t aligned = (msg_size / block_dim) * block_dim;
        if (aligned == 0) aligned = block_dim;

        auto launch = [&](int64_t n) {
            if (use_sdma)
                launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, n);
            else
                launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, n);
        };

        launch(warmup_iters);
        aclrtSynchronizeStream(stream);

        auto t0 = std::chrono::high_resolution_clock::now();
        launch(iters);
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (pe == 0) {
            double total_us      = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double time_per_iter = total_us / iters;
            double bw            = (double)aligned / time_per_iter / 1000.0;
            print_row(aligned, iters, time_per_iter, bw);
        }
    }
    aclshmem_barrier_all();
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <device_id>\n";
        return 1;
    }

    int         n_pes     = std::atoi(argv[1]);
    int         pe        = std::atoi(argv[2]);
    const char *ipport    = argv[3];
    int         device_id = std::atoi(argv[4]);
    uint32_t    block_dim = 32;

    aclInit(nullptr);
    aclrtSetDevice(device_id);

    void *stream = nullptr;
    aclrtCreateStream(&stream);

    if (!shmem_init(pe, n_pes, ipport)) {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    uint8_t *gva = (uint8_t *)aclshmem_malloc(16UL << 20);
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed\n";
        aclshmem_finalize();
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    aclshmem_barrier_all();

    // run both engines back-to-back in the same shmem session
    sweep("MTE  (aclshmemx_mte_put_nbi)",  false, pe, gva, stream, block_dim);
    sweep("SDMA (aclshmemx_sdma_put_nbi)", true,  pe, gva, stream, block_dim);

    aclshmem_free(gva);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    if (pe == 0) std::cout << "\n[DONE]\n";
    return 0;
}

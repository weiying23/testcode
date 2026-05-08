/**
 * MTE vs SDMA cross-card bandwidth benchmark.
 * Args:  <n_pes> <pe_id> <ipport> <device_id> <engine: mte|sdma>
 * One process runs one engine only. run.sh launches them in two separate batches.
 * Only Rank 0 prints results.
 */

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

static void shmem_init(int pe, int n_pes, const char *ipport, bool use_sdma)
{
    aclshmemx_init_attr_t attr;
    int ver = (1 << 16) + sizeof(aclshmemx_init_attr_t);

    size_t ip_len = std::min(strlen(ipport), (size_t)ACLSHMEM_MAX_IP_PORT_LEN - 1);
    std::copy_n(ipport, ip_len, attr.ip_port);
    attr.ip_port[ip_len] = '\0';

    attr.my_pe          = pe;
    attr.n_pes          = n_pes;
    attr.local_mem_size = 512UL * 1024 * 1024;
    attr.option_attr    = {ver,
                           use_sdma ? ACLSHMEM_DATA_OP_SDMA : ACLSHMEM_DATA_OP_MTE,
                           DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    g_uid.my_pe    = pe;
    g_uid.n_pes    = n_pes;
    attr.comm_args = &g_uid;

    aclshmemx_set_conf_store_tls(false, nullptr, 0);
    int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    if (ret != ACLSHMEM_SUCCESS)
        std::cerr << "[PE " << pe << "] shmem_init failed: " << ret << "\n";
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
    std::cout << "\n=== " << engine << " Engine ===\n";
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

static void run_engine(const char *name, bool use_sdma,
                       int pe, int n_pes, const char *ipport,
                       void *stream, uint32_t block_dim)
{
    shmem_init(pe, n_pes, ipport, use_sdma);

    uint8_t *gva = (uint8_t *)aclshmem_malloc(16UL << 20);
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed\n";
        aclshmem_finalize();
        return;
    }

    aclshmem_barrier_all();

    if (pe == 0) print_header(name);

    for (size_t msg_size : msg_sizes()) {
        int iters        = get_iters(msg_size);
        int warmup_iters = 20;

        size_t aligned = (msg_size / block_dim) * block_dim;
        if (aligned == 0) aligned = block_dim;

        // warmup
        if (use_sdma)
            launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, warmup_iters);
        else
            launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, warmup_iters);
        aclrtSynchronizeStream(stream);

        // timed run
        auto t0 = std::chrono::high_resolution_clock::now();
        if (use_sdma)
            launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, iters);
        else
            launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, iters);
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
    aclshmem_free(gva);
    aclshmem_finalize();
}

int main(int argc, char *argv[])
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <device_id> <mte|sdma>\n";
        return 1;
    }

    int         n_pes     = std::atoi(argv[1]);
    int         pe        = std::atoi(argv[2]);
    const char *ipport    = argv[3];
    int         device_id = std::atoi(argv[4]);
    bool        use_sdma  = (strcmp(argv[5], "sdma") == 0);

    aclInit(nullptr);
    aclrtSetDevice(device_id);

    void *stream = nullptr;
    aclrtCreateStream(&stream);

    run_engine(use_sdma ? "SDMA" : "MTE", use_sdma,
               pe, n_pes, ipport, stream, 32);

    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();

    if (pe == 0) std::cout << "\n[DONE]\n";
    return 0;
}

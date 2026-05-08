/**
 * MTE vs SDMA cross-card bandwidth benchmark.
 * Usage: built binary is launched by run.sh -- see run.sh for details.
 * Args:  <n_pes> <pe_id> <ipport> <device_id>
 * Only Rank 0 prints results.
 */

#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "shmem.h"
#include "host/shmem_host_def.h"

extern "C" void launch_mte_bw(uint32_t, void *, uint8_t *, int64_t, int64_t, uint8_t *);
extern "C" void launch_sdma_bw(uint32_t, void *, uint8_t *, int64_t, int64_t, uint8_t *);

static aclshmemx_uniqueid_t g_uid;
static const char *g_ipport;
static int g_device_id;

// NPU @ 1 GHz -> 1 cycle = 1 ns = 0.001 us
static double cycles_to_us(int64_t cycles) { return cycles / 1000.0; }

static void shmem_init(int pe, int n_pes, bool use_sdma)
{
    aclshmemx_init_attr_t attr;
    int ver = (1 << 16) + sizeof(aclshmemx_init_attr_t);

    size_t ip_len = std::min(strlen(g_ipport), (size_t)ACLSHMEM_MAX_IP_PORT_LEN - 1);
    std::copy_n(g_ipport, ip_len, attr.ip_port);
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
              << std::setw(16) << "Time/iter(us)"
              << "BW(GB/s)\n";
    std::cout << std::string(58, '-') << "\n";
}

static void print_row(size_t msg_size, int iters, double time_us, double bw)
{
    std::cout << std::left
              << std::setw(14) << msg_size
              << std::setw(14) << iters
              << std::setw(16) << std::fixed << std::setprecision(2) << time_us
              << std::fixed << std::setprecision(2) << bw << "\n";
}

static void run_engine(const char *name, bool use_sdma,
                       int pe, int n_pes, void *stream, uint32_t block_dim)
{
    shmem_init(pe, n_pes, use_sdma);

    uint8_t *gva = (uint8_t *)aclshmem_malloc(16UL << 20);
    if (!gva) {
        std::cerr << "[PE " << pe << "] aclshmem_malloc failed\n";
        aclshmem_finalize();
        return;
    }

    uint8_t *result_buf = nullptr;
    aclrtMalloc((void **)&result_buf, sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST);

    aclshmem_barrier_all();

    if (pe == 0) print_header(name);

    for (size_t msg_size : msg_sizes()) {
        int iters        = get_iters(msg_size);
        int warmup_iters = 20;

        // align to block_dim so each core gets a clean slice
        size_t aligned = (msg_size / block_dim) * block_dim;
        if (aligned == 0) aligned = block_dim;

        // warmup (result discarded)
        if (use_sdma)
            launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, warmup_iters, result_buf);
        else
            launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, warmup_iters, result_buf);
        aclrtSynchronizeStream(stream);

        // timed run
        if (use_sdma)
            launch_sdma_bw(block_dim, stream, gva, (int64_t)aligned, iters, result_buf);
        else
            launch_mte_bw(block_dim, stream, gva, (int64_t)aligned, iters, result_buf);
        aclrtSynchronizeStream(stream);

        if (pe == 0) {
            int64_t total_cycles = 0;
            aclrtMemcpy(&total_cycles, sizeof(int64_t),
                        result_buf, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);

            double total_us      = cycles_to_us(total_cycles);
            double time_per_iter = total_us / iters;
            double bw            = (double)aligned / time_per_iter / 1000.0;
            print_row(aligned, iters, time_per_iter, bw);
        }
    }

    aclshmem_barrier_all();
    aclrtFree(result_buf);
    aclshmem_free(gva);
    aclshmem_finalize();
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <n_pes> <pe_id> <ipport> <device_id>\n";
        return 1;
    }

    int n_pes   = std::atoi(argv[1]);
    int pe      = std::atoi(argv[2]);
    g_ipport    = argv[3];
    g_device_id = std::atoi(argv[4]);

    uint32_t block_dim = 32;

    aclInit(nullptr);
    aclrtSetDevice(g_device_id);

    void *stream = nullptr;
    aclrtCreateStream(&stream);

    run_engine("MTE",  false, pe, n_pes, stream, block_dim);
    run_engine("SDMA", true,  pe, n_pes, stream, block_dim);

    aclrtDestroyStream(stream);
    aclrtResetDevice(g_device_id);
    aclFinalize();

    if (pe == 0) std::cout << "\n[DONE]\n";
    return 0;
}

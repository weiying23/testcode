/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_OUT_LOGGER_H
#define SHMEM_SHM_OUT_LOGGER_H

#include <ctime>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/stat.h>

#undef inline
#include <iostream>
#define inline __inline__ __attribute__((always_inline))

namespace shm {
    
// 日志文件管理核心常量
constexpr size_t MAX_LOG_FILE_COUNT = 50;                               // 最多保留50个日志文件
constexpr size_t MAX_FILE_NAME_LEN = 255;                               // 文件名最大长度（不含\0）
constexpr uint64_t MAX_FILE_SIZE_THRESHOLD = 1024 * 1024 * 1024;        // 单个日志文件最大1GB
constexpr uint64_t DISK_AVAILABLE_LIMIT = 10 * MAX_FILE_SIZE_THRESHOLD; // 磁盘剩余空间门限10GB
constexpr size_t MAX_ENV_STRING_LEN = 12800;

// 内部辅助函数声明（仅在cpp中使用）
std::string get_home_dir();
bool is_invalid_path(const std::string& path);
std::string normalize_path(const std::string& path);
void make_dir_recursive(const std::string& dir);
bool is_disk_available(const std::string& dir);
bool starts_with(const std::string& str, const std::string& prefix);
bool ends_with(const std::string& str, const std::string& suffix);
bool is_all_digit(const std::string& str);

class log_file_sink {
public:
    log_file_sink();
    ~log_file_sink();
    void write_log(const std::string& log_content);

private:
    void init_log_dir();
    void delete_oldest_files();
    bool is_valid_log_filename(const std::string& filename, std::string& timestamp);
    std::string generate_new_log_path();
    bool open_new_file();
    void close_file();

private:
    std::string shmem_log_dir;
    int shmem_fd = -1;
    uint64_t shmem_current_file_size = 0;
    std::mutex shmem_file_mutex;
};

using external_log = void (*)(int32_t, const char *);

enum log_level : int32_t {
    DEBUG_LEVEL = 0,
    INFO_LEVEL,
    WARN_LEVEL,
    ERROR_LEVEL,
    FATAL_LEVEL,
    BUTT_LEVEL /* no use */
};

class shm_out_logger {
public:
    static shm_out_logger &Instance();
    shmem_error_code_t set_log_level(log_level level);
    void set_extern_log_func(external_log func, bool force_update = false);
    void log(int32_t level, const std::ostringstream &oss);

    shm_out_logger(const shm_out_logger &) = delete;
    shm_out_logger(shm_out_logger &&) = delete;

    shm_out_logger& operator=(const shm_out_logger &) = delete;
    shm_out_logger& operator=(shm_out_logger &&) = delete;

    ~shm_out_logger();

private:
    shm_out_logger();
    std::string build_log_content(int32_t level, const std::ostringstream &oss);
    const std::string &log_level_desc(int32_t level);

private:
    const std::string shmem_log_level_desc[BUTT_LEVEL] = {"debug", "info", "warn", "error", "fatal"};
    log_level shmem_log_level = WARN_LEVEL;
    external_log shmem_log_func = nullptr;
    log_file_sink* shmem_file_sink;
    bool is_log_stdout = false;
};

}  // namespace shm

#ifndef SHM_LOG_FILENAME_SHORT
#define SHM_LOG_FILENAME_SHORT (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#define SHM_OUT_LOG(LEVEL, ARGS)                                                       \
    do {                                                                               \
        std::ostringstream oss;                                                        \
        oss << "[SHM_SHMEM " << SHM_LOG_FILENAME_SHORT << ":" << __LINE__ << "] " << ARGS; \
        shm::shm_out_logger::Instance().log(LEVEL, oss);                               \
    } while (0)

#define SHM_LOG_DEBUG(ARGS) SHM_OUT_LOG(shm::DEBUG_LEVEL, ARGS)
#define SHM_LOG_INFO(ARGS) SHM_OUT_LOG(shm::INFO_LEVEL, ARGS)
#define SHM_LOG_WARN(ARGS) SHM_OUT_LOG(shm::WARN_LEVEL, ARGS)
#define SHM_LOG_ERROR(ARGS) SHM_OUT_LOG(shm::ERROR_LEVEL, ARGS)

#define SHM_ASSERT_RETURN(ARGS, RET)             \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT_RET_VOID(ARGS)                \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return;                              \
        }                                        \
    } while (0)

#define SHM_ASSERT_RETURN_NOLOG(ARGS, RET)       \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT(ARGS)                         \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
        }                                        \
    } while (0)

#define SHM_ASSERT_MULTIPLY_OVERFLOW(A, B, MAX, RET)                           \
    do {                                                                       \
        if ((A) <= 0 || (B) <= 0 || (MAX) <= 0) {                              \
            SHM_LOG_ERROR("INVALID PARAM " << #A << " " << #B << " " << #MAX); \
            return RET;                                                        \
        }                                                                      \
        if ((A) > (MAX) / (B)) {                                               \
            SHM_LOG_ERROR("OVERFLOW " << #A << " * " << #B << " > " << #MAX);  \
            return RET;                                                        \
        }                                                                      \
    } while (0)

#define SHMEM_CHECK_RET(x, ...)                                                                 \
    do {                                                                                        \
        int32_t check_ret = x;                                                                  \
        if (check_ret != 0) {                                                                   \
            if (sizeof(#__VA_ARGS__) > 1) {                                                     \
                SHM_LOG_ERROR(" return shmem error: " << check_ret << " - "                     \
                    << #__VA_ARGS__ << " failed. More error information can be found in plog"); \
            } else {                                                                            \
                SHM_LOG_ERROR(" return shmem error: " << check_ret);                            \
            }                                                                                   \
            return check_ret;                                                                   \
        }                                                                                       \
    } while (0)

#endif  // SHMEM_SHM_OUT_LOGGER_H

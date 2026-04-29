/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "shmemi_host_common.h"
#include "host/shmem_host_def.h"
#include "shmemi_logger.h"
#include <ctime>
#include <climits>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <unistd.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/statvfs.h>
#include <dirent.h>
#include <fcntl.h>
#include <pwd.h>
#include <vector>
#include <iostream>

namespace shm {
static bool get_log_to_stdout_from_env_cfg()
{
    const char *env_log_to_stdout = std::getenv("SHMEM_LOG_TO_STDOUT");
    return env_log_to_stdout != nullptr && strlen(env_log_to_stdout) <= MAX_ENV_STRING_LEN &&
           strcmp(env_log_to_stdout, "1") == 0;
}

log_file_sink::log_file_sink() 
{
    init_log_dir();
}

log_file_sink::~log_file_sink() 
{
    close_file();
}

void log_file_sink::write_log(const std::string& log_content) 
{
    std::lock_guard<std::mutex> lock(shmem_file_mutex);

    if (shmem_current_file_size + log_content.size() >= MAX_FILE_SIZE_THRESHOLD) {
        close_file();
    }

    if (shmem_fd < 0) {
        if (!open_new_file()) {
            return;
        }
    }

    ssize_t write_len = write(shmem_fd, log_content.c_str(), log_content.size());
    if (write_len != static_cast<ssize_t>(log_content.size())) {
        std::cout << "shmem_log: write file fail, want: " << log_content.size() <<
                    ", actual: " << write_len << std::endl;
        close_file();
        return;
    }

    shmem_current_file_size += write_len;
}


void log_file_sink::init_log_dir()
{
    std::string log_root = get_home_dir();
    log_root = log_root.empty() ? "/tmp" : log_root;
    log_root += "/shmem/log";

    const char* env_log_path = getenv("SHMEM_LOG_PATH");
    if (env_log_path != nullptr && strlen(env_log_path) <= MAX_ENV_STRING_LEN && !is_invalid_path(env_log_path)) {
        log_root = env_log_path;
    }

    shmem_log_dir = normalize_path(log_root);
    make_dir_recursive(shmem_log_dir);
}

void log_file_sink::delete_oldest_files()
{
    std::vector<std::pair<std::string, std::string>> log_files;
    DIR* dir = opendir(shmem_log_dir.c_str());
    if (!dir) {
        return;
    }

    struct dirent* ptr = nullptr;
    while ((ptr = readdir(dir)) != nullptr) {
        if (ptr->d_name[0] == '.') {
            continue;
        }

        std::string filename = ptr->d_name;
        std::string timestamp;
        if (is_valid_log_filename(filename, timestamp)) {
            log_files.emplace_back(shmem_log_dir + "/" + filename, timestamp);
        }
    }
    closedir(dir);

    std::sort(log_files.begin(), log_files.end(),
        [](const std::pair<std::string, std::string>& a, const std::pair<std::string, std::string>& b) {
            return a.second < b.second;
        });

    if (log_files.size() > MAX_LOG_FILE_COUNT) {
        size_t delete_count = log_files.size() - MAX_LOG_FILE_COUNT;
        for (size_t i = 0; i < delete_count; ++i) {
            std::cout << "shmem_log: delete old log: " << log_files[i].first << std::endl;
            remove(log_files[i].first.c_str());
        }
    }
}

bool log_file_sink::is_valid_log_filename(const std::string& filename, std::string& timestamp)
{
    const std::string prefix = "shmem_";
    const std::string suffix = ".log";
    if (!starts_with(filename, prefix) || !ends_with(filename, suffix)) {
        return false;
    }

    size_t sub_len = filename.size() - prefix.size() - suffix.size();
    std::string sub_str = filename.substr(prefix.size(), sub_len);
    size_t sep_pos = sub_str.find('_');
    if (sep_pos == std::string::npos) {
        return false;
    }

    std::string pid_str = sub_str.substr(0, sep_pos);
    timestamp = sub_str.substr(sep_pos + 1);
    if (!is_all_digit(pid_str) || !is_all_digit(timestamp)) {
        return false;
    }
    return true;
}

std::string log_file_sink::generate_new_log_path()
{
    time_t now = time(nullptr);
    struct tm local_tm {};
    localtime_r(&now, &local_tm);
    
    char time_buf[32] = {0};
    strftime(time_buf, sizeof(time_buf), "%Y%m%d%H%M%S", &local_tm);

    std::ostringstream oss;
    oss << shmem_log_dir << "/shmem_" << getpid() << "_" << time_buf << ".log";
    return oss.str();
}

bool log_file_sink::open_new_file()
{
    if (!is_disk_available(shmem_log_dir)) {
        return false;
    }
    delete_oldest_files();

    std::string log_path = generate_new_log_path();
    shmem_fd = open(log_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
    if (shmem_fd < 0) {
        std::cout << "shmem_log: open file fail: " << log_path << std::endl;
        return false;
    }

    shmem_current_file_size = 0;
    return true;
}

void log_file_sink::close_file()
{
    if (shmem_fd > 0) {
        fchmod(shmem_fd, S_IRUSR | S_IRGRP);
        close(shmem_fd);
        shmem_fd = -1;
        shmem_current_file_size = 0;
    }
}

std::string get_home_dir()
{
    int bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (bufsize == -1) {
        return "";
    }

    char buffer[bufsize];
    memset(buffer, 0, bufsize); 
    struct passwd pwd;
    struct passwd* result = nullptr;
    if (getpwuid_r(getuid(), &pwd, buffer, bufsize, &result) != 0 || !result) {
        return "";
    }
    return std::string(pwd.pw_dir);
}

bool is_invalid_path(const std::string& path)
{
    if (path.empty() || path.size() >= PATH_MAX) {
        return true;
    }
    if (path.find("..") != std::string::npos) {
        return true;
    }
    return false;
}

std::string normalize_path(const std::string& path)
{
    if (path.empty()) {
        return "";
    }
    size_t last_non_slash = path.find_last_not_of("/");
    if (last_non_slash == std::string::npos) {
        return "/";
    }
    if (last_non_slash != path.size() - 1) {
        return path.substr(0, last_non_slash + 1);
    }
    return path;
}

void make_dir_recursive(const std::string& dir)
{
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return;
    }

    mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP;
    size_t offset = 0;
    size_t dir_len = dir.size();
    do {
        const char* slash = strchr(dir.c_str() + offset, '/');
        offset = (slash == nullptr) ? dir_len : (slash - dir.c_str() + 1);
        std::string sub_dir = dir.substr(0, offset);
        if (sub_dir.empty()) {
            continue;
        }
        if (stat(sub_dir.c_str(), &st) != 0) {
            if (mkdir(sub_dir.c_str(), mode) != 0) {
                std::cout << "shmem_log: mkdir fail: " << sub_dir << std::endl;
                return;
            }
        }
    } while (offset != dir_len);
}

bool is_disk_available(const std::string& dir)
{
    struct statvfs vfs;
    if (statvfs(dir.c_str(), &vfs) == -1) {
        std::cout << "shmem_log: get disk stat fail" << std::endl;
        return false;
    }

    uint64_t available = static_cast<uint64_t>(vfs.f_bsize) * vfs.f_bfree;
    if (available <= DISK_AVAILABLE_LIMIT) {
        std::cout << "shmem_log: disk space low, avail: " << available 
                  << ", limit: " << DISK_AVAILABLE_LIMIT << std::endl;
        return false;
    }
    return true;
}

bool starts_with(const std::string& str, const std::string& prefix)
{
    if (str.size() < prefix.size()) {
        return false;
    }
    return str.substr(0, prefix.size()) == prefix;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.size() < suffix.size()) {
        return false;
    }
    return str.substr(str.size() - suffix.size()) == suffix;
}

bool is_all_digit(const std::string& str)
{
    return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
}

shm_out_logger& shm_out_logger::Instance()
{
    static shm_out_logger g_logger;
    return g_logger;
}

shmem_error_code_t shm_out_logger::set_log_level(log_level level)
{
    if (level < DEBUG_LEVEL || level >= BUTT_LEVEL) {
        return SHMEM_INVALID_VALUE;
    }
    shmem_log_level = level;
    return SHMEM_SUCCESS;
}

void shm_out_logger::set_extern_log_func(external_log func, bool force_update)
{
    if (shmem_log_func == nullptr || force_update) {
        shmem_log_func = func;
    }
}

void shm_out_logger::log(int32_t level, const std::ostringstream &oss)
{
    if (shmem_log_func != nullptr) {
        shmem_log_func(level, oss.str().c_str());
        return;
    }

    if (level < shmem_log_level) {
        return;
    }

    std::string log_content = build_log_content(level, oss);
    if (is_log_stdout){
        std::cout << log_content;
    } else if (shmem_file_sink) {
        shmem_file_sink->write_log(log_content);
    }
}

void shmem_log_bridge(int level, const char* log_msg)
{
    if (log_msg == nullptr) {
        return;
    }
    std::ostringstream oss;
    oss << log_msg;
    shm::shm_out_logger::Instance().log(static_cast<int32_t>(level), oss);
}

shm_out_logger::shm_out_logger()
{
    is_log_stdout = get_log_to_stdout_from_env_cfg();
    if (!is_log_stdout) {
        shmem_file_sink = new (std::nothrow) log_file_sink();
        if (shmem_file_sink == nullptr) { 
        std::cout << "New log_file_sink failed, logs cannot be stored in files." << std::endl; 
        }   
    }
    
    smem_set_extern_logger(shmem_log_bridge);
}

shm_out_logger::~shm_out_logger()
{
    shmem_log_func = nullptr;
    if (shmem_file_sink) {
        delete shmem_file_sink;
        shmem_file_sink = nullptr;
    }
}

std::string shm_out_logger::build_log_content(int32_t level, const std::ostringstream &oss)
{
    struct timeval tv {};
    char str_time[24] = {0};
    std::ostringstream log_oss;

    gettimeofday(&tv, nullptr);
    time_t time_stamp = tv.tv_sec;
    struct tm local_time {};
    
    if (strftime(str_time, sizeof str_time, "%Y-%m-%d %H:%M:%S.", localtime_r(&time_stamp, &local_time)) != 0) {
        log_oss << str_time << std::setw(6U) << std::setfill('0') << tv.tv_usec
                << " " << log_level_desc(level) << " " << syscall(SYS_gettid)
                << " pid[" << getpid() << "] " << oss.str() << std::endl;
    } else {
        log_oss << " Invalid time " << log_level_desc(level) << " " << syscall(SYS_gettid)
                << " pid[" << getpid() << "] " << oss.str() << std::endl;
    }

    return log_oss.str();
}

const std::string &shm_out_logger::log_level_desc(int32_t level)
{
    static std::string invalid = "invalid";
    if (level < DEBUG_LEVEL || level >= BUTT_LEVEL) {
        return invalid;
    }
    return shmem_log_level_desc[level];
}

} // namespace shm
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
    std::size_t bytes = 1ULL << 30;
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    int warmup = 2;
    int iters = 10;
    std::string mode = "read";
};

class Barrier {
public:
    explicit Barrier(int parties) : parties_(parties) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto generation = generation_;
        if (++arrived_ == parties_) {
            arrived_ = 0;
            ++generation_;
            lock.unlock();
            condition_.notify_all();
        } else {
            condition_.wait(lock, [&] { return generation != generation_; });
        }
    }

private:
    int parties_;
    int arrived_ = 0;
    unsigned int generation_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

std::uint64_t read_bytes(const std::uint64_t* data, std::size_t count) {
    std::uint64_t checksum = 0;
    for (std::size_t index = 0; index < count; ++index) {
        checksum += data[index];
    }
    return checksum;
}

void write_bytes(std::uint64_t* data, std::size_t count, std::uint64_t seed) {
    for (std::size_t index = 0; index < count; ++index) {
        data[index] = seed + static_cast<std::uint64_t>(index);
    }
}

void copy_bytes(const std::uint64_t* source, std::uint64_t* target,
                std::size_t count) {
    std::memcpy(target, source, count * sizeof(*source));
}

std::uint64_t worker(std::size_t work_bytes, std::uint64_t* source,
                     std::uint64_t* target, int thread_id, int warmup,
                     int iters, const std::string& mode, Barrier& barrier) {
    const std::size_t count = work_bytes / sizeof(std::uint64_t);
    const std::size_t offset = static_cast<std::size_t>(thread_id) * count;
    const std::uint64_t seed = static_cast<std::uint64_t>(thread_id) + 1;
    std::uint64_t checksum = 0;

    for (int iteration = 0; iteration < warmup; ++iteration) {
        if (mode == "read") {
            checksum += read_bytes(source + offset, count);
        } else if (mode == "write") {
            write_bytes(target + offset, count, seed);
        } else {
            copy_bytes(source + offset, target + offset, count);
        }
    }

    barrier.arrive_and_wait();
    for (int iteration = 0; iteration < iters; ++iteration) {
        if (mode == "read") {
            checksum += read_bytes(source + offset, count);
        } else if (mode == "write") {
            write_bytes(target + offset, count, seed);
        } else {
            copy_bytes(source + offset, target + offset, count);
        }
    }
    return checksum;
}

double gib_per_second(double bytes, double seconds) {
    return bytes / (seconds * 1024.0 * 1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string value = argv[arg];
        if (value == "--mode" && arg + 1 < argc) {
            args.mode = argv[++arg];
        } else if (value == "--threads" && arg + 1 < argc) {
            args.threads = std::atoi(argv[++arg]);
        } else if (value == "--size" && arg + 1 < argc) {
            args.bytes = static_cast<std::size_t>(std::atoll(argv[++arg]));
        } else if (value == "--warmup" && arg + 1 < argc) {
            args.warmup = std::atoi(argv[++arg]);
        } else if (value == "--iters" && arg + 1 < argc) {
            args.iters = std::atoi(argv[++arg]);
        } else if (value == "--help") {
            std::printf(
                "usage: host_mem_bw [--mode read|write|copy] [--threads N] "
                "[--size BYTES] [--warmup N] [--iters N]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[arg]);
            return 2;
        }
    }

    if ((args.mode != "read" && args.mode != "write" && args.mode != "copy") ||
        args.bytes % (sizeof(std::uint64_t) * static_cast<std::size_t>(args.threads)) != 0 ||
        args.threads <= 0 || args.warmup < 0 || args.iters <= 0) {
        std::fprintf(stderr, "invalid arguments\n");
        return 2;
    }

    const std::size_t per_thread_bytes =
        args.bytes / static_cast<std::size_t>(args.threads);
    std::vector<std::uint64_t> source(args.bytes / sizeof(std::uint64_t));
    std::vector<std::uint64_t> target(args.bytes / sizeof(std::uint64_t));
    std::iota(source.begin(), source.end(), 0);
    std::fill(target.begin(), target.end(), 0);

    Barrier barrier(args.threads + 1);
    std::vector<std::thread> threads;
    for (int thread_id = 0; thread_id < args.threads; ++thread_id) {
        threads.emplace_back(worker, per_thread_bytes, source.data(),
                             target.data(), thread_id, args.warmup,
                             args.iters, args.mode, std::ref(barrier));
    }

    barrier.arrive_and_wait();
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t checksum = 0;
    for (auto& thread : threads) thread.join();
    const auto ended = std::chrono::steady_clock::now();
    checksum ^= static_cast<std::uint64_t>(threads.size());

    const double elapsed =
        std::chrono::duration<double>(ended - started).count();
    const double total_bytes =
        static_cast<double>(args.bytes) * args.iters;
    std::printf(
        "mode=%s threads=%d bytes_per_iter=%zu iters=%d "
        "bytes_per_thread=%zu total_bytes=%zu elapsed=%.6fs "
        "aggregate_bw=%.3f GiB/s\n",
        args.mode.c_str(), args.threads, args.bytes, args.iters,
        per_thread_bytes, static_cast<std::size_t>(total_bytes), elapsed,
        gib_per_second(total_bytes, elapsed));
    return 0;
}

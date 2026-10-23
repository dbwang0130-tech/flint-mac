// SSD 随机读测速：绕过页缓存（F_NOCACHE），从大文件随机位置读固定大小的块，多线程并发。
// 用法: ssdbench <file> <chunk_kb> <threads> <seconds> [range_gb]
// range_gb：只在文件前 range_gb GB 内随机读（避开仍在页缓存里的部分）
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *path;
static size_t chunk;
static off_t file_size;
static double seconds;
static atomic_ullong total_bytes, total_reads;
static atomic_ullong lat_sum_us;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    unsigned seed = (unsigned)(uintptr_t)arg * 2654435761u;
    int fd = open(path, O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);
    void *buf;
    posix_memalign(&buf, 16384, chunk);
    const double t_end = now() + seconds;
    while (now() < t_end) {
        off_t blocks = (file_size - chunk) / 16384;
        off_t off = (off_t)(((uint64_t)rand_r(&seed) << 31 | rand_r(&seed)) % blocks) * 16384;
        double t0 = now();
        ssize_t r = pread(fd, buf, chunk, off);
        double t1 = now();
        if (r > 0) {
            atomic_fetch_add(&total_bytes, (unsigned long long)r);
            atomic_fetch_add(&total_reads, 1);
            atomic_fetch_add(&lat_sum_us, (unsigned long long)((t1 - t0) * 1e6));
        }
    }
    free(buf);
    close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s file chunk_kb threads seconds\n", argv[0]);
        return 1;
    }
    path = argv[1];
    chunk = (size_t)atoi(argv[2]) * 1024;
    int threads = atoi(argv[3]);
    seconds = atof(argv[4]);
    struct stat st;
    stat(path, &st);
    file_size = st.st_size;
    if (argc > 5) {
        off_t lim = (off_t)(atof(argv[5]) * 1e9);
        if (lim < file_size) file_size = lim;
    }
    pthread_t th[64];
    double t0 = now();
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, (void *)(uintptr_t)(i + 1));
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;
    unsigned long long n = total_reads;
    printf("chunk %5zu KB x %2d threads: %6.2f GB/s, %7.0f reads/s, avg latency %6.0f us\n", chunk / 1024, threads,
           total_bytes / dt / 1e9, n / dt, n ? (double)lat_sum_us / n : 0.0);
    return 0;
}

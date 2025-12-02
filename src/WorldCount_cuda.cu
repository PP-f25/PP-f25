#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// ==================== Constants ====================
#define MAX_WORD_LENGTH 32  // 單詞最大長度
#define BLOCK_SIZE 256
#define NUM_STOPWORDS 200  // 增加以包含所有 stopwords (實際 198 個)
#define MAX_STOPWORD_LEN 16
#define HASH_TABLE_SIZE 32768  // Global memory hash table 大小 (32768 slots，減少碰撞)
#define EMPTY_KEY UINT64_MAX  // 空槽位標記
#define LOCAL_HASH_SIZE 64  // 每個thread的local hash table (最佳平衡：性能 vs 正確性)

// Stopwords array (same as CPU version)
const char *stopwords[] = {"a", "about", 
    "above", "after", "again", "against", "ain",
    "all", "am", "an", "and", "any", "are", "aren", 
    "aren't", "as", "at", "be", "because", "been", 
    "before", "being", "below", "between", "both", 
    "but", "by", "can", "couldn", "couldn't", "d",
    "did", "didn", "didn't", "do", "does", "doesn", 
    "doesn't", "doing", "don", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn",
    "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
    "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
    "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
    "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
    "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me", 
    "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
    "myself", "needn", "needn't","no", "nor", "not", "now", "o", "of", 
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", 
    "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn", "shouldn't", "should've", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
    "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
    "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've", NULL};

const char *dataset_path[] = {
    // "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt"
    "../datasets/text8"
};

// ==================== Device Constant Memory ====================
__constant__ char d_stopwords[NUM_STOPWORDS][MAX_STOPWORD_LEN];
__constant__ bool d_delimiter_table[256];  // 分隔符查找表 (O(1) 查詢)

// ==================== Data Structures ====================
struct WordCount {
    char word[MAX_WORD_LENGTH];
    int count;
    
    __host__ __device__
    bool operator<(const WordCount& other) const {
        int cmp = 0;
        for (int i = 0; i < MAX_WORD_LENGTH; i++) {
            if (word[i] != other.word[i]) {
                cmp = word[i] - other.word[i];
                break;
            }
            if (word[i] == '\0') break;
        }
        return cmp < 0;
    }
    
    __host__ __device__
    bool operator==(const WordCount& other) const {
        for (int i = 0; i < MAX_WORD_LENGTH; i++) {
            if (word[i] != other.word[i]) return false;
            if (word[i] == '\0') break;
        }
        return true;
    }
};

// Hash Table Entry for GPU
struct HashEntry {
    uint64_t key_hash;  // word 的 hash 值
    char word[MAX_WORD_LENGTH];  // 實際單詞（用於驗證）
    int count;          // 出現次數
};

struct WordEqual {
    __host__ __device__
    bool operator()(const WordCount& a, const WordCount& b) const {
        for (int i = 0; i < MAX_WORD_LENGTH; i++) {
            if (a.word[i] != b.word[i]) return false;
            if (a.word[i] == '\0') return true;
        }
        return true;
    }
};

struct WordCountAdd {
    __host__ __device__
    WordCount operator()(const WordCount& a, const WordCount& b) const {
        WordCount result = a;
        result.count += b.count;
        return result;
    }
};

// ==================== Error Checking Macro ====================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== Device Functions ====================

// MurmurHash3 (簡化版，避免對齊問題)
__device__ inline uint32_t murmur3_32_simple(const char* key, int len) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    
    uint32_t hash = 0x42424242;  // seed
    
    // Process each byte
    for (int i = 0; i < len; i++) {
        uint32_t k = (uint32_t)(unsigned char)key[i];
        k *= c1;
        k = (k << 15) | (k >> 17);
        k *= c2;
        hash ^= k;
        hash = ((hash << 13) | (hash >> 19)) * 5 + 0xe6546b64;
    }
    
    // Finalization
    hash ^= len;
    hash ^= (hash >> 16);
    hash *= 0x85ebca6b;
    hash ^= (hash >> 13);
    hash *= 0xc2b2ae35;
    hash ^= (hash >> 16);
    
    return hash;
}

// 使用 MurmurHash3 作為主要 hash 函數
__device__ uint64_t compute_hash(const char* word, int len) {
    return (uint64_t)murmur3_32_simple(word, len);
}

// 字串比較（用於驗證 hash 碰撞）
__device__ bool str_equal_device(const char* a, const char* b) {
    for (int i = 0; i < MAX_WORD_LENGTH; i++) {
        if (a[i] != b[i]) return false;
        if (a[i] == '\0') return true;
    }
    return true;
}

// 分隔符檢查 (基礎版本，用於部分 kernel)
__device__ bool is_delimiter(char c) {
    const char* delims = " \t\n\r.,;:!?\"()[]{}\\<>-'";
    for (int i = 0; delims[i] != '\0'; i++) {
        if (c == delims[i]) return true;
    }
    return false;
}

// 快速分隔符檢查 (使用查找表，O(1) 時間複雜度)
__device__ inline bool is_delimiter_fast(char c) {
    return d_delimiter_table[(unsigned char)c];
}

__device__ void to_lowercase(char* dst, const char* src, int len) {
    for (int i = 0; i < len; i++) {
        if (src[i] >= 'A' && src[i] <= 'Z') {
            dst[i] = src[i] + 32;
        } else {
            dst[i] = src[i];
        }
    }
    dst[len] = '\0';
}

__device__ int str_len(const char* str) {
    int len = 0;
    while (str[len] != '\0' && len < MAX_WORD_LENGTH) len++;
    return len;
}

__device__ bool str_equal(const char* a, const char* b) {
    for (int i = 0; i < MAX_STOPWORD_LEN; i++) {
        if (a[i] != b[i]) return false;
        if (a[i] == '\0') return true;
    }
    return true;
}

// 修正：與 CPU 版本一致的 valid word 檢查
__device__ bool is_valid_word(const char* word, int len) {
    if (len == 0) return false;
    
    for (int i = 0; i < len; i++) {
        char c = word[i];
        bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        bool is_digit = (c >= '0' && c <= '9');
        bool is_underscore = (c == '_');
        
        if (!is_alpha && !is_digit && !is_underscore) {
            return false;
        }
    }
    return true;
}

// 修正：與 CPU 版本一致的 stopword 檢查
__device__ bool is_stopword_gpu(const char* word, int len) {
    // 長度 <= 2 的單詞視為 stopword
    if (len <= 2) return true;
    
    // 檢查是否在 stopword 列表中
    for (int i = 0; i < NUM_STOPWORDS; i++) {
        if (str_equal(word, d_stopwords[i])) {
            return true;
        }
    }
    return false;
}

// ==================== CUDA Kernels ====================

/**
 * 階段 1: 找出所有單詞的起始位置
 */
__global__ void find_word_boundaries(char* text, int text_length, 
                                     int* word_starts, int* word_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= text_length) return;
    
    // 檢查這個位置是否是單詞的開始
    // 單詞開始的條件：當前字符不是分隔符，且前一個字符是分隔符（或是文本開頭）
    bool is_word_start = false;
    
    if (idx == 0) {
        // 文本開頭，如果不是分隔符就是單詞開始
        if (!is_delimiter(text[idx])) {
            is_word_start = true;
        }
    } else {
        // 前一個是分隔符，當前不是分隔符
        if (is_delimiter(text[idx - 1]) && !is_delimiter(text[idx])) {
            is_word_start = true;
        }
    }
    
    // 如果是單詞開始，記錄位置
    if (is_word_start) {
        int pos = atomicAdd(word_count, 1);
        word_starts[pos] = idx;
    }
}

/**
 * 階段 2: 處理每個單詞
 */
__global__ void process_words(char* text, int text_length,
                              int* word_starts, int num_words,
                              WordCount* map_output, int* output_count,
                              int max_output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_words) return;
    
    int start_pos = word_starts[idx];
    
    // 計算單詞長度
    int word_len = 0;
    while (start_pos + word_len < text_length && 
           !is_delimiter(text[start_pos + word_len]) && 
           word_len < MAX_WORD_LENGTH - 1) {
        word_len++;
    }
    
    if (word_len == 0) return;
    
    // 轉小寫
    char lower_word[MAX_WORD_LENGTH];
    for (int i = 0; i < word_len; i++) {
        char c = text[start_pos + i];
        if (c >= 'A' && c <= 'Z') {
            lower_word[i] = c + 32;
        } else {
            lower_word[i] = c;
        }
    }
    lower_word[word_len] = '\0';
    
    // 檢查是否為合法單詞
    if (!is_valid_word(lower_word, word_len)) {
        return;
    }
    
    // 檢查是否為 stopword
    if (is_stopword_gpu(lower_word, word_len)) {
        return;
    }
    
    // 輸出 (word, 1)
    int out_idx = atomicAdd(output_count, 1);
    if (out_idx < max_output_size) {
        for (int i = 0; i <= word_len && i < MAX_WORD_LENGTH; i++) {
            map_output[out_idx].word[i] = lower_word[i];
        }
        map_output[out_idx].count = 1;
    }
}

// ==================== Hash-based Shuffle Kernel ====================

/**
 * Hash-based Shuffle Kernel
 * 使用 hash table 進行分組，替代 Thrust Sort
 * 輸入：所有 (word, 1) pairs
 * 輸出：Hash table with aggregated counts
 */
__global__ void shuffle_hash_grouping_kernel(
    WordCount* map_output,
    int num_pairs,
    HashEntry* hash_table,
    int* unique_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每個 thread 處理多個 (word, 1) pairs
    for (int i = tid; i < num_pairs; i += total_threads) {
        const char* word = map_output[i].word;
        int word_len = 0;
        while (word[word_len] != '\0' && word_len < MAX_WORD_LENGTH) word_len++;
        
        // 計算 hash
        uint64_t hash = compute_hash(word, word_len);
        int slot = hash % HASH_TABLE_SIZE;
        bool inserted = false;
        
        // Linear probing with atomic operations
        for (int probe = 0; probe < 256 && !inserted; probe++) {
            int idx = (slot + probe) % HASH_TABLE_SIZE;
            
            unsigned long long old = atomicCAS(
                (unsigned long long*)&hash_table[idx].key_hash,
                EMPTY_KEY,
                hash
            );
            
            if (old == EMPTY_KEY) {
                // 新 key，插入
                for (int j = 0; j < MAX_WORD_LENGTH; j++) {
                    hash_table[idx].word[j] = word[j];
                    if (word[j] == '\0') break;
                }
                atomicAdd(&hash_table[idx].count, 1);
                atomicAdd(unique_count, 1);
                inserted = true;
            } 
            else if (old == hash && str_equal_device(hash_table[idx].word, word)) {
                // 相同 key，累加
                atomicAdd(&hash_table[idx].count, 1);
                inserted = true;
            }
        }
    }
}

// ==================== Host Functions ====================

std::string read_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void copy_stopwords_to_device() {
    char h_stopwords[NUM_STOPWORDS][MAX_STOPWORD_LEN];
    memset(h_stopwords, 0, sizeof(h_stopwords));
    
    int count = 0;
    for (int i = 0; stopwords[i] != NULL && i < NUM_STOPWORDS; i++) {
        strncpy(h_stopwords[i], stopwords[i], MAX_STOPWORD_LEN - 1);
        count++;
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_stopwords, h_stopwords, sizeof(h_stopwords)));
    printf("Copied %d stopwords to GPU\n", count);
}

void init_delimiter_table() {
    bool table[256] = {false};
    const char* delims = " \t\n\r.,;:!?\"()[]{}\\<>-'";
    for (int i = 0; delims[i] != '\0'; i++) {
        table[(unsigned char)delims[i]] = true;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_delimiter_table, table, sizeof(table)));
}

// ==================== MapReduce Implementation (Hash-based Shuffle) ====================

void mapreduce_wordcount_hash_shuffle(const std::string& text) {
    printf("=== MapReduce Word Count (CUDA - Hash-based Shuffle Optimization) ===\n");
    printf("Text size: %zu bytes\n", text.size());
    printf("Hash table size: %d slots\n\n", HASH_TABLE_SIZE);
    
    int text_length = text.size();
    char* d_text;
    CUDA_CHECK(cudaMalloc(&d_text, text_length));
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text_length, cudaMemcpyHostToDevice));
    
    copy_stopwords_to_device();
    
    // ============ PHASE 1: MAP ============
    printf("Phase 1: MAP - Extracting words...\n");
    
    auto map_start_time = std::chrono::high_resolution_clock::now();
    
    // ===== 階段 1.1: 找出所有單詞邊界 =====
    int* d_word_starts;
    int* d_word_count;
    CUDA_CHECK(cudaMalloc(&d_word_starts, text_length * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_word_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_word_count, 0, sizeof(int)));
    
    int num_blocks_boundary = (text_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_word_boundaries<<<num_blocks_boundary, BLOCK_SIZE>>>(
        d_text, text_length, d_word_starts, d_word_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_word_count;
    CUDA_CHECK(cudaMemcpy(&h_word_count, d_word_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Found %d word boundaries\n", h_word_count);
    
    // ===== 階段 1.2: 處理每個單詞，產生 (word, 1) pairs =====
    int max_map_output = h_word_count + 1000;
    WordCount* d_map_output;
    int* d_output_count;
    CUDA_CHECK(cudaMalloc(&d_map_output, max_map_output * sizeof(WordCount)));
    CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output_count, 0, sizeof(int)));
    
    int num_blocks_process = (h_word_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_words<<<num_blocks_process, BLOCK_SIZE>>>(
        d_text, text_length, d_word_starts, h_word_count,
        d_map_output, d_output_count, max_map_output);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_output_count;
    CUDA_CHECK(cudaMemcpy(&h_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Map output: %d (word, 1) pairs\n", h_output_count);
    
    CUDA_CHECK(cudaFree(d_word_starts));
    CUDA_CHECK(cudaFree(d_word_count));
    
    auto shuffle_start_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> map_duration = shuffle_start_time - map_start_time;
    
    // ============ PHASE 2: SHUFFLE (Hash-based Grouping) ============
    printf("Phase 2: SHUFFLE - Hash-based grouping (optimized)...\n");
    
    // 分配 hash table
    HashEntry* d_hash_table;
    int* d_unique_count;
    CUDA_CHECK(cudaMalloc(&d_hash_table, HASH_TABLE_SIZE * sizeof(HashEntry)));
    CUDA_CHECK(cudaMalloc(&d_unique_count, sizeof(int)));
    
    // 初始化 hash table
    std::vector<HashEntry> init_table(HASH_TABLE_SIZE);
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        init_table[i].key_hash = EMPTY_KEY;
        init_table[i].count = 0;
        init_table[i].word[0] = '\0';
    }
    CUDA_CHECK(cudaMemcpy(d_hash_table, init_table.data(), 
                         HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_unique_count, 0, sizeof(int)));
    
    // 執行 hash-based grouping
    int num_blocks_shuffle = (h_output_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_shuffle = min(num_blocks_shuffle, 256);  // 限制 blocks 數量
    
    shuffle_hash_grouping_kernel<<<num_blocks_shuffle, BLOCK_SIZE>>>(
        d_map_output, h_output_count,
        d_hash_table, d_unique_count
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto reduce_start_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shuffle_duration = reduce_start_time - shuffle_start_time;
    
    // ============ PHASE 3: REDUCE ============
    printf("Phase 3: REDUCE - Collecting results from hash table...\n");
    
    // 複製結果回 CPU
    std::vector<HashEntry> h_hash_table(HASH_TABLE_SIZE);
    int h_unique_count;
    CUDA_CHECK(cudaMemcpy(h_hash_table.data(), d_hash_table,
                         HASH_TABLE_SIZE * sizeof(HashEntry),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_unique_count, d_unique_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // 收集結果
    std::vector<WordCount> h_results;
    h_results.reserve(h_unique_count + 100);
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        const HashEntry& entry = h_hash_table[i];
        if (entry.key_hash != EMPTY_KEY && entry.count > 0) {
            WordCount wc;
            strncpy(wc.word, entry.word, MAX_WORD_LENGTH - 1);
            wc.word[MAX_WORD_LENGTH - 1] = '\0';
            wc.count = entry.count;
            h_results.push_back(wc);
        }
    }
    
    auto reduce_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> reduce_duration = reduce_end_time - reduce_start_time;
    
    // 排序結果
    std::sort(h_results.begin(), h_results.end(),
              [](const WordCount& a, const WordCount& b) {
                  return a.count > b.count;
              });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end_time - map_start_time;
    
    // ==================== RESULTS ====================
    printf("\n=== Performance Report ===\n");
    printf("Total time  : %.2f ms\n", total_duration.count());
    printf("Map Time    : %.2f ms (%.1f%%) ← Produces %d (word,1) pairs\n", 
           map_duration.count(), map_duration.count() / total_duration.count() * 100, h_output_count);
    printf("Shuffle Time: %.2f ms (%.1f%%) ← Hash-based grouping (NOT sort!)\n", 
           shuffle_duration.count(), shuffle_duration.count() / total_duration.count() * 100);
    printf("Reduce Time : %.2f ms (%.1f%%)\n", 
           reduce_duration.count(), reduce_duration.count() / total_duration.count() * 100);
    
    printf("\n=== Word Count Results (Top 20) ===\n");
    for (size_t i = 0; i < 20 && i < h_results.size(); i++) {
        printf("%s: %d\n", h_results[i].word, h_results[i].count);
    }
    printf("\nTotal unique words: %zu\n", h_results.size());
    printf("Hash table utilization: %.2f%%\n", 
           100.0 * h_results.size() / HASH_TABLE_SIZE);
    
    // 清理
    cudaFree(d_text);
    cudaFree(d_map_output);
    cudaFree(d_output_count);
    cudaFree(d_hash_table);
    cudaFree(d_unique_count);
}

// ==================== Main Function ====================

int main(int argc, char** argv) {
    const char* filename = (argc > 1) ? argv[1] : dataset_path[0];
    
    printf("Reading file: %s\n", filename);
    std::string text = read_file(filename);
    
    if (text.empty()) {
        std::cerr << "Error: Empty file\n";
        return 1;
    }
    
    mapreduce_wordcount_hash_shuffle(text);
    
    return 0;
}
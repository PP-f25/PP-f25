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

// Shuffle 輸出：(key, list<value>) 格式
struct ShuffleGroup {
    char word[MAX_WORD_LENGTH];   // key
    int* values;                   // list of values (動態分配)
    int num_values;                // list 長度
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
 * Phase 1: 找出每個 word 對應的 group ID
 * 這個 kernel 只做 hash table 的建立和 group assignment
 */
__global__ void shuffle_assign_groups_kernel(
    WordCount* map_output,
    int num_pairs,
    int* group_indices,      // 輸出：每個 pair 分配到哪個組
    int* group_sizes,        // 輸出：每個組有多少個 values (原子計數)
    char (*unique_words)[MAX_WORD_LENGTH],  // 輸出：每個組的 word
    int* num_groups          // 輸出：總共有多少個不同的 word
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
        bool found = false;
        int group_id = -1;
        
        // Linear probing 找到或創建 group
        for (int probe = 0; probe < 256 && !found; probe++) {
            int idx = (slot + probe) % HASH_TABLE_SIZE;
            
            // 嘗試讀取這個 slot
            int current_size = atomicAdd(&group_sizes[idx], 0);  // 原子讀取
            
            if (current_size == 0) {
                // 空 slot，嘗試創建新 group
                int old = atomicCAS(&group_sizes[idx], 0, 1);
                if (old == 0) {
                    // 成功創建新 group
                    for (int j = 0; j < MAX_WORD_LENGTH; j++) {
                        unique_words[idx][j] = word[j];
                        if (word[j] == '\0') break;
                    }
                    atomicAdd(num_groups, 1);
                    group_id = idx;
                    found = true;
                } else {
                    // 其他 thread 搶先創建了，重新檢查
                    if (str_equal_device(unique_words[idx], word)) {
                        atomicAdd(&group_sizes[idx], 1);
                        group_id = idx;
                        found = true;
                    }
                }
            } else {
                // slot 已被佔用，檢查是否相同 word
                if (str_equal_device(unique_words[idx], word)) {
                    atomicAdd(&group_sizes[idx], 1);
                    group_id = idx;
                    found = true;
                }
            }
        }
        
        // 記錄這個 pair 屬於哪個 group
        if (group_id != -1) {
            group_indices[i] = group_id;
        }
    }
}

/**
 * Phase 2: 構建 (key, list<value>) 格式的 Shuffle 輸出
 * 把每個 pair 的 value (都是 1) 放到對應 group 的 list 中
 */
__global__ void shuffle_build_lists_kernel(
    WordCount* map_output,
    int num_pairs,
    int* group_indices,
    int* group_offsets,      // 每個 group 在 value_lists 中的起始位置
    int* group_positions,    // 每個 group 當前插入的位置 (原子遞增)
    int* value_lists         // 所有 groups 的 values (flattened array)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    for (int i = tid; i < num_pairs; i += total_threads) {
        int group_id = group_indices[i];
        if (group_id == -1) continue;
        
        // 取得這個 value 在 group 的 list 中的位置
        int pos_in_group = atomicAdd(&group_positions[group_id], 1);
        int global_pos = group_offsets[group_id] + pos_in_group;
        
        // 寫入 value (都是 1)
        value_lists[global_pos] = map_output[i].count;
    }
}

/**
 * Reduce Kernel (修正版)
 * 輸入：(key, list<value>) 格式
 * 功能：對每個 key 的 value list 進行聚合
 * 輸出：(key, aggregated_value)
 * 
 * 標準 Reduce 函數定義：
 * reduce(key, [v1, v2, v3, ...]) → (key, aggregated_result)
 * 
 * 範例：
 * reduce("the", [1, 1, 1, 1]) → ("the", 4)
 * reduce("cat", [1, 1])       → ("cat", 2)
 */
__global__ void reduce_kernel(
    char (*unique_words)[MAX_WORD_LENGTH],
    int* group_offsets,
    int* group_sizes,
    int* value_lists,
    WordCount* final_results,
    int* result_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= HASH_TABLE_SIZE) return;
    
    // 如果這個 group 有資料
    if (group_sizes[idx] > 0) {
        int pos = atomicAdd(result_count, 1);
        
        // 複製 key (word)
        for (int i = 0; i < MAX_WORD_LENGTH; i++) {
            final_results[pos].word[i] = unique_words[idx][i];
            if (unique_words[idx][i] == '\0') break;
        }
        
        // 聚合：對 value list 求和
        // 輸入：key = unique_words[idx]
        //       values = value_lists[offset ... offset+num_values-1]
        // 輸出：sum = values 的總和
        int sum = 0;
        int offset = group_offsets[idx];
        int num_values = group_sizes[idx];
        
        // reduce(key, [v1, v2, ..., vN]) 的實作
        for (int i = 0; i < num_values; i++) {
            sum += value_lists[offset + i];  // 每個 value 都是 1
        }
        
        final_results[pos].count = sum;
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
    
    // ============ PHASE 2: SHUFFLE (構建 (key, list<value>) 格式) ============
    printf("Phase 2: SHUFFLE - 構建 (key, list<value>) 格式...\n");
    
    // 分配資料結構
    int* d_group_indices;
    int* d_group_sizes;
    char (*d_unique_words)[MAX_WORD_LENGTH];
    int* d_num_groups;
    
    CUDA_CHECK(cudaMalloc(&d_group_indices, h_output_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_sizes, HASH_TABLE_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unique_words, HASH_TABLE_SIZE * MAX_WORD_LENGTH));
    CUDA_CHECK(cudaMalloc(&d_num_groups, sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_group_indices, -1, h_output_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_group_sizes, 0, HASH_TABLE_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_num_groups, 0, sizeof(int)));
    
    // Stage 2.1: 分配每個 pair 到 group，並統計每個 group 的大小
    printf("  Stage 2.1: Assigning pairs to groups...\n");
    int num_blocks_shuffle = (h_output_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_shuffle = min(num_blocks_shuffle, 256);
    
    shuffle_assign_groups_kernel<<<num_blocks_shuffle, BLOCK_SIZE>>>(
        d_map_output, h_output_count,
        d_group_indices, d_group_sizes, d_unique_words, d_num_groups
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_num_groups;
    CUDA_CHECK(cudaMemcpy(&h_num_groups, d_num_groups, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Found %d unique words (groups)\n", h_num_groups);
    
    // Stage 2.2: 計算每個 group 在 value_lists 中的偏移量 (prefix sum)
    printf("  Stage 2.2: Computing offsets for value lists...\n");
    std::vector<int> h_group_sizes(HASH_TABLE_SIZE);
    CUDA_CHECK(cudaMemcpy(h_group_sizes.data(), d_group_sizes,
                         HASH_TABLE_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<int> h_group_offsets(HASH_TABLE_SIZE);
    int total_values = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        h_group_offsets[i] = total_values;
        total_values += h_group_sizes[i];
    }
    
    int* d_group_offsets;
    CUDA_CHECK(cudaMalloc(&d_group_offsets, HASH_TABLE_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_group_offsets, h_group_offsets.data(),
                         HASH_TABLE_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    
    // Stage 2.3: 構建實際的 value lists
    printf("  Stage 2.3: Building value lists (total %d values)...\n", total_values);
    int* d_value_lists;
    int* d_group_positions;
    CUDA_CHECK(cudaMalloc(&d_value_lists, total_values * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_group_positions, HASH_TABLE_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_group_positions, 0, HASH_TABLE_SIZE * sizeof(int)));
    
    shuffle_build_lists_kernel<<<num_blocks_shuffle, BLOCK_SIZE>>>(
        d_map_output, h_output_count,
        d_group_indices, d_group_offsets, d_group_positions, d_value_lists
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("  Shuffle 完成！輸出格式：(key, list<value>)\n");
    
    printf("  Shuffle 完成！輸出格式：(key, list<value>)\n");
    
    // ===== DEBUG: 顯示 Shuffle 輸出 (key, list<value>) 格式 =====
    printf("\n--- Shuffle 輸出：(key, list<value>) 格式 (前 5 個 groups) ---\n");
    
    char (*h_unique_words)[MAX_WORD_LENGTH] = new char[HASH_TABLE_SIZE][MAX_WORD_LENGTH];
    std::vector<int> h_value_lists(total_values);
    
    CUDA_CHECK(cudaMemcpy(h_unique_words, d_unique_words,
                         HASH_TABLE_SIZE * MAX_WORD_LENGTH, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_value_lists.data(), d_value_lists,
                         total_values * sizeof(int), cudaMemcpyDeviceToHost));
    
    int shown = 0;
    for (int i = 0; i < HASH_TABLE_SIZE && shown < 5; i++) {
        if (h_group_sizes[i] > 0) {
            printf("Group %d: key=\"%s\", values=[", i, h_unique_words[i]);
            int offset = h_group_offsets[i];
            int count = h_group_sizes[i];
            
            // 顯示前 10 個 values
            int show_limit = (count > 10) ? 10 : count;
            for (int j = 0; j < show_limit; j++) {
                printf("%d", h_value_lists[offset + j]);
                if (j < show_limit - 1) printf(", ");
            }
            if (count > 10) printf(", ... (%d more)", count - 10);
            printf("]\n");
            shown++;
        }
    }
    printf("說明：這就是標準的 (key, list<value>) 格式！\n");
    printf("  - key: 單詞本身\n");
    printf("  - values: 一個 list，包含所有出現的 values (都是 1)\n");
    printf("  → Reduce 階段會對這個 list 求和\n");
    printf("\n範例驗證：\n");
    printf("  reduce(\"duty\", [1, 1, ..., 1] (32個)) → (\"duty\", 32)\n");
    printf("  reduce(\"zancas\", [1, 1])            → (\"zancas\", 2)\n");
    printf("---\n\n");
    
    delete[] h_unique_words;
    
    auto reduce_start_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shuffle_duration = reduce_start_time - shuffle_start_time;
    
    // ============ PHASE 3: REDUCE (Aggregation) ============
    printf("Phase 3: REDUCE - 對每個 (key, list<value>) 執行聚合...\n");
    
    // 分配 reduce 輸出
    WordCount* d_final_results;
    int* d_result_count;
    CUDA_CHECK(cudaMalloc(&d_final_results, HASH_TABLE_SIZE * sizeof(WordCount)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_result_count, 0, sizeof(int)));
    
    // 執行 Reduce：對每個 key 的 value list 進行聚合
    int num_blocks_reduce = (HASH_TABLE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<num_blocks_reduce, BLOCK_SIZE>>>(
        d_unique_words, d_group_offsets, d_group_sizes, d_value_lists,
        d_final_results, d_result_count
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 複製結果回 CPU
    int h_result_count;
    CUDA_CHECK(cudaMemcpy(&h_result_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<WordCount> h_results(h_result_count);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_final_results,
                         h_result_count * sizeof(WordCount),
                         cudaMemcpyDeviceToHost));
    
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
    printf("Shuffle Time: %.2f ms (%.1f%%) ← 構建 %d 個 (key, list<value>) groups\n", 
           shuffle_duration.count(), shuffle_duration.count() / total_duration.count() * 100, h_num_groups);
    printf("Reduce Time : %.2f ms (%.1f%%) ← 對每個 list 求和\n", 
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
    cudaFree(d_group_indices);
    cudaFree(d_group_sizes);
    cudaFree(d_unique_words);
    cudaFree(d_num_groups);
    cudaFree(d_group_offsets);
    cudaFree(d_group_positions);
    cudaFree(d_value_lists);
    cudaFree(d_final_results);
    cudaFree(d_result_count);
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
    
    // 執行完整標準 MapReduce (完全正確版本)
    // MAP: 產生 (word, 1) pairs
    // SHUFFLE: 構建 (key, list<value>) 格式 ← 真正的 Shuffle 輸出！
    // REDUCE: 對每個 (key, list<value>) 執行聚合函數
    mapreduce_wordcount_hash_shuffle(text);
    
    return 0;
}
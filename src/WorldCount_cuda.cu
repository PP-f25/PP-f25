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
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// ==================== Constants ====================
#define MAX_WORD_LENGTH 64
#define BLOCK_SIZE 256
#define NUM_STOPWORDS 200  // 增加以包含所有 stopwords (實際 198 個)
#define MAX_STOPWORD_LEN 16

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
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt"
};

// ==================== Device Constant Memory ====================
__constant__ char d_stopwords[NUM_STOPWORDS][MAX_STOPWORD_LEN];

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

// 修正：更精確的分隔符判斷，與 CPU 版本一致
__device__ bool is_delimiter(char c) {
    const char* delims = " \t\n\r.,;:!?\"()[]{}\\<>-'";
    for (int i = 0; delims[i] != '\0'; i++) {
        if (c == delims[i]) return true;
    }
    return false;
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

void mapreduce_wordcount(const std::string& text) {
    printf("=== MapReduce Word Count (CUDA) ===\n");
    printf("Text size: %zu bytes\n\n", text.size());
    
    int text_length = text.size();
    char* d_text;
    CUDA_CHECK(cudaMalloc(&d_text, text_length));
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text_length, cudaMemcpyHostToDevice));
    
    copy_stopwords_to_device();
    
    // ============ PHASE 1: MAP (兩階段法) ============
    printf("Phase 1: MAP - Extracting words (Two-Phase Method)...\n");
    
    // 開始 Map 階段計時
    auto map_start_time = std::chrono::high_resolution_clock::now();
    
    // ===== 階段 1.1: 找出所有單詞邊界 =====
    int* d_word_starts;
    int* d_word_count;
    CUDA_CHECK(cudaMalloc(&d_word_starts, text_length * sizeof(int)));  // 最多有 text_length 個單詞
    CUDA_CHECK(cudaMalloc(&d_word_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_word_count, 0, sizeof(int)));
    
    int num_blocks_boundary = (text_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    find_word_boundaries<<<num_blocks_boundary, BLOCK_SIZE>>>(
        d_text, text_length, d_word_starts, d_word_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int h_word_count;
    CUDA_CHECK(cudaMemcpy(&h_word_count, d_word_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Found %d word boundaries\n", h_word_count);
    
    // ===== 階段 1.2: 處理每個單詞 =====
    int max_map_output = h_word_count + 1000;  // 預留一些空間
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
    
    // 清理階段 1 的臨時記憶體
    CUDA_CHECK(cudaFree(d_word_starts));
    CUDA_CHECK(cudaFree(d_word_count));
    
    // ============ PHASE 2: SHUFFLE ============
    printf("Phase 2: SHUFFLE - Sorting by word (using GPU Thrust)...\n");
    
    // 開始 Shuffle 階段計時
    auto shuffle_start_time = std::chrono::high_resolution_clock::now();
    
    // 計算 Map 階段時間
    std::chrono::duration<double, std::milli> map_duration = shuffle_start_time - map_start_time;
    
    // 使用 Thrust 在 GPU 上進行排序
    thrust::device_ptr<WordCount> d_ptr(d_map_output);
    thrust::sort(thrust::device, d_ptr, d_ptr + h_output_count);
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待 Shuffle 完成
    
    // ============ PHASE 3: REDUCE ============
    printf("Phase 3: REDUCE - Aggregating counts (using GPU Thrust)...\n");
    
    // 開始 Reduce 階段計時
    auto reduce_start_time = std::chrono::high_resolution_clock::now();
    
    // 計算 Shuffle 階段時間
    std::chrono::duration<double, std::milli> shuffle_duration = reduce_start_time - shuffle_start_time;
    
    // 在 GPU 上使用 Thrust reduce_by_key 進行聚合
    WordCount* d_reduced_keys;
    int* d_reduced_counts;
    CUDA_CHECK(cudaMalloc(&d_reduced_keys, h_output_count * sizeof(WordCount)));
    CUDA_CHECK(cudaMalloc(&d_reduced_counts, h_output_count * sizeof(int)));
    
    thrust::device_ptr<WordCount> d_reduced_keys_ptr(d_reduced_keys);
    thrust::device_ptr<int> d_reduced_counts_ptr(d_reduced_counts);
    
    // 準備輸入的 counts (都是 1)
    int* d_input_counts;
    CUDA_CHECK(cudaMalloc(&d_input_counts, h_output_count * sizeof(int)));
    thrust::device_ptr<int> d_input_counts_ptr(d_input_counts);
    
    thrust::transform(d_ptr, d_ptr + h_output_count, d_input_counts_ptr,
                     [] __device__ (const WordCount& wc) { return wc.count; });
    
    auto new_end = thrust::reduce_by_key(
        d_ptr, d_ptr + h_output_count,
        d_input_counts_ptr,
        d_reduced_keys_ptr,
        d_reduced_counts_ptr,
        WordEqual()
    );
    
    int unique_words = new_end.first - d_reduced_keys_ptr;
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待 Reduce 完成
    
    // 結束 Reduce 階段計時
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 計算 Reduce 階段時間
    std::chrono::duration<double, std::milli> reduce_duration = end_time - reduce_start_time;
    
    // 將結果從 GPU 複製到 CPU 用於輸出
    std::vector<WordCount> h_keys(unique_words);
    std::vector<int> h_counts(unique_words);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_reduced_keys, 
                         unique_words * sizeof(WordCount), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_reduced_counts, 
                         unique_words * sizeof(int), cudaMemcpyDeviceToHost));
    
    // 組合結果
    std::vector<WordCount> h_results(unique_words);
    for (int i = 0; i < unique_words; i++) {
        h_results[i] = h_keys[i];
        h_results[i].count = h_counts[i];
    }
    
    // ============ PHASE 4: OUTPUT ============
    printf("Phase 4: OUTPUT - Collecting results...\n");
    
    std::sort(h_results.begin(), h_results.end(), 
              [](const WordCount& a, const WordCount& b) {
                  return a.count > b.count;
              });
    
    // ==================== TIMING REPORT ====================
    // 計算總時間
    std::chrono::duration<double, std::milli> total_duration = end_time - map_start_time;
    
    // 使用與 C++ 版本相同的輸出格式
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Map Time    : " << map_duration.count() << " ms" << std::endl;
    std::cout << "Shuffle Time: " << shuffle_duration.count() << " ms" << std::endl;
    std::cout << "Reduce Time : " << reduce_duration.count() << " ms" << std::endl;
    
    // printf("\n=== Word Count Results ===\n");
    // for (const auto& wc : h_results) {
    //     printf("%s: %d\n", wc.word, wc.count);
    // }
    
    // 清理 GPU 記憶體
    cudaFree(d_text);
    cudaFree(d_map_output);
    cudaFree(d_output_count);
    cudaFree(d_reduced_keys);
    cudaFree(d_reduced_counts);
    cudaFree(d_input_counts);
}

int main(int argc, char** argv) {
    const char* filename = (argc > 1) ? argv[1] : dataset_path[0];
    
    printf("Reading file: %s\n", filename);
    std::string text = read_file(filename);
    
    if (text.empty()) {
        std::cerr << "Error: Empty file\n";
        return 1;
    }
    
    mapreduce_wordcount(text);
    
    return 0;
}
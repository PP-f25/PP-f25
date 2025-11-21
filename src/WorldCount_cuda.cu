#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cctype>
#include <algorithm>
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
    "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
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
 * MAP Kernel: 修正版本
 */
__global__ void map_kernel(char* text, int text_length, 
                          WordCount* map_output, int* output_count,
                          int max_output_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每個 thread 處理一段文本
    int chunk_size = (text_length + total_threads - 1) / total_threads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, text_length);
    
    if (start >= text_length) return;
    
    // 關鍵修正：每個 thread 只處理「開始於」它範圍內的單詞
    // 如果 start 在單詞中間，跳到下一個單詞的開始
    if (tid > 0 && start > 0 && !is_delimiter(text[start - 1])) {
        // 我們在單詞中間，跳過這個單詞（讓前一個 thread 處理它）
        while (start < text_length && !is_delimiter(text[start])) {
            start++;
        }
        // 現在 start 指向分隔符，跳過所有連續的分隔符
        while (start < text_length && is_delimiter(text[start])) {
            start++;
        }
    }
    
    // 不要延伸 end！只處理嚴格在 [start, end) 範圍內開始的單詞
    // 如果單詞跨越 end 邊界，也要完整處理（讀取到單詞結尾）
    // 但不要延伸 end 本身，避免與下一個 thread 重疊
    
    // 處理這段文本中的單詞
    char word_buffer[MAX_WORD_LENGTH];
    int word_len = 0;
    int word_start_pos = start;  // 記錄當前單詞的開始位置
    
    // 處理從 start 到 end 的範圍
    // 策略：只處理「開始位置」在 [start, end) 內的單詞
    for (int i = start; i < text_length; i++) {
        char c = text[i];
        
        if (is_delimiter(c)) {
            // 單詞結束
            if (word_len > 0) {
                // 檢查這個單詞是否「開始」於我們的範圍內
                if (word_start_pos < end) {
                    word_buffer[word_len] = '\0';
                    
                    // 轉小寫
                    char lower_word[MAX_WORD_LENGTH];
                    to_lowercase(lower_word, word_buffer, word_len);
                    
                    // 檢查是否為合法單詞（無數字、特殊字元）
                    if (is_valid_word(lower_word, word_len)) {
                        // 檢查是否為 stopword
                        if (!is_stopword_gpu(lower_word, word_len)) {
                            // Emit (word, 1)
                            int idx = atomicAdd(output_count, 1);
                            
                            if (idx < max_output_size) {
                                for (int j = 0; j <= word_len && j < MAX_WORD_LENGTH; j++) {
                                    map_output[idx].word[j] = lower_word[j];
                                }
                                map_output[idx].count = 1;
                            }
                        }
                    }
                }
                
                word_len = 0;
            }
            
            // 如果已經超過我們的範圍，可以停止了
            if (i >= end) {
                break;
            }
            
            // 下一個單詞會從這裡開始
            word_start_pos = i + 1;
        } else {
            // 建構單詞
            if (word_len == 0) {
                // 新單詞開始
                word_start_pos = i;
            }
            if (word_len < MAX_WORD_LENGTH - 1) {
                word_buffer[word_len++] = c;
            }
        }
    }
    
    // 處理最後一個單詞（如果文本結尾沒有分隔符）
    if (word_len > 0 && word_start_pos < end) {
        word_buffer[word_len] = '\0';
        
        char lower_word[MAX_WORD_LENGTH];
        to_lowercase(lower_word, word_buffer, word_len);
        
        if (is_valid_word(lower_word, word_len)) {
            if (!is_stopword_gpu(lower_word, word_len)) {
                int idx = atomicAdd(output_count, 1);
                
                if (idx < max_output_size) {
                    for (int j = 0; j <= word_len && j < MAX_WORD_LENGTH; j++) {
                        map_output[idx].word[j] = lower_word[j];
                    }
                    map_output[idx].count = 1;
                }
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

void mapreduce_wordcount(const std::string& text) {
    printf("=== MapReduce Word Count (CUDA) ===\n");
    printf("Text size: %zu bytes\n\n", text.size());
    
    int text_length = text.size();
    char* d_text;
    CUDA_CHECK(cudaMalloc(&d_text, text_length));
    CUDA_CHECK(cudaMemcpy(d_text, text.c_str(), text_length, cudaMemcpyHostToDevice));
    
    copy_stopwords_to_device();
    
    // 分配 map 輸出空間
    int max_map_output = text_length / 3;
    WordCount* d_map_output;
    int* d_output_count;
    CUDA_CHECK(cudaMalloc(&d_map_output, max_map_output * sizeof(WordCount)));
    CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_output_count, 0, sizeof(int)));
    
    // ============ PHASE 1: MAP ============
    printf("Phase 1: MAP - Extracting words...\n");
    
    int num_blocks = (text_length + BLOCK_SIZE * 256 - 1) / (BLOCK_SIZE * 256);
    num_blocks = std::min(num_blocks, 128);
    if (num_blocks == 0) num_blocks = 1;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    map_kernel<<<num_blocks, BLOCK_SIZE>>>(d_text, text_length, 
                                           d_map_output, d_output_count,
                                           max_map_output);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float map_time;
    cudaEventElapsedTime(&map_time, start, stop);
    
    int h_output_count;
    CUDA_CHECK(cudaMemcpy(&h_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Map output: %d (word, 1) pairs\n", h_output_count);
    printf("  Map time: %.2f ms\n\n", map_time);
    
    // ============ PHASE 2: SHUFFLE ============
    printf("Phase 2: SHUFFLE - Sorting by word...\n");
    
    cudaEventRecord(start);
    thrust::device_ptr<WordCount> d_ptr(d_map_output);
    thrust::sort(thrust::device, d_ptr, d_ptr + h_output_count);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float shuffle_time;
    cudaEventElapsedTime(&shuffle_time, start, stop);
    printf("  Shuffle time: %.2f ms\n\n", shuffle_time);
    
    // ============ PHASE 3: REDUCE ============
    printf("Phase 3: REDUCE - Aggregating counts...\n");
    
    cudaEventRecord(start);
    
    WordCount* d_reduced_keys;
    int* d_reduced_counts;
    CUDA_CHECK(cudaMalloc(&d_reduced_keys, h_output_count * sizeof(WordCount)));
    CUDA_CHECK(cudaMalloc(&d_reduced_counts, h_output_count * sizeof(int)));
    
    thrust::device_ptr<WordCount> d_reduced_keys_ptr(d_reduced_keys);
    thrust::device_ptr<int> d_reduced_counts_ptr(d_reduced_counts);
    
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
    
    cudaEventRecord(stop);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float reduce_time;
    cudaEventElapsedTime(&reduce_time, start, stop);
    printf("  Unique words: %d\n", unique_words);
    printf("  Reduce time: %.2f ms\n\n", reduce_time);
    
    // ============ PHASE 4: OUTPUT ============
    printf("Phase 4: OUTPUT - Collecting results...\n");
    
    std::vector<WordCount> h_keys(unique_words);
    std::vector<int> h_counts(unique_words);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_reduced_keys, 
                         unique_words * sizeof(WordCount), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_reduced_counts, 
                         unique_words * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<WordCount> h_results(unique_words);
    for (int i = 0; i < unique_words; i++) {
        h_results[i] = h_keys[i];
        h_results[i].count = h_counts[i];
    }
    
    std::sort(h_results.begin(), h_results.end(), 
              [](const WordCount& a, const WordCount& b) {
                  return a.count > b.count;
              });
    
    // ==================== TIMING REPORT ====================
    float total_time = map_time + shuffle_time + reduce_time;
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║         MapReduce Performance Report (CUDA)            ║\n");
    printf("╠════════════════════════════════════════════════════════╣\n");
    printf("║ Phase 1: MAP                                           ║\n");
    printf("║   Time:        %8.2f ms  (%5.1f%%)                     ║\n", map_time, 100.0 * map_time / total_time);
    printf("║   Output:      %8d (word, 1) pairs                     ║\n", h_output_count);
    printf("║                                                        ║\n");
    printf("║ Phase 2: SHUFFLE (Sort by key)                         ║\n");
    printf("║   Time:        %8.2f ms  (%5.1f%%)                     ║\n", shuffle_time, 100.0 * shuffle_time / total_time);
    printf("║                                                        ║\n");
    printf("║ Phase 3: REDUCE (Aggregate by key)                     ║\n");
    printf("║   Time:        %8.2f ms  (%5.1f%%)                     ║\n", reduce_time, 100.0 * reduce_time / total_time);
    printf("║   Output:      %8d unique words                        ║\n", unique_words);
    printf("║                                                        ║\n");
    printf("╠════════════════════════════════════════════════════════╣\n");
    printf("║ Total MapReduce Time:  %8.2f ms                       ║\n", total_time);
    printf("║ Throughput:            %8.2f MB/s                   ║\n", (text.size() / 1024.0 / 1024.0) / (total_time / 1000.0));
    printf("╚════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    printf("=== Word Count Results ===\n");
    for (const auto& wc : h_results) {
        printf("%s: %d\n", wc.word, wc.count);
    }
    
    cudaFree(d_text);
    cudaFree(d_map_output);
    cudaFree(d_output_count);
    cudaFree(d_reduced_keys);
    cudaFree(d_reduced_counts);
    cudaFree(d_input_counts);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
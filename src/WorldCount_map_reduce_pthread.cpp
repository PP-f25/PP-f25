#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_set> 
#include <cstring>  
#include <cctype>
#include <unordered_map>
#include <pthread.h>
#include <stdlib.h>
#include <iomanip>
#include <chrono>

// 停用詞列表
const char * stopwords[] = {"a", "about", 
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
    "she's", "should", "shouldn", "shouldn't", "should_ve", "so", 
    "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
    "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
    "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",NULL};

std::unordered_set<std::string> stopwords_set;

const char *dataset_path[] = {
     // "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt",
    // "../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_English.txt",
    // "../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_English.txt",
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    // "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    // "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    // "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"
    // "../archive/text8"
    // "../enwik9/enwik9"
};

void initialize_stopwords() {
    if (stopwords_set.empty()) {
        for (int i = 0; stopwords[i] != NULL; i++) {
            stopwords_set.insert(stopwords[i]);
        }
    }
}

bool is_stopword_fast(const std::string& word) {
    return stopwords_set.count(word) > 0; 
}

struct KVPair {
    std::string key;
    int value;
};

using PartitionedData = std::vector<std::vector<std::vector<KVPair>>>;

// === 使用 Barrier 而非條件變數 ===
struct BarrierSync {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int waiting;
    
    void init(int n) {
        count = n;
        waiting = 0;
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&cond, NULL);
    }
    
    void wait() {
        pthread_mutex_lock(&mutex);
        waiting++;
        if (waiting == count) {
            waiting = 0;
            pthread_cond_broadcast(&cond);
        } else {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);
    }
    
    void destroy() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cond);
    }
};

// 簡化的工作結構
struct WorkerData {
    int thread_id;
    int num_threads;
    
    // Map
    const char* text;
    size_t content_size;
    std::vector<KVPair>* map_result;
    
    // Shuffle
    const std::vector<std::vector<KVPair>>* map_outputs;
    PartitionedData* partitions;
    
    // Reduce
    const PartitionedData* all_partitions;
    std::vector<KVPair>* reduce_result;
    
    // 同步
    BarrierSync* barrier;
    volatile bool* should_exit;
};

inline bool is_delimiter(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || 
           c == '.' || c == ',' || c == ';' || c == ':' || 
           c == '!' || c == '?' || c == '"' || c == '(' || 
           c == ')' || c == '[' || c == ']' || c == '{' || 
           c == '}' || c == '<' || c == '>' || c == '-' || c == '\'';
}

std::vector<KVPair> map_func_optimized(const char* text, size_t start, size_t end) {
    std::vector<KVPair> mapped_data;
    mapped_data.reserve(1000); 
    
    std::string word;
    word.reserve(32); 
    
    for (size_t i = start; i < end; i++) {
        char c = text[i];
        
        if (is_delimiter(c)) {
            if (!word.empty()) {
                if (word.length() > 2) {
                    bool valid = true;
                    for (char ch : word) {
                        if (!isalnum(ch) && ch != '_') {
                            valid = false;
                            break;
                        }
                    }
                    if (valid && !is_stopword_fast(word)) {
                        mapped_data.emplace_back(KVPair{word, 1});
                    }
                }
                word.clear();
            }
        } else {
            word += tolower(c);
        }
    }
    
    if (!word.empty() && word.length() > 2) {
        bool valid = true;
        for (char ch : word) {
            if (!isalnum(ch) && ch != '_') {
                valid = false;
                break;
            }
        }
        if (valid && !is_stopword_fast(word)) {
            mapped_data.emplace_back(KVPair{word, 1});
        }
    }
    return mapped_data;
}

// 統一的 worker function
void* worker_thread_func(void* arg) {
    WorkerData* data = (WorkerData*)arg;
    int tid = data->thread_id;
    int num_threads = data->num_threads;
    
    while (true) {
        // === MAP Phase ===
        data->barrier->wait();  // 等待 Map 開始
        
        if (*data->should_exit) break;
        
        // 計算分區 (修正版:調整 end_idx)
        size_t chunk_size = data->content_size / num_threads;
        size_t start_idx = tid * chunk_size;
        size_t end_idx = (tid == num_threads - 1) ? data->content_size : (tid + 1) * chunk_size;
        
        // 修正:調整 end_idx 確保完整單字
        if (tid < num_threads - 1 && end_idx < data->content_size) {
            while (end_idx < data->content_size && !is_delimiter(data->text[end_idx])) {
                end_idx++;
            }
            while (end_idx < data->content_size && is_delimiter(data->text[end_idx])) {
                end_idx++;
            }
        }
        
        *data->map_result = map_func_optimized(data->text, start_idx, end_idx);
        
        data->barrier->wait();  // Map 完成
        
        // === SHUFFLE Phase ===
        const auto& local_vec = (*data->map_outputs)[tid];
        auto& my_partitions = (*data->partitions)[tid];
        
        // 預先計算大小以減少 reallocation
        std::vector<size_t> partition_sizes(num_threads, 0);
        for (const auto& kv : local_vec) {
            std::size_t hash_val = std::hash<std::string>{}(kv.key);
            int partition_idx = hash_val % num_threads;
            partition_sizes[partition_idx]++;
        }
        for (int i = 0; i < num_threads; ++i) {
            my_partitions[i].reserve(partition_sizes[i]);
        }
        
        for (const auto& kv : local_vec) {
            std::size_t hash_val = std::hash<std::string>{}(kv.key);
            int partition_idx = hash_val % num_threads;
            my_partitions[partition_idx].push_back(kv);
        }
        
        data->barrier->wait();  // Shuffle 完成
        
        // === REDUCE Phase ===
        int p = tid;  // 每個執行緒負責一個 partition
        
        size_t total_items = 0;
        for (int t = 0; t < num_threads; ++t) {
            total_items += (*data->all_partitions)[t][p].size();
        }
        
        std::unordered_map<std::string, int> counts;
        counts.reserve(total_items / 2);
        
        for (int t = 0; t < num_threads; ++t) {
            const auto& data_chunk = (*data->all_partitions)[t][p];
            for (const auto& kv : data_chunk) {
                counts[kv.key] += kv.value;
            }
        }
        
        data->reduce_result->reserve(counts.size());
        for (const auto& pair : counts) {
            data->reduce_result->push_back(KVPair{pair.first, pair.second});
        }
        
        data->barrier->wait();  // Reduce 完成
    }
    
    return NULL;
}

int main(int argc, char* argv[]) {
    initialize_stopwords();
    
    int num_threads = 4;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            if (num_threads <= 0) num_threads = 1;
            i++;
        }
    }
    
    std::cout << "Running Optimized Pthread MapReduce with " << num_threads << " threads..." << std::endl;
    
    // 創建執行緒池和同步機制
    pthread_t* threads = new pthread_t[num_threads];
    WorkerData* workers = new WorkerData[num_threads];
    BarrierSync barrier;
    barrier.init(num_threads + 1);  // +1 for main thread
    volatile bool should_exit = false;
    
    int num_datasets = sizeof(dataset_path) / sizeof(dataset_path[0]);

    for (int d = 0; d < num_datasets; ++d) {
        std::string filepath = dataset_path[d];
        std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
        std::cout << "\n>>> Processing: " << filename << std::endl;

        std::ifstream ifs(dataset_path[d], std::ios::in);
        if (!ifs.is_open()) {
            std::cout << "Failed to open file: " << dataset_path[d] << "\n";
            continue;
        }

        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string content(ss.str());
        ifs.close();

        if (content.empty()) continue;

        const char* text = content.c_str();
        size_t content_size = content.size();

        // 準備資料結構
        std::vector<std::vector<KVPair>> thread_mapped_results(num_threads);
        PartitionedData partitions(num_threads, std::vector<std::vector<KVPair>>(num_threads));
        std::vector<std::vector<KVPair>> thread_reduce_results(num_threads);
        
        // 初始化 workers (只在第一次)
        if (d == 0) {
            for (int i = 0; i < num_threads; i++) {
                workers[i].thread_id = i;
                workers[i].num_threads = num_threads;
                workers[i].barrier = &barrier;
                workers[i].should_exit = &should_exit;
                workers[i].map_result = &thread_mapped_results[i];
                workers[i].map_outputs = &thread_mapped_results;
                workers[i].partitions = &partitions;
                workers[i].all_partitions = &partitions;
                workers[i].reduce_result = &thread_reduce_results[i];
                
                pthread_create(&threads[i], NULL, worker_thread_func, &workers[i]);
            }
        }
        
        // 更新每次的資料指標
        for (int i = 0; i < num_threads; i++) {
            workers[i].text = text;
            workers[i].content_size = content_size;
            workers[i].map_result = &thread_mapped_results[i];
            workers[i].map_outputs = &thread_mapped_results;
            workers[i].partitions = &partitions;
            workers[i].all_partitions = &partitions;
            workers[i].reduce_result = &thread_reduce_results[i];
        }

        auto total_start = std::chrono::high_resolution_clock::now();
        
        // === Map Phase ===
        auto map_start = std::chrono::high_resolution_clock::now();
        barrier.wait();  // 開始 Map
        barrier.wait();  // 等待 Map 完成
        auto map_end = std::chrono::high_resolution_clock::now();

        // === Shuffle Phase ===
        auto shuffle_start = std::chrono::high_resolution_clock::now();
        barrier.wait();  // Shuffle 完成 (worker 內部執行)
        auto shuffle_end = std::chrono::high_resolution_clock::now();

        // === Reduce Phase ===
        auto reduce_start = std::chrono::high_resolution_clock::now();
        barrier.wait();  // Reduce 完成
        auto reduce_end = std::chrono::high_resolution_clock::now();

        // 合併結果
        std::vector<KVPair> final_results;
        size_t total_final_size = 0;
        for (const auto& vec : thread_reduce_results) total_final_size += vec.size();
        final_results.reserve(total_final_size);
        for (const auto& vec : thread_reduce_results) {
            final_results.insert(final_results.end(), vec.begin(), vec.end());
        }

        std::chrono::duration<double, std::milli> map_elapsed = map_end - map_start;
        std::chrono::duration<double, std::milli> shuffle_elapsed = shuffle_end - shuffle_start;
        std::chrono::duration<double, std::milli> reduce_elapsed = reduce_end - reduce_start;
        std::chrono::duration<double, std::milli> total_elapsed = reduce_end - total_start;

        // 結果驗證
        std::cout << "--- Word Count Results ---" << std::endl;
        int count = 0;
        for (auto& pair : final_results) {
            if (pair.value > 500) {
                std::cout << pair.key << ": " << pair.value << std::endl;
                count++;
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total Time  : " << total_elapsed.count() << " ms" << std::endl;
        std::cout << "Map Time    : " << map_elapsed.count() << " ms" << std::endl;
        std::cout << "Shuffle Time: " << shuffle_elapsed.count() << " ms" << std::endl;
        std::cout << "Reduce Time : " << reduce_elapsed.count() << " ms" << std::endl;
    }
    
    // 清理
    should_exit = true;
    barrier.wait();  // 通知退出
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    barrier.destroy();
    delete[] threads;
    delete[] workers;

    return 0;
}
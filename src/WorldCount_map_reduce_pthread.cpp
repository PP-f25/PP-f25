#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_set> 
#include <cstring>  
#include <cctype>
#include <map>
#include <unordered_map>
#include <pthread.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <chrono>

// --- 停用詞優化 ---
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
    //"../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    // "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    // "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    // "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"
    // "../archive/text8"
    "../enwik9/enwik9"
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

// --- 數據結構 ---
struct KVPair {
    std::string key;
    int value;
};

using PartitionedData = std::vector<std::vector<std::vector<KVPair>>>;

// 工作類型枚舉
enum WorkType {
    WORK_NONE,
    WORK_MAP,
    WORK_SHUFFLE,
    WORK_REDUCE,
    WORK_EXIT
};

// 統一的 Worker 結構
struct WorkerThread {
    int thread_id;
    pthread_t handle;
    
    // 同步機制
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    
    WorkType current_work;
    bool work_ready;
    bool work_completed;
    
    // Map 相關
    const char* text;
    size_t start_idx;
    size_t end_idx;
    std::vector<KVPair>* map_result;
    
    // Shuffle 相關
    int num_partitions;
    const std::vector<std::vector<KVPair>>* map_outputs;
    PartitionedData* partitions;
    
    // Reduce 相關
    int partition_id;
    int num_producers;
    const PartitionedData* all_partitions;
    std::vector<KVPair>* reduce_result;
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

// 統一的 Worker Thread 函數
void* worker_thread_func(void* arg) {
    WorkerThread* worker = (WorkerThread*)arg;
    
    while (true) {
        // 等待工作
        pthread_mutex_lock(&worker->mutex);
        while (!worker->work_ready) {
            pthread_cond_wait(&worker->cond_start, &worker->mutex);
        }
        
        WorkType work = worker->current_work;
        worker->work_ready = false;
        pthread_mutex_unlock(&worker->mutex);
        
        // 執行工作
        if (work == WORK_EXIT) {
            break;
        }
        else if (work == WORK_MAP) {
            *worker->map_result = map_func_optimized(
                worker->text, 
                worker->start_idx, 
                worker->end_idx
            );
        }
        else if (work == WORK_SHUFFLE) {
            int tid = worker->thread_id;
            const auto& local_vec = (*worker->map_outputs)[tid];
            auto& my_partitions = (*worker->partitions)[tid];
            
            std::vector<size_t> partition_sizes(worker->num_partitions, 0);
            for (const auto& kv : local_vec) {
                std::size_t hash_val = std::hash<std::string>{}(kv.key);
                int partition_idx = hash_val % worker->num_partitions;
                partition_sizes[partition_idx]++;
            }
            for (int i = 0; i < worker->num_partitions; ++i) {
                my_partitions[i].reserve(partition_sizes[i]);
            }
            
            for (const auto& kv : local_vec) {
                std::size_t hash_val = std::hash<std::string>{}(kv.key);
                int partition_idx = hash_val % worker->num_partitions;
                my_partitions[partition_idx].push_back(kv);
            }
        }
        else if (work == WORK_REDUCE) {
            int p = worker->partition_id;
            
            size_t total_items = 0;
            for (int t = 0; t < worker->num_producers; ++t) {
                total_items += (*worker->all_partitions)[t][p].size();
            }
            
            std::unordered_map<std::string, int> counts;
            counts.reserve(total_items / 2);
            
            for (int t = 0; t < worker->num_producers; ++t) {
                const auto& data_chunk = (*worker->all_partitions)[t][p];
                for (const auto& kv : data_chunk) {
                    counts[kv.key] += kv.value;
                }
            }
            
            worker->reduce_result->reserve(counts.size());
            for (const auto& pair : counts) {
                worker->reduce_result->push_back(KVPair{pair.first, pair.second});
            }
        }
        
        // 通知完成
        pthread_mutex_lock(&worker->mutex);
        worker->work_completed = true;
        pthread_cond_signal(&worker->cond_done);
        pthread_mutex_unlock(&worker->mutex);
    }
    
    return NULL;
}

// 初始化 Worker Pool
void init_worker_pool(std::vector<WorkerThread>& workers, int num_threads) {
    workers.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        workers[i].thread_id = i;
        workers[i].current_work = WORK_NONE;
        workers[i].work_ready = false;
        workers[i].work_completed = false;
        
        pthread_mutex_init(&workers[i].mutex, NULL);
        pthread_cond_init(&workers[i].cond_start, NULL);
        pthread_cond_init(&workers[i].cond_done, NULL);
        
        pthread_create(&workers[i].handle, NULL, worker_thread_func, &workers[i]);
    }
}

// 分配 Map 工作
void dispatch_map_work(std::vector<WorkerThread>& workers, 
                       const char* text, size_t content_size,
                       std::vector<std::vector<KVPair>>& results) {
    int num_threads = workers.size();
    size_t chunk_size = content_size / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (i == num_threads - 1) ? content_size : (i + 1) * chunk_size;
        
        if (i < num_threads - 1 && end_idx < content_size) {
            while (end_idx < content_size && !is_delimiter(text[end_idx])) end_idx++;
            while (end_idx < content_size && is_delimiter(text[end_idx])) end_idx++;
        }
        
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].text = text;
        workers[i].start_idx = start_idx;
        workers[i].end_idx = end_idx;
        workers[i].map_result = &results[i];
        workers[i].current_work = WORK_MAP;
        workers[i].work_ready = true;
        workers[i].work_completed = false;
        pthread_cond_signal(&workers[i].cond_start);
        pthread_mutex_unlock(&workers[i].mutex);
    }
}

// 分配 Shuffle 工作
void dispatch_shuffle_work(std::vector<WorkerThread>& workers,
                          const std::vector<std::vector<KVPair>>& map_outputs,
                          PartitionedData& partitions) {
    int num_threads = workers.size();
    
    for (int i = 0; i < num_threads; i++) {
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].num_partitions = num_threads;
        workers[i].map_outputs = &map_outputs;
        workers[i].partitions = &partitions;
        workers[i].current_work = WORK_SHUFFLE;
        workers[i].work_ready = true;
        workers[i].work_completed = false;
        pthread_cond_signal(&workers[i].cond_start);
        pthread_mutex_unlock(&workers[i].mutex);
    }
}

// 分配 Reduce 工作
void dispatch_reduce_work(std::vector<WorkerThread>& workers,
                         const PartitionedData& partitions,
                         std::vector<std::vector<KVPair>>& results) {
    int num_threads = workers.size();
    
    for (int i = 0; i < num_threads; i++) {
        pthread_mutex_lock(&workers[i].mutex);
        workers[i].partition_id = i;
        workers[i].num_producers = num_threads;
        workers[i].all_partitions = &partitions;
        workers[i].reduce_result = &results[i];
        workers[i].current_work = WORK_REDUCE;
        workers[i].work_ready = true;
        workers[i].work_completed = false;
        pthread_cond_signal(&workers[i].cond_start);
        pthread_mutex_unlock(&workers[i].mutex);
    }
}

// 等待所有工作完成
void wait_all_workers(std::vector<WorkerThread>& workers) {
    for (auto& worker : workers) {
        pthread_mutex_lock(&worker.mutex);
        while (!worker.work_completed) {
            pthread_cond_wait(&worker.cond_done, &worker.mutex);
        }
        pthread_mutex_unlock(&worker.mutex);
    }
}

// 清理 Worker Pool
void cleanup_worker_pool(std::vector<WorkerThread>& workers) {
    for (auto& worker : workers) {
        pthread_mutex_lock(&worker.mutex);
        worker.current_work = WORK_EXIT;
        worker.work_ready = true;
        pthread_cond_signal(&worker.cond_start);
        pthread_mutex_unlock(&worker.mutex);
        
        pthread_join(worker.handle, NULL);
        
        pthread_mutex_destroy(&worker.mutex);
        pthread_cond_destroy(&worker.cond_start);
        pthread_cond_destroy(&worker.cond_done);
    }
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
    
    // 一次性創建執行緒池
    std::vector<WorkerThread> workers;
    init_worker_pool(workers, num_threads);
    
    int num_datasets = sizeof(dataset_path) / sizeof(dataset_path[0]);

    for (int d = 0; d < num_datasets; ++d) {
        std::string filepath = dataset_path[d];
        std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);

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

        // --- Map Phase ---
        auto map_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<KVPair>> thread_mapped_results(num_threads);
        dispatch_map_work(workers, text, content_size, thread_mapped_results);
        wait_all_workers(workers);
        
        auto map_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> map_elapsed = map_end - map_start;

        // --- Shuffle Phase ---
        auto shuffle_start = std::chrono::high_resolution_clock::now();
        
        PartitionedData partitions(num_threads, std::vector<std::vector<KVPair>>(num_threads));
        dispatch_shuffle_work(workers, thread_mapped_results, partitions);
        wait_all_workers(workers);
        
        auto shuffle_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> shuffle_elapsed = shuffle_end - shuffle_start;

        // --- Reduce Phase ---
        auto reduce_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<KVPair>> thread_reduce_results(num_threads);
        dispatch_reduce_work(workers, partitions, thread_reduce_results);
        wait_all_workers(workers);
        
        std::vector<KVPair> final_results;
        size_t total_final_size = 0;
        for (const auto& vec : thread_reduce_results) total_final_size += vec.size();
        final_results.reserve(total_final_size);
        for (const auto& vec : thread_reduce_results) {
            final_results.insert(final_results.end(), vec.begin(), vec.end());
        }
        
        auto reduce_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> reduce_elapsed = reduce_end - reduce_start;
        std::chrono::duration<double, std::milli> total_elapsed = reduce_end - map_start;

        // ===== Print Results =====
        std::cout << "--- Word Count MapReduce Results (Reusable Threads) ---" << std::endl;
        int count = 0;
        for (auto& pair : final_results) {
            if (pair.value > 100000) {
                std::cout << pair.key << ": " << pair.value << std::endl;
                count++;
            }
        }
        // std::cout << "Total unique words > 500: " << count << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total Time  : " << total_elapsed.count() << " ms" << std::endl;
        std::cout << "Map Time    : " << map_elapsed.count() << " ms" << std::endl;
        std::cout << "Shuffle Time: " << shuffle_elapsed.count() << " ms" << std::endl;
        std::cout << "Reduce Time : " << reduce_elapsed.count() << " ms" << std::endl;
    }
    
    // 清理執行緒池
    cleanup_worker_pool(workers);

    return 0;
}
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
#include <chrono> // 引入 chrono

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
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    // "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    // "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    // "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt",
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

// 3D 資料結構: [Producer_ID][Partition_ID][Data_Vector]
using PartitionedData = std::vector<std::vector<std::vector<KVPair>>>;

// Map 任務結構
struct MapTaskArg {
    const char* text;
    size_t start_idx;
    size_t end_idx;
    std::vector<KVPair>* local_result;
};

// Shuffle 任務結構
struct ShuffleTaskArg {
    int thread_id;
    int num_partitions;
    const std::vector<std::vector<KVPair>>* map_outputs;
    PartitionedData* partitions; // 指向 3D 結構
};

// Reduce 任務結構
struct ReduceTaskArg {
    int partition_id; // 該執行緒負責的分區 ID
    int num_producers; // Map 執行緒的總數 (生產者數量)
    const PartitionedData* all_partitions; // 唯讀的 3D 結構來源
    std::vector<KVPair>* local_result; // 該執行緒的 Reduce 結果
};

// 優化的分隔符檢查函數
inline bool is_delimiter(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || 
           c == '.' || c == ',' || c == ';' || c == ':' || 
           c == '!' || c == '?' || c == '"' || c == '(' || 
           c == ')' || c == '[' || c == ']' || c == '{' || 
           c == '}' || c == '<' || c == '>' || c == '-' || c == '\'';
}

// 優化的 Map 函數
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

// Map thread function
void* map_thread(void* arg) {
    MapTaskArg* task = (MapTaskArg*)arg;
    *task->local_result = map_func_optimized(task->text, task->start_idx, task->end_idx);
    return NULL;
}

// Shuffle thread function (無鎖)
void* shuffle_thread(void* arg) {
    ShuffleTaskArg* task = (ShuffleTaskArg*)arg;
    int tid = task->thread_id;
    
    // 讀取自己 Map 階段的產出
    const auto& local_vec = (*task->map_outputs)[tid];
    
    // 寫入自己的 Partition 區域 [tid][partition_id]
    auto& my_partitions = (*task->partitions)[tid];
    
    for (const auto& kv : local_vec) {
        std::size_t hash_val = std::hash<std::string>{}(kv.key);
        int partition_idx = hash_val % task->num_partitions;
        
        my_partitions[partition_idx].push_back(kv);
    }
    return NULL;
}

// Reduce thread function
void* reduce_thread(void* arg) {
    ReduceTaskArg* task = (ReduceTaskArg*)arg;
    int p = task->partition_id;
    
    // 使用局部 Map 統計
    std::unordered_map<std::string, int> counts;
    
    // 遍歷所有生產者 (Map執行緒)
    for (int t = 0; t < task->num_producers; ++t) {
        const auto& data_chunk = (*task->all_partitions)[t][p];
        for (const auto& kv : data_chunk) {
            counts[kv.key] += kv.value;
        }
    }
    
    // 將結果轉回 vector
    task->local_result->reserve(counts.size());
    for (const auto& pair : counts) {
        task->local_result->push_back(KVPair{pair.first, pair.second});
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
    
    std::cout << "Running WorldCount (Pthreads Optimized - Lock Free) with " << num_threads << " threads (Chrono Timer)" << std::endl;

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

        // --- Map Phase ---
        auto map_start = std::chrono::high_resolution_clock::now(); 

        pthread_t* map_threads = new pthread_t[num_threads];
        MapTaskArg* map_tasks = new MapTaskArg[num_threads];
        std::vector<std::vector<KVPair>> thread_mapped_results(num_threads); 
        
        size_t chunk_size = content_size / num_threads;
        
        for (int i = 0; i < num_threads; i++) {
            size_t start_idx = i * chunk_size;
            size_t end_idx = (i == num_threads - 1) ? content_size : (i + 1) * chunk_size;
            
            if (i > 0 && start_idx < content_size) {
                while (start_idx < content_size && !is_delimiter(text[start_idx])) start_idx++;
                while (start_idx < content_size && is_delimiter(text[start_idx])) start_idx++;
            }
            
            map_tasks[i].text = text;
            map_tasks[i].start_idx = start_idx;
            map_tasks[i].end_idx = end_idx;
            map_tasks[i].local_result = &thread_mapped_results[i];
            
            pthread_create(&map_threads[i], NULL, map_thread, &map_tasks[i]);
        }
        
        for (int i = 0; i < num_threads; i++) {
            pthread_join(map_threads[i], NULL);
        }
        
        auto map_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> map_elapsed = map_end - map_start;

        // --- Shuffle Phase ---
        auto shuffle_start = std::chrono::high_resolution_clock::now();

        PartitionedData partitions(num_threads, std::vector<std::vector<KVPair>>(num_threads));

        pthread_t* shuffle_threads = new pthread_t[num_threads];
        ShuffleTaskArg* shuffle_tasks = new ShuffleTaskArg[num_threads];

        for (int i = 0; i < num_threads; ++i) {
            shuffle_tasks[i].thread_id = i;
            shuffle_tasks[i].num_partitions = num_threads;
            shuffle_tasks[i].map_outputs = &thread_mapped_results;
            shuffle_tasks[i].partitions = &partitions;

            pthread_create(&shuffle_threads[i], NULL, shuffle_thread, &shuffle_tasks[i]);
        }

        for (int i = 0; i < num_threads; ++i) {
            pthread_join(shuffle_threads[i], NULL);
        }

        auto shuffle_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> shuffle_elapsed = shuffle_end - shuffle_start;

        // --- Reduce Phase ---
        auto reduce_start = std::chrono::high_resolution_clock::now();

        pthread_t* reduce_threads = new pthread_t[num_threads];
        ReduceTaskArg* reduce_tasks = new ReduceTaskArg[num_threads];
        std::vector<std::vector<KVPair>> thread_reduce_results(num_threads);

        for (int i = 0; i < num_threads; ++i) {
            reduce_tasks[i].partition_id = i;
            reduce_tasks[i].num_producers = num_threads;
            reduce_tasks[i].all_partitions = &partitions;
            reduce_tasks[i].local_result = &thread_reduce_results[i];

            pthread_create(&reduce_threads[i], NULL, reduce_thread, &reduce_tasks[i]);
        }

        for (int i = 0; i < num_threads; ++i) {
            pthread_join(reduce_threads[i], NULL);
        }

        // 最終合併
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
        std::cout << "--- Word Count MapReduce Results (Verify) ---" << std::endl;
        int count = 0;
        for (auto& pair : final_results) {
            if (pair.value > 100) {
                count++;
            }
        }
        std::cout << "Total unique words > 100: " << count << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Map Time    : " << map_elapsed.count() << " ms" << std::endl;
        std::cout << "Shuffle Time: " << shuffle_elapsed.count() << " ms" << std::endl;
        std::cout << "Reduce Time : " << reduce_elapsed.count() << " ms" << std::endl;
        std::cout << "Total Time  : " << total_elapsed.count() << " ms" << std::endl;

        // 清理
        delete[] map_threads;
        delete[] map_tasks;
        delete[] shuffle_threads;
        delete[] shuffle_tasks;
        delete[] reduce_threads;
        delete[] reduce_tasks;
    }

    return 0;
}
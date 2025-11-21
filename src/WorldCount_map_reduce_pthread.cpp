// #include<iostream>
// #include <stdio.h>
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <string.h>
// #include <unordered_set> 
// #include <cstring>  
// #include <cctype>
// #include <map>
// #include <pthread.h>
// #include <stdlib.h>
// #include <algorithm>
// #include <sys/time.h> // 使用 gettimeofday 進行掛鐘時間計時
// #include <iomanip> // 為了控制輸出的小數點位數

// // --- 時間函數 (掛鐘時間 Wall Time) ---
// double wtime() {
//     struct timeval t;
//     gettimeofday(&t, NULL);
//     return (double)t.tv_sec + (double)t.tv_usec * 1.0E-6;
// }

// // --- 停用詞優化 ---
// const char * stopwords[] = {"a", "about", 
//     "above", "after", "again", "against", "ain",
//     "all", "am", "an", "and", "any", "are", "aren", 
//     "aren't", "as", "at", "be", "because", "been", 
//     "before", "being", "below", "between", "both", 
//     "but", "by", "can", "couldn", "couldn't", "d",
//     "did", "didn", "didn't", "do", "does", "doesn", 
//     "doesn't", "doing", "don", "don't", "down", "during",
//     "each", "few", "for", "from", "further", "had", "hadn",
//     "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't",
//     "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
//     "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll",
//     "i'm", "in", "into", "is", "isn", "isn't", "it", "it'd", "it'll",
//     "it's", "its", "itself", "i've", "just", "ll", "m", "ma", "me", 
//     "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my",
//     "myself", "needn", "needn't","no", "nor", "not", "now", "o", "of", 
//     "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", 
//     "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll",
//     "she's", "should", "shouldn", "shouldn't", "should_ve", "so", 
//     "some", "such", "t", "than", "that", "that'll", "the", "their",
//     "theirs", "them", "themselves", "then", "there", "these", "they", 
//     "they'd", "they'll", "they're", "they've", "this", "those", "through",
//     "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", 
//     "wasn't", "we", "we'd", "we'll", "we're", "were", "weren", "weren't", 
//     "we've", "what", "when", "where", "which", "while", "who", "whom", "why",
//     "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd",
//     "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",NULL};

// std::unordered_set<std::string> stopwords_set;
// const char *dataset_path[] = {
//     "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
// };

// void initialize_stopwords() {
//     if (stopwords_set.empty()) {
//         for (int i = 0; stopwords[i] != NULL; i++) {
//             stopwords_set.insert(stopwords[i]);
//         }
//     }
// }

// bool is_stopword_fast(const std::string& word) {
//     return stopwords_set.count(word) > 0; 
// }
// // --- 數據結構 ---

// struct KVPair {
//     char* key;
//     int value;
// };

// // Map 任務結構
// struct MapTaskArg {
//     char* chunk_start;
//     size_t chunk_size;
//     std::vector<KVPair>* local_result;
// };

// // Reduce 任務結構
// struct ReduceTaskArg {
//     std::vector<std::pair<std::string, std::vector<int>>>* groups;
//     size_t start_idx;
//     size_t end_idx;
//     std::vector<KVPair>* result;
//     pthread_mutex_t* mutex;
// };

// // Map function (優化後)
// std::vector<KVPair> map_func(char* article_chunk) {
//     std::vector<KVPair> mapped_data;
//     const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
//     char* saveptr;
    
//     char* token;
//     token = strtok_r(article_chunk, delimiters, &saveptr);

//     while (token != NULL) {
//         // turn to lower
//         for (int i = 0; token[i]; i++) {
//             token[i] = tolower(token[i]);
//         }

//         // 1. 過濾長度
//         if (strlen(token) <= 2) {
//             token = strtok_r(NULL, delimiters, &saveptr);
//             continue;
//         }

//         // 2. 過濾非英數符號 (檢查是否為有效單字)
//         bool is_word_char = true;
//         if (strlen(token) == 0) {
//              is_word_char = false;
//         }

//         for (int i = 0; token[i]; i++) {
//             if (!isalnum(token[i]) && token[i] != '_') { 
//                 is_word_char = false;
//                 break;
//             }
//         }

//         if (!is_word_char) {
//             token = strtok_r(NULL, delimiters, &saveptr);
//             continue;
//         }
        
//         // 3. 停用詞檢查 (O(1) 查找)
//         std::string current_word(token); 
//         if (is_stopword_fast(current_word)) {
//             token = strtok_r(NULL, delimiters, &saveptr);
//             continue;
//         }

//         KVPair pair;
//         pair.key = strdup(token); 
//         pair.value = 1;
        
//         mapped_data.push_back(pair);

//         token = strtok_r(NULL, delimiters, &saveptr);
//     }
//     return mapped_data;
// }

// // Map thread function
// void* map_thread(void* arg) {
//     MapTaskArg* task = (MapTaskArg*)arg;
    
//     char* end_ptr = task->chunk_start + task->chunk_size;
//     char original_char = *end_ptr; 
//     *end_ptr = '\0'; 
    
//     *task->local_result = map_func(task->chunk_start);
    
//     *end_ptr = original_char; 
    
//     return NULL;
// }

// // Reduce function
// KVPair reduce(const std::string& key, const std::vector<int>& values) {
//     int sum = 0;
//     for (int val : values) {
//         sum += val;
//     }

//     KVPair result;
//     result.key = strdup(key.c_str());
//     result.value = sum;
    
//     return result;
// }

// // Reduce thread function
// void* reduce_thread(void* arg) {
//     ReduceTaskArg* task = (ReduceTaskArg*)arg;
    
//     std::vector<KVPair> local_result;
    
//     for (size_t i = task->start_idx; i < task->end_idx; i++) {
//         auto& group = (*task->groups)[i];
//         local_result.push_back(reduce(group.first, group.second));
//     }
    
//     // 將局部結果加入共享結果向量 (需要鎖定)
//     pthread_mutex_lock(task->mutex);
//     task->result->insert(task->result->end(), local_result.begin(), local_result.end());
//     pthread_mutex_unlock(task->mutex);
    
//     return NULL;
// }

// // Shuffle function
// std::map<std::string, std::vector<int>> shuffle(const std::vector<KVPair>& mapped_data) {
//     std::map<std::string, std::vector<int>> shuffled_map;

//     for (const auto& pair : mapped_data) {
//         shuffled_map[std::string(pair.key)].push_back(pair.value);
//         free(pair.key); 
//     }
//     return shuffled_map;
// }

// int main(int argc, char* argv[]) {
//     // 初始化停用詞集合
//     initialize_stopwords();
    
//     // 預設執行緒數和解析參數
//     int num_threads = 4;
    
//     for (int i = 1; i < argc; i++) {
//         if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
//             num_threads = atoi(argv[i + 1]);
//             if (num_threads <= 0) num_threads = 1;
//             i++;
//         }
//     }
    
//     std::cout << "Running WorldCount (Pthreads) with " << num_threads << " threads" << std::endl;

//     // 讀取檔案 (序列化)
//     std::ifstream ifs(dataset_path[0], std::ios::in);
//     if (!ifs.is_open()) {
//         std::cout << "Failed to open file.\n";
//         return 1;
//     }

//     std::stringstream ss;
//     ss << ifs.rdbuf();
//     std::string content(ss.str());
//     ifs.close();

//     char* mutable_content = strdup(content.c_str());
//     size_t content_size = content.size();

//     // --- 開始 Map 計時 ---
//     double map_start_time = wtime(); 

//     // ===== Map Phase (Parallel) =====
//     pthread_t* map_threads = new pthread_t[num_threads];
//     MapTaskArg* map_tasks = new MapTaskArg[num_threads];
//     std::vector<std::vector<KVPair>> thread_mapped_results(num_threads); 
    
//     size_t chunk_size = content_size / num_threads;
    
//     for (int i = 0; i < num_threads; i++) {
//         size_t start_idx = i * chunk_size;
//         size_t end_idx = (i == num_threads - 1) ? content_size : (i + 1) * chunk_size;
        
//         if (i > 0 && start_idx < content_size) {
//             while (start_idx < content_size && !isspace(mutable_content[start_idx])) {
//                 start_idx++;
//             }
//         }
        
//         map_tasks[i].chunk_start = mutable_content + start_idx;
//         map_tasks[i].chunk_size = end_idx - start_idx;
//         map_tasks[i].local_result = &thread_mapped_results[i];
        
//         pthread_create(&map_threads[i], NULL, map_thread, &map_tasks[i]);
//     }
    
//     for (int i = 0; i < num_threads; i++) {
//         pthread_join(map_threads[i], NULL);
//     }

//     // 彙總 Map 結果 (序列化)
//     std::vector<KVPair> all_mapped_results;
//     size_t total_mapped_size = 0;
//     for (const auto& local_vec : thread_mapped_results) {
//         total_mapped_size += local_vec.size();
//     }
//     all_mapped_results.reserve(total_mapped_size);
//     for (auto& local_vec : thread_mapped_results) {
//         all_mapped_results.insert(all_mapped_results.end(), 
//                                   std::make_move_iterator(local_vec.begin()), 
//                                   std::make_move_iterator(local_vec.end()));
//     }
    
//     // --- 結束 Map 計時，開始 Shuffle 計時 ---
//     double shuffle_start_time = wtime();
//     double map_time_ms = (shuffle_start_time - map_start_time) * 1000.0;

//     // ===== Shuffle Phase (Sequential) =====
//     std::map<std::string, std::vector<int>> shuffled_results = shuffle(all_mapped_results);

//     // --- 結束 Shuffle 計時，開始 Reduce 計時 ---
//     double reduce_start_time = wtime();
//     double shuffle_time_ms = (reduce_start_time - shuffle_start_time) * 1000.0;

//     // ===== Reduce Phase (Parallel) =====
//     std::vector<std::pair<std::string, std::vector<int>>> groups_vector(
//         shuffled_results.begin(), shuffled_results.end());
    
//     pthread_t* reduce_threads = new pthread_t[num_threads];
//     ReduceTaskArg* reduce_tasks = new ReduceTaskArg[num_threads];
//     std::vector<KVPair> final_results;
//     pthread_mutex_t reduce_mutex = PTHREAD_MUTEX_INITIALIZER;
    
//     size_t groups_per_thread = groups_vector.size() / num_threads;
    
//     for (int i = 0; i < num_threads; i++) {
//         reduce_tasks[i].groups = &groups_vector;
//         reduce_tasks[i].start_idx = i * groups_per_thread;
//         reduce_tasks[i].end_idx = (i == num_threads - 1) ? 
//             groups_vector.size() : (i + 1) * groups_per_thread;
//         reduce_tasks[i].result = &final_results;
//         reduce_tasks[i].mutex = &reduce_mutex;
        
//         pthread_create(&reduce_threads[i], NULL, reduce_thread, &reduce_tasks[i]);
//     }
    
//     for (int i = 0; i < num_threads; i++) {
//         pthread_join(reduce_threads[i], NULL);
//     }
    
//     // --- 結束 Reduce 計時 ---
//     double end_time = wtime();
//     double reduce_time_ms = (end_time - reduce_start_time) * 1000.0;

//     // ===== Print Results =====
//     std::cout << "\n--- Word Count MapReduce Results (" << num_threads << " threads) ---" << std::endl;
//     std::cout << std::fixed << std::setprecision(0); // 設置輸出格式
//     for (auto& pair : final_results) {
//         if (pair.value > 100) {
//             std::cout << pair.key << ": " << pair.value << std::endl;
//         }
//     }

//     std::cout << std::fixed << std::setprecision(4); // 設置時間輸出精度
//     std::cout << "Map Time    : " << map_time_ms << " ms" << std::endl;
//     std::cout << "Shuffle Time: " << shuffle_time_ms << " ms" << std::endl;
//     std::cout << "Reduce Time : " << reduce_time_ms << " ms" << std::endl;

//     // 清理資源
//     free(mutable_content); 
//     delete[] map_threads;
//     delete[] map_tasks;
//     delete[] reduce_threads;
//     delete[] reduce_tasks;
//     pthread_mutex_destroy(&reduce_mutex);

//     return 0;
// }
#include<iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_set> 
#include <cstring>  
#include <cctype>
#include <map>
#include <pthread.h>
#include <stdlib.h>
#include <algorithm>
#include <sys/time.h>
#include <iomanip>

// --- 時間函數 (掛鐘時間 Wall Time) ---
double wtime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0E-6;
}

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
    // "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    // "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    // "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt",
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

// Map 任務結構
struct MapTaskArg {
    const char* text;
    size_t start_idx;
    size_t end_idx;
    std::vector<KVPair>* local_result;
};

// Reduce 任務結構
struct ReduceTaskArg {
    std::vector<std::pair<std::string, std::vector<int>>>* groups;
    size_t start_idx;
    size_t end_idx;
    std::vector<KVPair>* result;
    pthread_mutex_t* mutex;
};

// 優化的分隔符檢查函數
inline bool is_delimiter(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || 
           c == '.' || c == ',' || c == ';' || c == ':' || 
           c == '!' || c == '?' || c == '"' || c == '(' || 
           c == ')' || c == '[' || c == ']' || c == '{' || 
           c == '}' || c == '<' || c == '>' || c == '-' || c == '\'';
}

// 優化的 Map 函數：避免 strtok_r，直接掃描字串
std::vector<KVPair> map_func_optimized(const char* text, size_t start, size_t end) {
    std::vector<KVPair> mapped_data;
    mapped_data.reserve(1000); // 預先分配空間
    
    std::string word;
    word.reserve(32); // 預先分配空間
    
    for (size_t i = start; i < end; i++) {
        char c = text[i];
        
        if (is_delimiter(c)) {
            if (!word.empty()) {
                // 檢查長度
                if (word.length() > 2) {
                    // 檢查是否為純字母
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
            // 直接轉小寫並添加
            word += tolower(c);
        }
    }
    
    // 處理最後一個單字
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

// Map thread function (優化版本)
void* map_thread(void* arg) {
    MapTaskArg* task = (MapTaskArg*)arg;
    
    // 直接呼叫優化的 map 函數
    *task->local_result = map_func_optimized(task->text, task->start_idx, task->end_idx);
    
    return NULL;
}

// Reduce function
KVPair reduce(const std::string& key, const std::vector<int>& values) {
    int sum = 0;
    for (int val : values) {
        sum += val;
    }
    return KVPair{key, sum};
}

// Reduce thread function
void* reduce_thread(void* arg) {
    ReduceTaskArg* task = (ReduceTaskArg*)arg;
    
    std::vector<KVPair> local_result;
    local_result.reserve(task->end_idx - task->start_idx);
    
    for (size_t i = task->start_idx; i < task->end_idx; i++) {
        auto& group = (*task->groups)[i];
        local_result.push_back(reduce(group.first, group.second));
    }
    
    // 將局部結果加入共享結果向量 (需要鎖定)
    pthread_mutex_lock(task->mutex);
    task->result->insert(task->result->end(), local_result.begin(), local_result.end());
    pthread_mutex_unlock(task->mutex);
    
    return NULL;
}

// Shuffle function
std::map<std::string, std::vector<int>> shuffle(const std::vector<KVPair>& mapped_data) {
    std::map<std::string, std::vector<int>> shuffled_map;
    for (const auto& pair : mapped_data) {
        shuffled_map[pair.key].push_back(pair.value);
    }
    return shuffled_map;
}

int main(int argc, char* argv[]) {
    // 初始化停用詞集合
    initialize_stopwords();
    
    // 預設執行緒數和解析參數
    int num_threads = 4;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            if (num_threads <= 0) num_threads = 1;
            i++;
        }
    }
    
    std::cout << "Running WorldCount (Pthreads Optimized) with " << num_threads << " threads" << std::endl;

    // 讀取檔案
    const int data_len =7;
    for (int i=0;i<data_len;++i){

    }
    std::ifstream ifs(dataset_path[0], std::ios::in);
    if (!ifs.is_open()) {
        std::cout << "Failed to open file.\n";
        return 1;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content(ss.str());
    ifs.close();

    const char* text = content.c_str();
    size_t content_size = content.size();

    // --- 開始 Map 計時 ---
    double map_start_time = wtime(); 

    // ===== Map Phase (Parallel) =====
    pthread_t* map_threads = new pthread_t[num_threads];
    MapTaskArg* map_tasks = new MapTaskArg[num_threads];
    std::vector<std::vector<KVPair>> thread_mapped_results(num_threads); 
    
    size_t chunk_size = content_size / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (i == num_threads - 1) ? content_size : (i + 1) * chunk_size;
        
        // 優化的邊界調整
        if (i > 0 && start_idx < content_size) {
            // 向後找到第一個分隔符
            while (start_idx < content_size && !is_delimiter(text[start_idx])) {
                start_idx++;
            }
            // 跳過連續的分隔符
            while (start_idx < content_size && is_delimiter(text[start_idx])) {
                start_idx++;
            }
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

    // 彙總 Map 結果
    std::vector<KVPair> all_mapped_results;
    size_t total_mapped_size = 0;
    for (const auto& local_vec : thread_mapped_results) {
        total_mapped_size += local_vec.size();
    }
    all_mapped_results.reserve(total_mapped_size);
    
    for (auto& local_vec : thread_mapped_results) {
        all_mapped_results.insert(all_mapped_results.end(), 
                                  std::make_move_iterator(local_vec.begin()), 
                                  std::make_move_iterator(local_vec.end()));
    }
    
    // --- 結束 Map 計時，開始 Shuffle 計時 ---
    double shuffle_start_time = wtime();
    double map_time_ms = (shuffle_start_time - map_start_time) * 1000.0;

    // ===== Shuffle Phase (Sequential) =====
    std::map<std::string, std::vector<int>> shuffled_results = shuffle(all_mapped_results);

    // --- 結束 Shuffle 計時，開始 Reduce 計時 ---
    double reduce_start_time = wtime();
    double shuffle_time_ms = (reduce_start_time - shuffle_start_time) * 1000.0;

    // ===== Reduce Phase (Parallel) =====
    std::vector<std::pair<std::string, std::vector<int>>> groups_vector(
        shuffled_results.begin(), shuffled_results.end());
    
    pthread_t* reduce_threads = new pthread_t[num_threads];
    ReduceTaskArg* reduce_tasks = new ReduceTaskArg[num_threads];
    std::vector<KVPair> final_results;
    final_results.reserve(groups_vector.size());
    pthread_mutex_t reduce_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    size_t groups_per_thread = groups_vector.size() / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        reduce_tasks[i].groups = &groups_vector;
        reduce_tasks[i].start_idx = i * groups_per_thread;
        reduce_tasks[i].end_idx = (i == num_threads - 1) ? 
        groups_vector.size() : (i + 1) * groups_per_thread;
        reduce_tasks[i].result = &final_results;
        reduce_tasks[i].mutex = &reduce_mutex;
        
        pthread_create(&reduce_threads[i], NULL, reduce_thread, &reduce_tasks[i]);
    }
    
    for (int i = 0; i < num_threads; i++) {
        pthread_join(reduce_threads[i], NULL);
    }
    
    // --- 結束 Reduce 計時 ---
    double end_time = wtime();
    double reduce_time_ms = (end_time - reduce_start_time) * 1000.0;

    // ===== Print Results =====
    std::cout << "\n--- Word Count MapReduce Results (" << num_threads << " threads) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(0);
    for (auto& pair : final_results) {
        if (pair.value > 100) {
            std::cout << pair.key << ": " << pair.value << std::endl;
        }
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout <<"Total time: " << (end_time-map_start_time)*1000.0 << " ms" << std::endl;
    std::cout << "Map Time    : " << map_time_ms << " ms" << std::endl;
    std::cout << "Shuffle Time: " << shuffle_time_ms << " ms" << std::endl;
    std::cout << "Reduce Time : " << reduce_time_ms << " ms" << std::endl;

    // 清理資源
    delete[] map_threads;
    delete[] map_tasks;
    delete[] reduce_threads;
    delete[] reduce_tasks;
    pthread_mutex_destroy(&reduce_mutex);

    return 0;
}
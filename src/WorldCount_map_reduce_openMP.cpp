#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_map>
#include <cstring>
#include <cctype>
#include <map>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <iomanip>

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

// 資料集路徑列表 (全部解開註解)
const char *dataset_path[] = {
    "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt",
    "../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_English.txt",
    "../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_English.txt",
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"
};

void initialize_stopwords() {
    if (stopwords_set.empty()) {
        for (int i = 0;  stopwords[i] != NULL; i++) {
            stopwords_set.insert( stopwords[i]);
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

inline bool is_delimiter(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '.' || c == ',' || c == ';' || c == ':' ||
           c == '!' || c == '?' || c == '"' || c == '(' ||
           c == ')' || c == '[' || c == ']' || c == '{' ||
           c == '}' || c == '<' || c == '>' || c == '-' || c == '\'';
}

std::vector<KVPair> map_func_optimized(const char* text, long start, long end) {
    std::vector<KVPair> mapped_data;
    mapped_data.reserve(1000);
    
    std::string word;
    word.reserve(32);
    
    for (long i = start; i < end; i++) {
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

std::map<std::string, std::vector<int>> shuffle(const std::vector<KVPair>& mapped_data) {
    std::map<std::string, std::vector<int>> shuffled_map;
    for (const auto& pair : mapped_data) {
        shuffled_map[pair.key].push_back(pair.value);
    }
    return shuffled_map;
}

KVPair reduce(const std::string& key, const std::vector<int>& values) {
    int sum = 0;
    for (int val : values) {
        sum += val;
    }
    return KVPair{key, sum};
}

int main(int argc, char *argv[]){
    initialize_stopwords();

    // 1. 解析命令列參數
    int user_threads = omp_get_max_threads();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            try { user_threads = std::stoi(argv[i+1]); break; } catch (...) { return 1; }
        }
    }
    if (user_threads <= 0) user_threads = 1;
    int max_threads = omp_get_max_threads();
    if (user_threads > max_threads) user_threads = max_threads;

    std::cout << "Running MapReduce with " << user_threads << " threads..." << std::endl;

    // 計算資料集數量
    int num_datasets = sizeof(dataset_path) / sizeof(dataset_path[0]);
    
    // --- 針對每個檔案進行迴圈 ---
    for (int d = 0; d < num_datasets; ++d) {
        
        // 為了美觀，印出當前處理的檔案名稱
        std::string filepath = dataset_path[d];
        // 只取檔名部分 (移除路徑)
        std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
        
        std::cout << "\n>>> Processing: " << filename << std::endl;

        // 2. 讀取檔案
        std::ifstream ifs(dataset_path[d], std::ios::in);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open file: " << dataset_path[d] << std::endl;
            continue; // 如果開檔失敗，跳過此檔案，繼續下一個
        }

        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string content(ss.str());
        ifs.close(); 
        
        if (content.empty()) {
            std::cerr << "File is empty: " << filename << std::endl;
            continue;
        }
        
        const char* text = content.c_str();
        long total_length = content.length();
        
        std::vector<std::vector<KVPair>> thread_mapped_results(user_threads);

        // --- 開始 Map 計時 ---
        double map_start_time = omp_get_wtime();

        // 3. 優化的 Parallel Map 階段
        #pragma omp parallel num_threads(user_threads)
        {
            int tid = omp_get_thread_num();
            long chunk_size = total_length / user_threads;
            long start_index = tid * chunk_size;
            long end_index = (tid == user_threads - 1) ? total_length : (tid + 1) * chunk_size;

            if (tid > 0 && start_index < total_length) {
                while (start_index < total_length && !is_delimiter(text[start_index])) start_index++;
                while (start_index < total_length && is_delimiter(text[start_index])) start_index++;
            }

            if (start_index < end_index) {
                thread_mapped_results[tid] = map_func_optimized(text, start_index, end_index);
            }
        }

        // 4. 彙總 Map 結果
        std::vector<KVPair> mapped_results;
        size_t total_size = 0;
        for (const auto& local_vec : thread_mapped_results) total_size += local_vec.size();
        mapped_results.reserve(total_size);
        
        for (const auto& local_vec : thread_mapped_results) {
            mapped_results.insert(mapped_results.end(), local_vec.begin(), local_vec.end());
        }

        // --- 結束 Map 計時，開始 Shuffle 計時 ---
        double shuffle_start_time = omp_get_wtime();
        double map_time_ms = (shuffle_start_time - map_start_time) * 1000.0;
        
        // 5. Shuffle
        std::map<std::string, std::vector<int>> shuffled_results = shuffle(mapped_results);
        
        // --- 結束 Shuffle 計時，開始 Reduce 計時 ---
        double reduce_start_time = omp_get_wtime();
        double shuffle_time_ms = (reduce_start_time - shuffle_start_time) * 1000.0;
        
        // 6. Reduce
        std::vector<KVPair> final_results;
        final_results.reserve(shuffled_results.size());
        for (const auto& group : shuffled_results) {
            final_results.push_back(reduce(group.first, group.second));
        }
        
        // --- 結束 Reduce 計時 ---
        double end_time = omp_get_wtime();
        double reduce_time_ms = (end_time - reduce_start_time) * 1000.0;

        // 7. 輸出結果
        std::cout << "--- Word Count map reduce Results ---" << std::endl;

    
        std::cout << std::fixed << std::setprecision(0);
        for (auto& pair : final_results) {
            if (pair.value > 100) {
                std::cout << pair.key << ": " << pair.value << std::endl;
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Map Time: " << map_time_ms << " ms" << std::endl;
        std::cout << "Shuffle Time: " << shuffle_time_ms << " ms" << std::endl;
        std::cout << "Reduce Time : " << reduce_time_ms << " ms" << std::endl;
    }
    
    return 0;
}
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
#include <mutex>
#include <chrono> // 引入 chrono

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

// 資料集路徑列表
const char *dataset_path[] = {
    // "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt",
    // "../datasets/KingSolomonsMines_HRiderHaggard/KingSolomonsMines_HRiderHaggard_English.txt",
    // "../datasets/OliverTwist_CharlesDickens/OliverTwist_CharlesDickens_English.txt",
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt",
    // "../datasets/Others/NotreDameDeParis_VictorHugo/NotreDameDeParis_VictorHugo_English.txt",
    // "../datasets/Others/TheThreeMusketeers_AlexandreDumas/TheThreeMusketeers_AlexandreDumas_English.txt",
    // "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"
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

// 定義三維資料結構：[Producer_TID][PartitionID][Data]
using PartitionedData = std::vector<std::vector<std::vector<KVPair>>>;

void parallel_shuffle(const std::vector<std::vector<KVPair>>& map_outputs, 
                      PartitionedData& all_thread_partitions, 
                      int num_partitions) {
    
    int num_producers = map_outputs.size();

    #pragma omp parallel num_threads(num_producers)
    {
        int tid = omp_get_thread_num();
        
        if (tid < num_producers) {
            const auto& local_vec = map_outputs[tid]; 
            
            auto& my_partitions = all_thread_partitions[tid]; 
            
            for (const auto& kv : local_vec) {
                std::size_t hash_val = std::hash<std::string>{}(kv.key);
                int partition_idx = hash_val % num_partitions;
                
                my_partitions[partition_idx].push_back(kv);
            }
        }
    }
}

std::vector<KVPair> parallel_reduce(const PartitionedData& all_thread_partitions, int num_partitions) {
    int num_threads = all_thread_partitions.size();
    std::vector<std::vector<KVPair>> thread_results(num_partitions);
    
    #pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < num_partitions; ++p) {
        std::unordered_map<std::string, int> counts;
        
        for (int t = 0; t < num_threads; ++t) {
            const auto& data_chunk = all_thread_partitions[t][p];
            for (const auto& kv : data_chunk) {
                counts[kv.key] += kv.value;
            }
        }
        
        thread_results[p].reserve(counts.size());
        for (const auto& pair : counts) {
            thread_results[p].push_back(KVPair{pair.first, pair.second});
        }
    }
    
    std::vector<KVPair> final_results;
    for (const auto& res : thread_results) {
        final_results.insert(final_results.end(), res.begin(), res.end());
    }
    return final_results;
}

int main(int argc, char *argv[]){
    initialize_stopwords();

    int user_threads = omp_get_max_threads();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            try { user_threads = std::stoi(argv[i+1]); break; } catch (...) { return 1; }
        }
    }
    if (user_threads <= 0) user_threads = 1;
    int max_threads = omp_get_max_threads();
    if (user_threads > max_threads) user_threads = max_threads;

    std::cout << "Running FULL PARALLEL MapReduce with " << user_threads << " threads (Chrono Timer)..." << std::endl;

    int num_datasets = sizeof(dataset_path) / sizeof(dataset_path[0]);
    
    for (int d = 0; d < num_datasets; ++d) {
        std::string filepath = dataset_path[d];
        std::string filename = filepath.substr(filepath.find_last_of("/\\") + 1);
        std::cout << "\n>>> Processing: " << filename << std::endl;

        std::ifstream ifs(dataset_path[d], std::ios::in);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open file: " << dataset_path[d] << std::endl;
            continue; 
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string content(ss.str());
        ifs.close(); 
        
        if (content.empty()) continue;
        
        const char* text = content.c_str();
        long total_length = content.length();
        
        // --- Map Phase ---
        std::vector<std::vector<KVPair>> thread_mapped_results(user_threads);
        
        // chrono start
        auto map_start = std::chrono::high_resolution_clock::now();

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
        
        auto map_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> map_elapsed = map_end - map_start;
        
        // --- Parallel Shuffle Phase ---
        auto shuffle_start = std::chrono::high_resolution_clock::now();

        PartitionedData partitions(user_threads, std::vector<std::vector<KVPair>>(user_threads));
        parallel_shuffle(thread_mapped_results, partitions, user_threads);
        
        auto shuffle_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> shuffle_elapsed = shuffle_end - shuffle_start;
        
        // --- Parallel Reduce Phase ---
        auto reduce_start = std::chrono::high_resolution_clock::now();

        std::vector<KVPair> final_results = parallel_reduce(partitions, user_threads);
        
        auto reduce_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> reduce_elapsed = reduce_end - reduce_start;

        // 總時間
        std::chrono::duration<double, std::milli> total_elapsed = reduce_end - map_start;

        std::cout << "--- Word Count Results (Top 5 for verify) ---" << std::endl;
        int count = 0;
        for (auto& pair : final_results) {
            if (pair.value > 100) {
                std::cout << pair.key << ": " << pair.value << std::endl;
               count++;
            }
        }
        std::cout << "Total unique words > 100: " << count << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Map Time    : " << map_elapsed.count() << " ms" << std::endl;
        std::cout << "Shuffle Time: " << shuffle_elapsed.count() << " ms" << std::endl;
        std::cout << "Reduce Time : " << reduce_elapsed.count() << " ms" << std::endl;
        std::cout << "Total Time  : " << total_elapsed.count() << " ms" << std::endl;
    }
    return 0;
}
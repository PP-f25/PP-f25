#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <iomanip> 
#include <unordered_set>
#include <unordered_map>
#include <chrono>

// namespace fs = std::filesystem;

// 定義 Key-Value Pair 結構
struct KVPair {
    char* key;
    int value;
};

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

std::unordered_set<std::string> stopwords_set;

void initialize_stopwords_seq() {
    if (stopwords_set.empty()) {
        for (int i = 0; stopwords[i] != NULL; i++) {
            stopwords_set.insert(stopwords[i]);
        }
    }
}

bool is_stopword_fast_seq(const std::string& word) {
    return stopwords_set.count(word) > 0; 
}

const char *dataset_path[] = {
    // "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
    //"../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt"
    "../enwik9/enwik9"
    // "../archive/text8"
};

bool is_stopword(const char *word) {
    for (int i = 0; stopwords[i] != NULL; i++) {
        if (strcmp(stopwords[i], word) == 0) {
            return true; 
        }
    }
    return false; 
}

// Map 函數：將文本切分成 (word, 1)
std::vector<KVPair> map_function(char* article_chunk) {
    std::vector<KVPair> mapped_data;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
    char* saveptr;
    char* token = strtok_r(article_chunk, delimiters, &saveptr);

    while (token != NULL) {
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

        if (strlen(token) <= 2 ) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        bool is_word_char = true;
        if (strlen(token) == 0) {
            is_word_char = false;
        }

        for (int i = 0; token[i]; i++) {
            if (!isalnum(token[i]) && token[i] != '_') { 
                is_word_char = false;
                break;
            }
        }

        if (!is_word_char) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        std::string current_word(token);
        if (is_stopword_fast_seq(current_word)) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        KVPair pair;
        pair.key = strdup(token); 
        pair.value = 1;
        mapped_data.push_back(pair);

        token = strtok_r(NULL, delimiters, &saveptr);
    }
    return mapped_data;
}

// Reduce 函數：將同一 key 的 values 加總
int reduce_function(const std::vector<int>& values) {
    int sum = 0;
    for (int val : values) {
        sum += val;
    }
    return sum;
}

int main(int argc, char** argv) {
    initialize_stopwords_seq();
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    auto t_total_start = std::chrono::high_resolution_clock::now();

    // 1. Read file and Scatter
    
    std::string global_str;
    if (rank == 0) {
        std::ifstream ifs(dataset_path[0], std::ios::in);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::stringstream ss;
        ss << ifs.rdbuf();
        global_str = ss.str();
    }
    /*
    std::string global_str;
    if (rank == 0) {
        // [NEW] 修改這裡：讀取 datasets 下所有檔案
        std::string root_path = "../datasets"; 
        long long total_files_count = 0;

        try {
            if (!fs::exists(root_path)) {
                 std::cerr << "Error: Directory " << root_path << " does not exist.\n";
                 MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (const auto& entry : fs::recursive_directory_iterator(root_path)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    std::ifstream ifs(entry.path());
                    if (ifs.is_open()) {
                        std::stringstream ss;
                        ss << ifs.rdbuf();
                        global_str += ss.str();
                        global_str += " "; 
                        total_files_count++;
                    }
                }
            }
            std::cout << "Rank 0: Loaded " << total_files_count << " files. Total size: " << global_str.size() << " bytes.\n";

        } catch (const std::exception& e) {
            std::cerr << "Error reading files: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    */
    auto t_map_start = std::chrono::high_resolution_clock::now();
    auto t_scatter_start = std::chrono::high_resolution_clock::now();
    int total_len = 0;
    if (rank == 0) total_len = (int)global_str.size();

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    if (rank == 0) {
        int base = total_len / size;
        int rem  = total_len % size;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < rem ? 1 : 0);
        }
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    int local_len = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> local_buf(local_len + 1);
    MPI_Scatterv(rank == 0 ? global_str.data() : nullptr, sendcounts.data(), displs.data(), MPI_CHAR, local_buf.data(), local_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    local_buf[local_len] = '\0';

    // MPI_Barrier(MPI_COMM_WORLD);

    // --- Map Phase ---
    auto t_scatter_end = std::chrono::high_resolution_clock::now();
    std::vector<KVPair> local_mapped = map_function(local_buf.data());
    auto t_map_end = std::chrono::high_resolution_clock::now();

    // --- Shuffle Phase ---
    auto t_shuffle_start = std::chrono::high_resolution_clock::now();
    
    // 3.1 Prepare send buffers
    std::vector<std::string> send_buffers(size);
    for (const auto& pair : local_mapped) {
        std::string key(pair.key);
        std::hash<std::string> hasher;
        size_t h = hasher(key);
        int dest_rank = h % size;
        
        send_buffers[dest_rank] += key + " 1\n";
        free(pair.key); 
    }

    // 3.2 Exchange data sizes
    std::vector<int> send_data_sizes(size);
    std::vector<int> recv_data_sizes(size);
    for(int i=0; i<size; i++) send_data_sizes[i] = send_buffers[i].size();

    MPI_Alltoall(send_data_sizes.data(), 1, MPI_INT, recv_data_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // 3.3 Exchange data
    std::vector<int> sdispls(size), rdispls(size);
    int total_recv_size = 0;
    int total_send_size = 0;
    
    sdispls[0] = 0;
    rdispls[0] = 0;
    for(int i=0; i<size; i++) {
        if (i > 0) {
            sdispls[i] = sdispls[i-1] + send_data_sizes[i-1];
            rdispls[i] = rdispls[i-1] + recv_data_sizes[i-1];
        }
        total_send_size += send_data_sizes[i];
        total_recv_size += recv_data_sizes[i];
    }

    std::string flat_send_buf;
    flat_send_buf.reserve(total_send_size);
    for(const auto& s : send_buffers) flat_send_buf += s;

    std::vector<char> flat_recv_buf(total_recv_size);

    MPI_Alltoallv(flat_send_buf.data(), send_data_sizes.data(), sdispls.data(), MPI_CHAR,
                  flat_recv_buf.data(), recv_data_sizes.data(), rdispls.data(), MPI_CHAR,
                  MPI_COMM_WORLD);
    
    // Parse received data
    std::map<std::string, std::vector<int>> shuffled_data;
    if (total_recv_size > 0) {
        std::string recv_str(flat_recv_buf.begin(), flat_recv_buf.end());
        std::stringstream ss(recv_str);
        std::string key;
        int val;
        while (ss >> key >> val) {
            shuffled_data[key].push_back(val);
        }
    }
    auto t_shuffle_end = std::chrono::high_resolution_clock::now();

    // --- Reduce Phase ---
    auto t_reduce_start = std::chrono::high_resolution_clock::now();
    std::vector<KVPair> final_results;
    for (const auto& group : shuffled_data) {
        int sum = reduce_function(group.second);
        KVPair res;
        res.key = strdup(group.first.c_str());
        res.value = sum;
        final_results.push_back(res);
    }

    // 5. Gather all results to Rank 0
    std::stringstream ss_local_res;
    for (const auto& pair : final_results) {
        ss_local_res << pair.key << " " << pair.value << "\n";
    }
    std::string local_res_str = ss_local_res.str();
    int local_res_len = (int)local_res_str.size();

    std::vector<int> recvcounts;
    if (rank == 0) recvcounts.resize(size);

    MPI_Gather(&local_res_len, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displss;
    std::vector<char> all_results_buf;
    if (rank == 0) {
        displss.resize(size);
        int total_len = 0;
        for (int i = 0; i < size; i++) {
            displss[i] = total_len;
            total_len += recvcounts[i];
        }
        all_results_buf.resize(total_len);
    }

    MPI_Gatherv(local_res_str.data(), local_res_len, MPI_CHAR,
                rank == 0 ? all_results_buf.data() : nullptr,
                recvcounts.data(), rank == 0 ? displss.data() : nullptr, MPI_CHAR,
                0, MPI_COMM_WORLD);

    auto t_reduce_end = std::chrono::high_resolution_clock::now();
    auto t_total_end = std::chrono::high_resolution_clock::now();

    // 6. Collect times
    double local_total_map_time = std::chrono::duration<double>(t_map_end - t_map_start).count();
    double local_scatter_time = std::chrono::duration<double>(t_scatter_end - t_scatter_start).count();
    double local_map_time = local_total_map_time - local_scatter_time;
    double local_shuffle_time = std::chrono::duration<double>(t_shuffle_end - t_shuffle_start).count();
    double local_reduce_time = std::chrono::duration<double>(t_reduce_end - t_reduce_start).count();
    double local_total_time = std::chrono::duration<double>(t_total_end - t_total_start).count();
    

    double max_map_time, max_shuffle_time, max_reduce_time, max_total_time;
    double max_scatter_time;
    double max_total_map_time;
    MPI_Reduce(&local_map_time, &max_map_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_shuffle_time, &max_shuffle_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_reduce_time, &max_reduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scatter_time, &max_scatter_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total_map_time, &max_total_map_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 7. Print Results (Rank 0 only)
    if (rank == 0) {
        std::cout << "--- MPI MapReduce Results ---" << std::endl;
        
        // Parse all_results_buf
        std::string all_res_str(all_results_buf.begin(), all_results_buf.end());
        std::stringstream ss_all(all_res_str);
        std::string key;
        int val;
        
        // Use a map to sort them for printing
        std::map<std::string, int> global_results_map;
        while (ss_all >> key >> val) {
            global_results_map[key] = val;
        }

        for (const auto& pair : global_results_map) {
            if (pair.second > 100000) {
                std::cout << pair.first << ": " << pair.second << std::endl;
            }
        }
        
        std::cout << "Total Time : " << max_total_time*1000 << " ms" << std::endl;
        std::cout << "Total Map Time : " << max_total_map_time*1000 << " ms" << std::endl;
        std::cout << "Map Time: " << max_map_time*1000 << " ms" << std::endl;
        std::cout << "Scatter Time : " << max_scatter_time*1000 << " ms" << std::endl;
        std::cout << "Shuffle Time: " << max_shuffle_time*1000 << " ms" << std::endl;
        std::cout << "Reduce Time : " << max_reduce_time*1000 << " ms" << std::endl;
        
    }

    // Cleanup
    for (const auto& pair : final_results) {
        free(pair.key);
    }

    MPI_Finalize();
    return 0;
}


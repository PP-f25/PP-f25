#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cctype>
#include <chrono>

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
    "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",NULL};

const char *dataset_path[] = {
    "datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
};
bool is_stopword(char *word) {
    for (int i = 0; stopwords[i] != NULL; i++) {
        if (strcmp(stopwords[i], word) == 0) {
            return true; 
        }
    }
    return false; 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 
    // std::cout << "Rank: " << rank << " / " << size << std::endl;

    std::chrono::high_resolution_clock::time_point t_scatter_start, t_scatter_end;
    std::chrono::high_resolution_clock::time_point t_local_start, t_local_end;
    std::chrono::high_resolution_clock::time_point t_gather_start, t_gather_end;

    // Rank 0 讀檔
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

    // 總長度
    int total_len = 0;
    if (rank == 0) 
        total_len = (int)global_str.size();

    // 廣播總長度給別人 //?
    //MPI_Bcast(&total_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 分配
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

    //
    MPI_Barrier(MPI_COMM_WORLD);
    t_scatter_start = std::chrono::high_resolution_clock::now();

    int local_len = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT,
                &local_len, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    std::vector<char> local_buf(local_len + 1, 0);

    MPI_Scatterv(rank == 0 ? global_str.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_CHAR, local_buf.data(), local_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_scatter_end = std::chrono::high_resolution_clock::now();

    local_buf[local_len] = '\0'; 

    MPI_Barrier(MPI_COMM_WORLD);
    t_local_start = std::chrono::high_resolution_clock::now();

    std::unordered_map<std::string, int> local_counts;

    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
    char* saveptr;
    char* article_buffer = local_buf.data();

    char* token = strtok_r(article_buffer, delimiters, &saveptr);
    while (token != NULL) {
        // to lower
        for (int i = 0; token[i]; i++) {
            token[i] = (char)tolower(token[i]);
        }

        if (strlen(token) <= 2) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        if (is_stopword(token)) {
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

        local_counts[std::string(token)]++;
        token = strtok_r(NULL, delimiters, &saveptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_local_end = std::chrono::high_resolution_clock::now();

    //
    MPI_Barrier(MPI_COMM_WORLD);
    t_gather_start = std::chrono::high_resolution_clock::now();

    std::stringstream ss_local;
    for (auto &p : local_counts) {
        ss_local << p.first << " " << p.second << "\n";
    }
    std::string local_str_result = ss_local.str();
    int local_result_len = (int)local_str_result.size();

    std::vector<int> recvcounts(size);
    MPI_Gather(&local_result_len, 1, MPI_INT,
            recvcounts.data(), 1, MPI_INT,
            0, MPI_COMM_WORLD);

    std::vector<int> recvdispls(size);
    int total_result_len = 0;
    if (rank == 0) {
        recvdispls[0] = 0;
        for (int i = 0; i < size; i++) {
            total_result_len += recvcounts[i];
            if (i > 0) recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
        }
    }

    std::vector<char> all_results;
    if (rank == 0) {
        all_results.resize(total_result_len);
    }

    MPI_Gatherv(local_str_result.data(), local_result_len, MPI_CHAR,
                rank == 0 ? all_results.data() : nullptr,
                recvcounts.data(), recvdispls.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_gather_end = std::chrono::high_resolution_clock::now();

    double local_scatter = std::chrono::duration<double>(t_scatter_end - t_scatter_start).count();
    double local_local   = std::chrono::duration<double>(t_local_end - t_local_start).count();
    double local_gather  = std::chrono::duration<double>(t_gather_end - t_gather_start).count();
    double max_scatter, max_local, max_gather;

    MPI_Reduce(&local_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_local,   &max_local,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gather,  &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::unordered_map<std::string, int> global_counts;
        std::string merged_str(all_results.begin(), all_results.end());
        std::stringstream ss_merge(merged_str);
        std::string w;
        int c;
        while (ss_merge >> w >> c) {
            global_counts[w] += c;
        }

        std::cout << "--- Word Count (MPI) Results ---\n";
        for (auto &p : global_counts) {
            if (p.second > 100) {
                std::cout << p.first << ": " << p.second << std::endl;
            }
        }
        std::cout << "Scatterv (distribute text): " << max_scatter*1000 << " ms" << std::endl;
        std::cout << "Local word count        : " << max_local*1000   << " ms" << std::endl;
        std::cout << "Gather + merge          : " << max_gather*1000    << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

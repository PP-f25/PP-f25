<<<<<<< HEAD
#include <iostream>
=======
#include<iostream>
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <cstring>  
#include <cctype>
#include <map>
<<<<<<< HEAD
#include <algorithm>
#include <iomanip> 
#include <chrono> // 1. ? chrono

// 蝘駁?? sys/time.h ??wtime() ?賣

// --- ?閰??---
=======
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
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
<<<<<<< HEAD

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
    "../datasets/Others/DonQuixote_MiguelCervantesSaavedra/DonQuixote_MiguelCervantesSaavedra_English.txt"
    // "../datasets/TheAdventuresOfTomSawyer_MarkTwain/TheAdventuresOfTomSawyer_MarkTwain_English.txt"
};

=======
const char *dataset_path[] = {
    "../datasets/AChristmasCarol_CharlesDickens/AChristmasCarol_CharlesDickens_English.txt"
};

bool is_stopword(char *word) {
    for (int i = 0; stopwords[i] != NULL; i++) {
        if (strcmp(stopwords[i], word) == 0) {
            return true; 
        }
    }
    return false; 
}
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
struct KVPair {
    char* key; // 
    int value;
};
<<<<<<< HEAD

//map
std::vector<KVPair> map_seq(char* article_chunk) {
=======
struct ShuffledData {
    char* key;
    std::vector<int> values; //
};
//map
std::vector<KVPair> map(char* article_chunk) {
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    std::vector<KVPair> mapped_data;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";
    char* saveptr;
    
    char* token;
<<<<<<< HEAD
    token = strtok_r(article_chunk, delimiters, &saveptr);

    while (token != NULL) {
=======
    token = strtok_r(article_chunk, delimiters, &saveptr); // )

    while (token != NULL) {
        // ... (tolower, strlen, is_stopword, isalnum) ...
        
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
        // turn to lower
        for (int i = 0; token[i]; i++) {
            token[i] = tolower(token[i]);
        }

<<<<<<< HEAD
        // 1. ?蕪?瑕漲
        if (strlen(token) <= 2) {
=======
        // get the next token
        if (strlen(token) <= 2) {
            
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

<<<<<<< HEAD
        // 3. ?蕪?急??詨??畾泵??閰?
        bool is_word_char = true;
        if (strlen(token) == 0) { 
=======
        // check if stop words
        if (is_stopword(token)) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        // fliter out the word that has number or special char eg:123, c--
        bool is_word_char = true;
        if (strlen(token) == 0) { // strtok_r somtime generate empty token
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
             is_word_char = false;
        }

        for (int i = 0; token[i]; i++) {
<<<<<<< HEAD
=======
            // isalnum() check if alpha or num
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
            if (!isalnum(token[i]) && token[i] != '_') { 
                is_word_char = false;
                break;
            }
        }

        if (!is_word_char) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }
<<<<<<< HEAD
        
        // 2. ?閰炎??
        std::string current_word(token);
        if (is_stopword_fast_seq(current_word)) {
            token = strtok_r(NULL, delimiters, &saveptr);
            continue;
        }

        KVPair pair;
=======
        // "(Emit)" (key, 1)
        KVPair pair;
        // strdup = (malloc + strcpy)，to build a new memory
        // 
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
        pair.key = strdup(token); 
        pair.value = 1;
        
        mapped_data.push_back(pair);

        token = strtok_r(NULL, delimiters, &saveptr);
    }
    return mapped_data;
}


//shuffle
<<<<<<< HEAD
std::map<std::string, std::vector<int>> shuffle_seq(const std::vector<KVPair>& mapped_data) {
    std::map<std::string, std::vector<int>> shuffled_map;

    for (const auto& pair : mapped_data) {
        shuffled_map[std::string(pair.key)].push_back(pair.value);
=======
std::map<std::string, std::vector<int>> shuffle(const std::vector<KVPair>& mapped_data) {
    
    // use std::map 
    //  sort by key
    std::map<std::string, std::vector<int>> shuffled_map;

    for (const auto& pair : mapped_data) {
        // make  (key, 1) insert to  key  vector 
        shuffled_map[std::string(pair.key)].push_back(pair.value);
        
        // release pair key;
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
        free(pair.key); 
    }
    return shuffled_map;
}

//reduce
<<<<<<< HEAD
KVPair reduce_seq(const std::string& key, const std::vector<int>& values) {
=======
KVPair reduce(const std::string& key, const std::vector<int>& values) {
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    int sum = 0;
    for (int val : values) {
        sum += val;
    }

    KVPair result;
<<<<<<< HEAD
    result.key = strdup(key.c_str());
=======
    result.key = strdup(key.c_str()); // 再次複製
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    result.value = sum;
    
    return result;
}


int main(){
<<<<<<< HEAD
    initialize_stopwords_seq();

    std::cout << "Running sequential MapReduce version" << std::endl;

    // read file
=======

    // read file

>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    std::ifstream ifs(dataset_path[0], std::ios::in);
    if (!ifs.is_open()) {
        std::cout << "Failed to open file.\n";
        return 1;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string str(ss.str());
<<<<<<< HEAD
    ifs.close(); 
    
    char* article_buffer = strdup(str.c_str());

    // 2. 雿輻 chrono::high_resolution_clock
    // --- ?? Map 閮? ---
    auto map_start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Map
    std::vector<KVPair> mapped_results = map_seq(article_buffer);

    // --- 蝯? Map 閮?嚗?憪?Shuffle 閮? ---
    auto shuffle_start_time = std::chrono::high_resolution_clock::now();
    
    // 閮? Map ?? (雿輻 duration 頧 double 瘥怎?)
    std::chrono::duration<double, std::milli> map_duration = shuffle_start_time - map_start_time;
    
    // 2. Shuffle
    std::map<std::string, std::vector<int>> shuffled_results = shuffle_seq(mapped_results);
    
    // --- 蝯? Shuffle 閮?嚗?憪?Reduce 閮? ---
    auto reduce_start_time = std::chrono::high_resolution_clock::now();
    
    // 閮? Shuffle ??
    std::chrono::duration<double, std::milli> shuffle_duration = reduce_start_time - shuffle_start_time;
=======
    
    ifs.close(); 

    
    std::unordered_map<std::string, int> word_counts;



    // split it into words
    char* token;
    const char* delimiters = " \t\n\r.,;:!?\"()[]{}<>-'";

    char* article_buffer = &str[0];
    std::vector<KVPair> mapped_results = map(article_buffer);

    // 2. Shuffle
    std::map<std::string, std::vector<int>> shuffled_results = shuffle(mapped_results);
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    
    // 3. Reduce
    std::vector<KVPair> final_results;
    for (const auto& group : shuffled_results) {
<<<<<<< HEAD
        final_results.push_back(reduce_seq(group.first, group.second));
    }

    // --- 蝯? Reduce 閮? ---
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 閮? Reduce ???蜇??
    std::chrono::duration<double, std::milli> reduce_duration = end_time - reduce_start_time;
    std::chrono::duration<double, std::milli> total_duration = end_time - map_start_time;
    
    // 4. print
    std::cout << "\n--- Word Count MapReduce Results (Sequential) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(0); 

=======
        final_results.push_back(reduce(group.first, group.second));
    }

    // 4. print
    std::cout << "--- Word Count Results ---" << std::endl;
>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    for (auto& pair : final_results) {
        if (pair.value > 100) {
            std::cout << pair.key << ": " << pair.value << std::endl;
        }
<<<<<<< HEAD
        free(pair.key);
    }
    
    std::cout << std::fixed << std::setprecision(4); 
    // 3. 頛詨 .count() ??瘥怎??詨?
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Map Time    : " << map_duration.count() << " ms" << std::endl;
    std::cout << "Shuffle Time: " << shuffle_duration.count() << " ms" << std::endl;
    std::cout << "Reduce Time : " << reduce_duration.count() << " ms" << std::endl;

    free(article_buffer);
=======
        free(pair.key); // release reduce  key
    }

>>>>>>> 9a100b9e2622780c7a535cd868af30f3dc9b03a5
    return 0;
}
#pragma once

#include <filesystem>

namespace Util {

inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

inline bool isValidMp4Filename(std::string filename) {
    // Trim spaces
    filename = trim(filename);

    const std::string invalidChars = "\\/:*?\"<>|";
    
    if (filename.size() < 5) return false; 

    if (filename.find_first_of(invalidChars) != std::string::npos) {
        return false;
    }

    std::string ext = filename.substr(filename.size() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".mp4";
}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        spdlog::error(errMsg);
        throw std::runtime_error(errMsg);
    }
}

inline std::vector<std::string> getFilesInDirectory(const std::string &dirPath) {
        std::vector<std::string> fileNames;
        for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
            if (entry.is_regular_file()) {
                fileNames.push_back(entry.path().string());
            }
        }
        return fileNames;
    }
}

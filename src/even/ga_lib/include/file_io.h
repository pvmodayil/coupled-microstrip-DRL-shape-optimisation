#ifndef FILE_IO_H
#define FILE_IO_H

#include <vector>
#include <string>
#include <unordered_map>

namespace fileio{

// Function to read a CSV file and store each column as a vector
std::unordered_map<std::string, std::vector<double>> read_csv(const std::string& filename);

// Function to write a map of vector to a CSV file
void write_csv(const std::string& filename, const std::unordered_map<std::string, std::vector<double>>& data);

} // namespace fileio

#endif
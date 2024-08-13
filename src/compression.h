#ifndef COMPRESS_H
#define COMPRESS_H

#include <iostream>
#include <vector>
#include <bitset>
#include <unordered_map>
#include <zlib.h>
#include <fstream>
#include <lzma.h>
#include <cmath>

std::unordered_map<float, uint8_t> mapValuesToBinary(const std::vector<float>& distinct_values) {
    if (distinct_values.size() != 2) {
        throw std::invalid_argument("There must be exactly 2 distinct float values.");
    }

    std::unordered_map<float, uint8_t> value_to_binary;
    value_to_binary[distinct_values[0]] = 0;
    value_to_binary[distinct_values[1]] = 1;

    return value_to_binary;
}

std::vector<uint8_t> packTensor1D(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::unordered_map<float, uint8_t>& value_to_binary) {

    int depth = input.size();
    int height = input[0].size();
    int width = input[0][0].size();
    int total_bits = depth * height * width;
    int packed_size = (total_bits + 7) / 8;

    std::vector<uint8_t> packed_tensor(packed_size, 0);

    int bit_index = 0;
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int packed_index = bit_index / 8;
                int bit_offset = bit_index % 8;

                packed_tensor[packed_index] |= (value_to_binary.at(input[d][h][w]) << bit_offset);
                ++bit_index;
            }
        }
    }

    return packed_tensor;
}

double compressionRatio(const std::vector<uint8_t>& original_data, const std::vector<uint8_t>& compressed_data) {
    double original_size = original_data.size();
    double compressed_size = compressed_data.size();
    return ((original_size - compressed_size) / original_size) * 100.0;
}


std::vector<uint8_t> compressDataLZMA(const std::vector<uint8_t>& data) {
    size_t compressed_size = lzma_stream_buffer_bound(data.size());
    std::vector<uint8_t> compressed_data(compressed_size);

    lzma_ret ret = lzma_easy_buffer_encode(LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64,
                                           nullptr, data.data(), data.size(),
                                           compressed_data.data(), &compressed_size, compressed_data.size());

    if (ret != LZMA_OK) {
        throw std::runtime_error("LZMA compression failed!");
    }

    compressed_data.resize(compressed_size);
    return compressed_data;
}


std::vector<uint8_t> compressData(const std::vector<uint8_t>& data) {
    uLongf compressed_size = compressBound(data.size());
    std::vector<uint8_t> compressed_data(compressed_size);

    int result = compress(compressed_data.data(), &compressed_size, data.data(), data.size());
    if (result != Z_OK) {
        throw std::runtime_error("Compression failed!");
    }

    compressed_data.resize(compressed_size);
    return compressed_data;
}

#endif

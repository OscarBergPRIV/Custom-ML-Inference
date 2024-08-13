#include <iostream>
#include "layers.h"
#include <random>
#include "compression.h"

class Network {
public:
    std::vector<Conv2d> conv_layers;
    std::vector<BatchNorm2d> batchnorm_layers;
    std::vector<SiLU> silu_layers;
    std::vector<MaxPool2d> maxpool_layers;
    std::vector<QUANT> quant;

    Network(int num_bits, std::vector<float> thetas, std::vector<float> scale, float b) {
    
        conv_layers.emplace_back(3, 1, 3, 16, 1, true);
        conv_layers.back().initialize_from_file("params/weights.txt", "params/bias.txt");
        batchnorm_layers.emplace_back(16);
        silu_layers.emplace_back();
        maxpool_layers.emplace_back(2, 2);

        conv_layers.emplace_back(3, 1, 16, 32, 1, false);
        conv_layers.back().initialize_kernel_random();
        batchnorm_layers.emplace_back(32);
        silu_layers.emplace_back();
        maxpool_layers.emplace_back(2, 2);

        conv_layers.emplace_back(3, 1, 32, 64, 1, false);
        conv_layers.back().initialize_kernel_random();
        batchnorm_layers.emplace_back(64);
        silu_layers.emplace_back();
        maxpool_layers.emplace_back(2, 2);

        conv_layers.emplace_back(3, 1, 64, 128, 1, false);
        conv_layers.back().initialize_kernel_random();
        batchnorm_layers.emplace_back(128);
        silu_layers.emplace_back();
        maxpool_layers.emplace_back(2, 2);
        
        quant.emplace_back(num_bits, thetas, scale, b);

    }

    void forward(std::vector<std::vector<std::vector<float>>>& input) {
        for (int i = 0; i < conv_layers.size(); ++i) {
            std::vector<std::vector<std::vector<float>>> conv_output;
            conv_layers[i].convolve(input, conv_output);
            batchnorm_layers[i].normalize(conv_output);
            silu_layers[i].apply(conv_output);
            std::vector<std::vector<std::vector<float>>> maxpool_output;
            maxpool_layers[i].pool(conv_output, maxpool_output);
            if (i == conv_layers.size() - 1) {
                std::vector<std::vector<std::vector<float>>> quant_output;
                quant[0].apply(maxpool_output, quant_output);
                input = quant_output;
            } else {
                input = maxpool_output;
            }
        }
    }
};

int main() {

    std::vector<float> thetas = {0.0f};
    std::vector<float> scale = {10.0f};
    float b = -5.0f;
    Network network(1, thetas, scale, b);
    network.conv_layers[0].print_kernel();

    const int channels = 3;
    const int height = 640;
    const int width = 640;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 1.0); // mean 0, stddev 1

    std::vector<std::vector<std::vector<float>>> input(channels, 
        std::vector<std::vector<float>>(height, 
        std::vector<float>(width)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input[c][h][w] = dist(gen);
            }
        }
    }
    std::cout << input[0][0][0] << "\n";

    network.forward(input);
    if (!input.empty() && !input[0].empty() && !input[0][0].empty()) {
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        std::cout << "Shape: (" << depth << ", " << height << ", " << width << ")\n";
    }
    printf("%f\n", input[0][0][0]);
    printf("%f\n", input[20][10][15]);
    
    std::vector<float> distinct_values = {-5.0f, 5.0f};

    std::unordered_map<float, uint8_t> value_to_binary = mapValuesToBinary(distinct_values);

    std::vector<uint8_t> packed_tensor = packTensor1D(input, value_to_binary);

    std::vector<uint8_t> compressed_tensor = compressData(packed_tensor);

    double ratio_zip = compressionRatio(packed_tensor, compressed_tensor);

    std::cout << ratio_zip << "\n";
    
    return 0;
}

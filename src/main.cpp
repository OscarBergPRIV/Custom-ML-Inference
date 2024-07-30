#include <iostream>
#include "layers.h"
#include <random>

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

    std::vector<float> thetas = {0.1f, 0.3f, 0.5f};
    std::vector<float> scale = {1.0f, 0.8f, 0.6f};
    float b = 0.2f;
    Network network(2, thetas, scale, b);
    network.conv_layers[0].print_kernel();
    /*
    std::vector<std::vector<std::vector<float>>> input = {
        {
            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {13.0, 14.0, 15.0, 16.0, 5.0, -60.0, 7.0, 8.0, 9.0, 10.0},
            {5.0, 6.0, 7.0, -80.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {9.0, 10.0, 11.0, 12.0, 5.0, -60.0, 70.0, 8.0, 9.0, 10.0},
            {1.0, 10.0, 5.0, 4.0, -56.0, 100.0, 2.0, 3.0, 4.0, 5.0},
            {-9.0, 10.0, 15.0, 12.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
            {-1.0, 10.0, 50.0, 4.0, 56.0, 1.0, 2.0, 3.0, 4.0, 5.0},
            {1.0, 2.0, 30.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
        }
    };*/


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

    /*
    
    std::cout << "Output:\n";
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            printf("channel values: ");
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
        }
    }*/


    return 0;
}

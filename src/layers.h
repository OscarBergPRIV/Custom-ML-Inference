#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <ctime>

class Conv2d {
private:
    int kernel_size;
    int in_chan;
    int out_chan;
    int stride;
    bool use_bias;
    int padding;
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel;
    std::vector<float> bias;

public:
    Conv2d(int k_size, int s, int in_ch, int out_ch, int p, bool use_bias = true)
        : kernel_size(k_size), stride(s), in_chan(in_ch), out_chan(out_ch), padding(p), use_bias(use_bias) {
        kernel = std::vector<std::vector<std::vector<std::vector<float>>>>(out_chan, 
                 std::vector<std::vector<std::vector<float>>>(in_chan, 
                 std::vector<std::vector<float>>(kernel_size, 
                 std::vector<float>(kernel_size))));
        initialize_kernel_random();

        if (use_bias) {
            printf("USE BIAS\n");
            bias = std::vector<float>(out_chan, 0);
        }
    }

    void apply_padding(const std::vector<std::vector<std::vector<float>>>& input, 
                       std::vector<std::vector<std::vector<float>>>& padded_input) {
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int padded_height = input_height + 2 * padding;
        int padded_width = input_width + 2 * padding;

        padded_input = std::vector<std::vector<std::vector<float>>>(in_chan, 
                       std::vector<std::vector<float>>(padded_height, 
                       std::vector<float>(padded_width, 0)));

        for (int c = 0; c < in_chan; ++c) {
            for (int y = 0; y < input_height; ++y) {
                for (int x = 0; x < input_width; ++x) {
                    padded_input[c][y + padding][x + padding] = input[c][y][x];
                }
            }
        }
    }


    void convolve(const std::vector<std::vector<std::vector<float>>>& input, 
                  std::vector<std::vector<std::vector<float>>>& output) {
        std::vector<std::vector<std::vector<float>>> padded_input;
        apply_padding(input, padded_input);

        int input_height = padded_input[0].size();
        int input_width = padded_input[0][0].size();
        int output_height = (input_height - kernel_size) / stride + 1;
        int output_width = (input_width - kernel_size) / stride + 1;

        output = std::vector<std::vector<std::vector<float>>>(out_chan, 
                 std::vector<std::vector<float>>(output_height, 
                 std::vector<float>(output_width, 0)));

        for (int o = 0; o < out_chan; ++o) {
            for (int i = 0; i < in_chan; ++i) {
                for (int y = 0; y < output_height; ++y) {
                    for (int x = 0; x < output_width; ++x) {
                        float sum = 0.0f;
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride + ky;
                                int in_x = x * stride + kx;
                                sum += padded_input[i][in_y][in_x] * kernel[o][i][ky][kx];
                            }
                        }
                        output[o][y][x] = sum + (use_bias ? bias[o] : 0.0f);
                    }
                }
            }
        }
    }

    void set_bias(const std::vector<float>& new_bias) {
        if (use_bias) {
            bias = new_bias;
        }
    }

    void initialize_from_file(const std::string& weights_file, const std::string& bias_file) {
        std::ifstream weights_ifs(weights_file);
        if (!weights_ifs.is_open()) {
            throw std::runtime_error("Could not open weights file.");
        }

        for (int o = 0; o < out_chan; ++o) {
            for (int i = 0; i < in_chan; ++i) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        if (!(weights_ifs >> kernel[o][i][ky][kx])) {
                            printf("FILE ERROR\n");
                            throw std::runtime_error("Weights file does not have enough values.");
                        }
                    }
                }
            }
        }
        weights_ifs.close();

        if (use_bias) {
            std::ifstream bias_ifs(bias_file);
            if (!bias_ifs.is_open()) {
                throw std::runtime_error("Could not open bias file.");
            }

            for (int o = 0; o < out_chan; ++o) {
                if (!(bias_ifs >> bias[o])) {
                    printf("FILE ERROR BIAS\n");
                    throw std::runtime_error("Bias file does not have enough values.");
                }
            }
            bias_ifs.close();
        }
    }

    void initialize_kernel_random() {
        for (int o = 0; o < out_chan; ++o) {
            for (int i = 0; i < in_chan; ++i) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        kernel[o][i][ky][kx] = static_cast<float>(rand()) / RAND_MAX;
                    }
                }
            }
        }
    }
    void print_kernel() {
        for (int o = 0; o < out_chan; ++o) {
            std::cout << "Kernel for output channel " << o << ":\n";
            for (int i = 0; i < in_chan; ++i) {
                std::cout << "  Input channel " << i << ":\n";
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        std::cout << kernel[o][i][ky][kx] << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
    }
};

class MaxPool2d {
private:
    int pool_size;
    int stride;

public:
    MaxPool2d(int p_size, int s)
        : pool_size(p_size), stride(s) {}

    void pool(const std::vector<std::vector<std::vector<float>>>& input, 
              std::vector<std::vector<std::vector<float>>>& output) {
        int in_chan = input.size();
        int input_height = input[0].size();
        int input_width = input[0][0].size();
        int output_height = (input_height - pool_size) / stride + 1;
        int output_width = (input_width - pool_size) / stride + 1;

        output = std::vector<std::vector<std::vector<float>>>(in_chan, 
                 std::vector<std::vector<float>>(output_height, 
                 std::vector<float>(output_width, 0)));

        for (int c = 0; c < in_chan; ++c) {
            for (int y = 0; y < output_height; ++y) {
                for (int x = 0; x < output_width; ++x) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int py = 0; py < pool_size; ++py) {
                        for (int px = 0; px < pool_size; ++px) {
                            int in_y = y * stride + py;
                            int in_x = x * stride + px;
                            max_val = std::max(max_val, input[c][in_y][in_x]);
                        }
                    }
                    output[c][y][x] = max_val;
                }
            }
        }
    }
};

class BatchNorm2d {
private:
    int num_features;
    float epsilon;
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> running_mean;
    std::vector<float> running_var;

public:
    BatchNorm2d(int num_feat, float eps = 1e-5)
        : num_features(num_feat), epsilon(eps) {
        gamma.resize(num_features, 1.0);
        beta.resize(num_features, 0.0);
        running_mean.resize(num_features, 0.0);
        running_var.resize(num_features, 1.0);
    }

    void normalize(std::vector<std::vector<std::vector<float>>>& input) {
        int batch_size = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        std::vector<float> batch_mean(num_features, 0.0);
        std::vector<float> batch_var(num_features, 0.0);

        for (int c = 0; c < num_features; ++c) {
            float mean = 0.0;
            float var = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        mean += input[b][y][x];
                    }
                }
            }
            mean /= (batch_size * height * width);
            batch_mean[c] = mean;

            for (int b = 0; b < batch_size; ++b) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        var += (input[b][y][x] - mean) * (input[b][y][x] - mean);
                    }
                }
            }
            var /= (batch_size * height * width);
            batch_var[c] = var;
        }

        for (int c = 0; c < num_features; ++c) {
            for (int b = 0; b < batch_size; ++b) {
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        input[b][y][x] = gamma[c] * ((input[b][y][x] - batch_mean[c]) / std::sqrt(batch_var[c] + epsilon)) + beta[c];
                    }
                }
            }
        }

        for (int c = 0; c < num_features; ++c) {
            running_mean[c] = 0.9 * running_mean[c] + 0.1 * batch_mean[c];
            running_var[c] = 0.9 * running_var[c] + 0.1 * batch_var[c];
        }
    }

    void print_parameters() {
        std::cout << "Gamma: ";
        for (const auto& g : gamma) {
            std::cout << g << " ";
        }
        std::cout << "\nBeta: ";
        for (const auto& b : beta) {
            std::cout << b << " ";
        }
        std::cout << "\nRunning Mean: ";
        for (const auto& rm : running_mean) {
            std::cout << rm << " ";
        }
        std::cout << "\nRunning Var: ";
        for (const auto& rv : running_var) {
            std::cout << rv << " ";
        }
        std::cout << "\n";
    }
};

class SiLU {
public:
    void apply(std::vector<std::vector<std::vector<float>>>& input) {
        for (auto& channel : input) {
            for (auto& row : channel) {
                for (auto& value : row) {
                    value = silu(value);
                }
            }
        }
    }

private:
    float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }
};


class QUANT {
private:
    int num_bits;
    std::vector<float> thetas;
    std::vector<float> scale;
    float b;

public:
    QUANT(int num_bits, const std::vector<float>& thetas, const std::vector<float>& scale, float b)
        : num_bits(num_bits), thetas(thetas), scale(scale), b(b) {

        if ((1 << num_bits) - 1 != thetas.size()) {
            throw std::invalid_argument("Quant-precision (num_bits) does not match given thetas!");
        }

        if (thetas.size() != scale.size()) {
            throw std::invalid_argument("Size of thetas and scale are not equal!");
        }
    }
    void apply(const std::vector<std::vector<std::vector<float>>>& input, 
               std::vector<std::vector<std::vector<float>>>& output) {

        int in_chan = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        output = std::vector<std::vector<std::vector<float>>>(in_chan, 
                 std::vector<std::vector<float>>(height, 
                 std::vector<float>(width, 0)));

        for (int c = 0; c < in_chan; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    output[c][h][w] = step(input[c][h][w]);
                }
            }
        }
    }

    float step(float input) {
        float sum = 0;
        for (int i = 0; i < thetas.size()-1; ++i) {
            sum += heaviside(input-thetas[i]) * scale[i];
        }
        return sum + b;
    }

    float heaviside(float input) {
        if (input > 0.0f) {
            return 1.0f;
        } else {
            return 0.0f;
        }
    }

};



#endif

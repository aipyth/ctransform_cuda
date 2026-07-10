#pragma once
#include <iostream>
#include <string>
#include <vector>

template <typename T>
void printVector(std::string s, const std::vector<T>& vec) {
    std::cout << s << ": ";
    for (T x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

template <typename T>
void printVector2D(
    std::string s,
    const std::vector<T>& vec,
    std::size_t nx,
    std::size_t ny
) {
    std::cout << s << std::endl;;
    for (std::size_t ix = 0; ix < nx; ix++) {
        std::cout << "| ";
        for (std::size_t iy = 0; iy < ny; iy++) {
            std::cout << vec[ix * ny + iy] << " ";
        }
        std::cout << "|" << std::endl;
    }
}

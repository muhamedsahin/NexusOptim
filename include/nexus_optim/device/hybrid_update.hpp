#pragma once

/**
 * @file hybrid_update.hpp
 * @brief Bir parametre tamponunu ayni adimda CPU SIMD ve CUDA arasinda boler.
 *
 * GPU dilimi cihaz cekirdeginde, CPU dilimi AVX cekirdeginde eszamanli guncellenir.
 * Cihaz tamponu tekrar kullanilir. Ayni host isaretcileriyle gelen sonraki adim
 * parametre ve momentleri yeniden yuklemez; yalnizca gradyan gider. Adim sonunda
 * parametre ve momentler hosta yazilir. Host bu arada baska kodla degistirilirse
 * sonuc cihazdaki eski kopyadan uretilir. CUDA derlenmemisse tum tampon CPU'da kalir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus_optim/simd/kernel_config.hpp"

#include <cstddef>

namespace nexus_optim {

struct HybridCounts {
  std::size_t cpu = 0;
  std::size_t gpu = 0;
};

HybridCounts adam_update_hybrid(float* params, const float* grads, float* exp_avg, float* exp_avg_sq,
                                float* max_exp_avg_sq, std::size_t n, AdamCoeffs coeffs);

HybridCounts sgd_update_hybrid(float* params, const float* grads, float* velocity, std::size_t n,
                               SgdCoeffs coeffs);

}  // namespace nexus_optim

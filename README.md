<div align="center">

# 🚀 NexusOptim

### Hesaplanmış gradyanları parametre tamponlarına uygulayan C++20 optimizer kütüphanesi

*14 optimizer · 7 zamanlayıcı · skaler, AVX2, AVX-512 ve CUDA güncelleme çekirdekleri.*

[![Version](https://img.shields.io/badge/s%C3%BCr%C3%BCm-1.0.0-blueviolet?style=for-the-badge)](CMakeLists.txt)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2_%7C_AVX--512-FF6F00?style=for-the-badge)](src/simd)
[![CUDA](https://img.shields.io/badge/CUDA-sm__86-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](src/cuda)
[![License](https://img.shields.io/badge/Lisans-Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

[Optimizer'lar](#-optimizer-türleri) • [Zamanlayıcılar](#-zamanlayıcılar) • [Kurulum](#-kurulum) • [Örnek](#-step-örneği) • [Kılavuz](docs/kilavuz.md) • [Nexus](https://github.com/muhamedsahin/Nexus)

</div>

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

## 🎯 Kapsam

Ad alanı `nexus_optim`, CMake hedefi `nexus_optim::nexus_optim`, çatı başlığı `nexus_optim/nexus_optim.hpp`.
Tensör ayırma, matris çarpımı, otomatik türev ve veri yükleme bu kütüphanenin **dışındadır**; parametre ve gradyan işaretçileri çağırana aittir.

Durum sözlüğü, SIMD seçimi, CUDA, Python iş parçacıkları ve bu makinedeki ölçüm: [docs/kilavuz.md](docs/kilavuz.md).

---

## 🧮 Optimizer türleri

| Aile | Türler | Başlık |
| --- | --- | --- |
| 📏 SGD | `SGD`, `MomentumSGD`, `NesterovSGD` | `algorithms/sgd.hpp` |
| 📊 Uyarlanır | `Adagrad`, `RMSProp`, `Adadelta` | `algorithms/{adagrad,rmsprop,adadelta}.hpp` |
| 🧠 Adam ailesi | `Adam`, `AdamW`, `NAdam`, `RAdam`, `AdaBelief` | `algorithms/{adam,nadam,radam,adabelief}.hpp` |
| 🏋 Büyük batch | `LAMB`, `LARS` | `algorithms/{lamb,lars}.hpp` |
| 🦁 İşaret tabanlı | `Lion` | `algorithms/lion.hpp` |
| 👀 Sarmalayıcı | `Lookahead` | `algorithms/lookahead_wrapper.hpp` |

`adamw.hpp`, `momentum_sgd.hpp` ve `nesterov_sgd.hpp` yalnız ilgili başlığı dahil eder; kendi sınıfları yoktur.
Üst depodaki `include/nexus/optim/optimizers.hpp` cephesi `AdamW` ve `SGD` bağlar; `RAdam` ve `NAdam` Python eklentisine bağlı değildir.

> 🔬 **Nexus FAZ 3 doğrulaması (24 Ekim 2026):** Nexus'un çok-tensörlü füzyonlu AdamW'i, bu kütüphanenin skaler AdamW yoluyla 1000 adımda **bit-bit aynı**, AVX2 yoluyla ≤ 4.6e-7 bağıl farkla eşleşir. 124M parametrede bu kütüphanenin AVX2 yolu 164 ms/adım ile Nexus foreach yolundan (260 ms) hızlıdır.

---

## ⏱ Zamanlayıcılar

| Tür | Başlık |
| --- | --- |
| `StepLR` | `scheduler/step_lr.hpp` |
| `ExponentialLR` | `scheduler/exponential_lr.hpp` |
| `CosineAnnealingLR` | `scheduler/cosine_annealing_lr.hpp` |
| `CosineAnnealingWarmRestarts` | `scheduler/cosine_warm_restarts.hpp` |
| `OneCycleLR` | `scheduler/one_cycle_lr.hpp` |
| `LinearWarmup` | `scheduler/linear_warmup.hpp` |
| `ReduceLROnPlateau` | `scheduler/reduce_lr_on_plateau.hpp` |

Ortak taban `LRSchedulerBase`. Zamanlayıcı grup alanındaki `learning_rate` değerini yerinde yazar; `step()` bu alanı okur.

---

## 🛠 Kurulum

C++20 derleyicisi ve CMake 3.20.

```bat
:: CPU
cmake -S . -B build -DNEXUS_OPTIM_WITH_CUDA=OFF
cmake --build build --config Release --parallel
cmake --install build --config Release --prefix install

:: CUDA (CMAKE_CUDA_ARCHITECTURES verilmezse 86)
cmake -S . -B build-cuda -DNEXUS_OPTIM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --config Release --parallel
```

```cmake
add_subdirectory(path/to/NexusOptim)
target_link_libraries(my_app PRIVATE nexus_optim::nexus_optim)
```

| Seçenek | Varsayılan | Anlamı |
| --- | --- | --- |
| `NEXUS_OPTIM_WITH_CUDA` | `OFF` | CUDA güncelleme çekirdekleri (`src/cuda/cuda_optimizer_kernels.cu`) |
| `NEXUS_OPTIM_BUILD_TESTS` | `ON` | `tests/` GoogleTest hedefleri |
| `NEXUS_OPTIM_BUILD_BENCHMARKS` | `ON` | `benchmarks/` programları |
| `NEXUS_OPTIM_BUILD_EXAMPLES` | `ON` | `examples/` programları |
| `NEXUS_OPTIM_ENABLE_AVX512` | `ON` | AVX-512 çeviri birimi; çalışma anında CPUID ile seçilir |
| `NEXUS_OPTIM_WARNINGS_AS_ERRORS` | `ON` | Uyarılar hata sayılır |
| `NEXUS_MATRIXFLASH_INCLUDE` | boş | Doluysa MatrixFlash-Pro başlık yolu |

---

## 🚀 `step()` örneği

`SGD::Options::lr` saklanır; güncellemenin öğrenme hızı `ParamGroupOptions::learning_rate` alanından okunur.
Ağırlık sönümü, grupta `weight_decay_override` yoksa `Options::weight_decay` değeridir.

```cpp
#include <nexus_optim/nexus_optim.hpp>

#include <vector>

int main() {
    std::vector<float> params{1.f, 2.f, 3.f, 4.f};
    std::vector<float> grads{0.1f, -0.2f, 0.3f, -0.4f};

    nexus_optim::ParamGroupOptions group_options;
    group_options.learning_rate = 0.1;

    nexus_optim::SGD<float>::Options options;
    options.lr = 0.1;
    options.weight_decay = 1e-4;

    nexus_optim::SGD<float> optimizer(
        {nexus_optim::make_group(params.data(), grads.data(), params.size(), group_options)},
        options);

    optimizer.step();
}
```

`make_group` CPU grubu kurar. `step()` argümansızdır ve `OptimizerBase` üzerinden `step_impl`'e gider.
SIMD yolu `force_simd_level(...)` ile benchmark/test için sabitlenebilir.

## 📄 Lisans

Apache-2.0 — metin [`LICENSE`](LICENSE) dosyasındadır.

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>

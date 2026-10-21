# NexusOptim

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

NexusOptim, hesaplanmış gradyanları parametre tamponlarına uygulayan bir C++20 kütüphanesidir. Sürüm `1.0.0`, ad alanı `nexus_optim`, CMake hedefi `nexus_optim::nexus_optim`. Çatı başlığı `nexus_optim/nexus_optim.hpp`. Tensör ayırma, matris çarpımı, otomatik türev ve veri yükleme bu kütüphanenin dışında kalır. Parametre ve gradyan işaretçileri çağırana aittir.

Lisans Apache-2.0. Metin `LICENSE` dosyasındadır ve bu belgede değiştirilmemiştir.

Ayrıntılı davranış (durum sözlüğü, SIMD seçimi, CUDA, Python iş parçacıkları ve bu makinedeki ölçüm) `docs/kilavuz.md` içindedir.

## Optimizer türleri

`include/nexus_optim/algorithms` altındaki başlıklardan sayılan, kurulabilir türler:

| Tür | Başlık |
|---|---|
| `SGD` | `algorithms/sgd.hpp` |
| `MomentumSGD` | `algorithms/sgd.hpp` |
| `NesterovSGD` | `algorithms/sgd.hpp` |
| `Adagrad` | `algorithms/adagrad.hpp` |
| `RMSProp` | `algorithms/rmsprop.hpp` |
| `Adadelta` | `algorithms/adadelta.hpp` |
| `Adam` | `algorithms/adam.hpp` |
| `AdamW` | `algorithms/adam.hpp` |
| `NAdam` | `algorithms/nadam.hpp` |
| `RAdam` | `algorithms/radam.hpp` |
| `AdaBelief` | `algorithms/adabelief.hpp` |
| `LAMB` | `algorithms/lamb.hpp` |
| `LARS` | `algorithms/lars.hpp` |
| `Lion` | `algorithms/lion.hpp` |

`Lookahead` (`algorithms/lookahead_wrapper.hpp`) iç optimizer'ı saran ayrı bir türdür. `adamw.hpp`, `momentum_sgd.hpp` ve `nesterov_sgd.hpp` yalnızca ilgili başlığı dahil eder; kendi sınıfları yoktur.

Üst depodaki `include/nexus/optim/optimizers.hpp` cephesi `AdamW` ve `SGD` bağlar. `RAdam` ve `NAdam` bu kütüphanenin çekirdeğindedir; Python eklentisine bağlanmamıştır.

## Zamanlayıcılar

`include/nexus_optim/scheduler` altındaki kurulabilir türler:

| Tür | Başlık |
|---|---|
| `StepLR` | `scheduler/step_lr.hpp` |
| `ExponentialLR` | `scheduler/exponential_lr.hpp` |
| `CosineAnnealingLR` | `scheduler/cosine_annealing_lr.hpp` |
| `CosineAnnealingWarmRestarts` | `scheduler/cosine_warm_restarts.hpp` |
| `OneCycleLR` | `scheduler/one_cycle_lr.hpp` |
| `LinearWarmup` | `scheduler/linear_warmup.hpp` |
| `ReduceLROnPlateau` | `scheduler/reduce_lr_on_plateau.hpp` |

`LRSchedulerBase` bu yedisinin ortak tabanıdır. Zamanlayıcı, grup alanındaki `learning_rate` değerini yerinde yazar. `step()` bu alanı okur.

## Kurulum

C++20 derleyicisi ve CMake 3.20 gerekir. CUDA kapalıyken:

```bat
cmake -S . -B build -DNEXUS_OPTIM_WITH_CUDA=OFF
cmake --build build --config Release --parallel
cmake --install build --config Release --prefix install
```

CUDA açıkken çekirdekler `src/cuda/cuda_optimizer_kernels.cu` ile derlenir. `CMAKE_CUDA_ARCHITECTURES` verilmezse CMake varsayılanı `86` olur.

```bat
cmake -S . -B build-cuda -DNEXUS_OPTIM_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --config Release --parallel
```

Başka bir hedefe eklemek:

```cmake
add_subdirectory(path/to/NexusOptim)
target_link_libraries(my_app PRIVATE nexus_optim::nexus_optim)
```

| Seçenek | Varsayılan | Anlamı |
|---|---|---|
| `NEXUS_OPTIM_WITH_CUDA` | `OFF` | CUDA güncelleme çekirdekleri |
| `NEXUS_OPTIM_BUILD_TESTS` | `ON` | `tests/` GoogleTest hedefleri |
| `NEXUS_OPTIM_BUILD_BENCHMARKS` | `ON` | `benchmarks/` programları |
| `NEXUS_OPTIM_BUILD_EXAMPLES` | `ON` | `examples/` programları |
| `NEXUS_OPTIM_ENABLE_AVX512` | `ON` | AVX-512 çeviri birimini derlemek; çalışma anında seçim CPUID ile yapılır |
| `NEXUS_OPTIM_WARNINGS_AS_ERRORS` | `ON` | Uyarılar hata sayılır |
| `NEXUS_MATRIXFLASH_INCLUDE` | boş | Doluysa MatrixFlash-Pro başlık yolu |

## `step()` örneği

`SGD::Options::lr` saklanır. Güncellemenin öğrenme hızı `ParamGroupOptions::learning_rate` alanından okunur. Ağırlık sönümü, grupta `weight_decay_override` yoksa `Options::weight_decay` değeridir.

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

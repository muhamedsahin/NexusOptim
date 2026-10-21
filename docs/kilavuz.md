# NexusOptim kılavuzu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Bu dosya kütüphanenin durum sözlüğünü, SIMD seçimini, isteğe bağlı CUDA yolunu ve Python eklentisindeki iş parçacığı davranışını anlatır. Kurulum ve tür listesi `README.md` içindedir.

Pandoc kurulu değildir. `docs/NexusOptim-Kilavuz.pdf` ve `docs/NexusOptim-Detayli-Kilavuz.pdf` yeniden üretilmedi. `docs/nexusoptim-manual.html` de bu geçişte değiştirilmedi. Güncel anlatım bu markdown dosyasıdır. Ayrı bir dokümantasyon sitesi yoktur.

## `state_dict`

`OptimizerBase::state_dict()` türetilmiş sınıfın `state_dict_impl` işlevini çağırır. Dönüş tipi `OptimizerState` (`include/nexus_optim/core/optimizer_state.hpp`):

| Alan | İçerik |
|---|---|
| `algorithm` | Algoritma adı |
| `step` | Adım sayacı |
| `group_learning_rates` | Grup öğrenme hızları |
| `blobs` | Moment tamponlarının küçük endian ham kopyaları |
| `scalars` | Bias çarpımları ve benzeri skaler yan durum |

Kütüphane bu yapıyı diske yazmaz. Dosya biçimi çağıran uygulamaya aittir. `load_state_dict` aynı yerleşimdeki tamponları geri kopyalar. Blob boyutu parametre yerleşimiyle uyuşmazsa `std::runtime_error` fırlatılır.

Başlıklardaki `algorithm` dizeleri:

| Tür | `algorithm` |
|---|---|
| `SGD`, `MomentumSGD`, `NesterovSGD` | `"sgd"` |
| `Adam` | `"adam"` |
| `AdamW` | `"adamw"` |
| `Adagrad` | `"adagrad"` |
| `RMSProp` | `"rmsprop"` |
| `Adadelta` | `"adadelta"` |
| `NAdam` | `"nadam"` |
| `RAdam` | `"radam"` |
| `AdaBelief` | `"adabelief"` |
| `LAMB` | `"lamb"` |
| `LARS` | `"lars"` |
| `Lion` | `"lion"` |

`Adam` / `AdamW` yüklemesi, boş olmayan `algorithm` alanının ayrıştırılmış (decoupled) bayrakla uyumunu denetler. `Lookahead` kendi `state_dict` işlevini tanımlamaz; iç optimizer'ın durumu `base().state_dict()` ile alınır. Yavaş ağırlık kopyası `OptimizerState` içine yazılmaz.

```cpp
nexus_optim::OptimizerState checkpoint = optimizer.state_dict();
optimizer.load_state_dict(checkpoint);
```

## SIMD seçimi

`src/simd/simd_dispatch.cpp` süreç başında bir kez CPUID okur. x86 sırasında sıra şöyledir:

1. AVX-512F, FMA ve işletim sisteminin ZMM durumunu açması (`XCR0`) birlikte varsa ve kütüphane `NEXUS_OPTIM_ENABLE_AVX512` ile derlendiyse yol `SimdLevel::Avx512` olur.
2. AVX2, FMA ve işletim sisteminin AVX durumunu açması birlikte varsa yol `SimdLevel::Avx2` olur.
3. Aksi hâlde yol `SimdLevel::Scalar` olur.

x86 dışında derlenen kod, NEON çeviri birimi varsa `SimdLevel::Neon`, yoksa skaler yolu seçer. `force_simd_level` bu seçimi ölçüm için geçersiz kılabilir.

`NEXUS_OPTIM_ENABLE_AVX512=ON` yalnızca AVX-512 nesne kodunu derler. Ryzen 7 5800H üzerinde AVX-512 seçilmez. Bu işlemcide çalışan yol, AVX2 koşulları sağlandığı için AVX2'dir.

Çekirdekler `simd_dispatch.hpp` üzerinden `sgd_update`, `adam_update` ve diğer algoritma güncellemelerine dağılır. Seçim döngünün içinde tekrarlanmaz.

## CUDA isteğe bağlıdır

`NEXUS_OPTIM_WITH_CUDA` varsayılanı `OFF`. Kapalı derlemede `Device::CUDA` grubu üzerinde `step()` çalışma zamanı hatası fırlatır. Açık derlemede `include/nexus_optim/cuda/cuda_launch.hpp` şu FP32 girişlerini sunar: `sgd_update`, `adam_update`, `adagrad_update`, `rmsprop_update`, `adadelta_update`, `nadam_update`, `radam_update`, `adabelief_update`, `lamb_update`, `lars_update`, `lion_update`.

İşaretçiler aygıt belleğini göstermelidir. Kütüphane eğitim döngüsü adına host ile aygıt arasında kopya taşımaz. `copy_host_to_device` ve `copy_device_to_host` ayrı işlevlerdir. CUDA yolu FP32 depolama bekler; dar tiplerde Adam ve SGD CPU tarafında FP32 ana kopya kullanır.

## Python: çağrı başına katıl

Kalıcı iş parçacığı havuzu Python eklentisinin içinde kullanılmaz. Havuz, DLL bellekten kalkarken erişim ihlaliyle çöküyordu. Üst depodaki `python/bindings/module.cpp`, modül açılışında `ThreadPool::use_caller_thread()` çağırır. Windows'ta `python38.dll` … `python314.dll` yüklüyse `hosted_by_python()` aynı kapıyı ayrıca açar.

Bu kapı açıkken:

- Havuz yardımcı işçisi başlatmaz (`helpers_ = 0`).
- `ThreadPool::instance()` nesneyi `new` ile ayırır ve süreç bitene kadar yok etmez. Böylece pyd kapanırken işçi `join` edilmez.
- `parallel_for`, `join_per_call()` dalına girer.

`kParallelGrain` değeri `50000` (`include/nexus_optim/core/types.hpp`). Eleman sayısı bunun altındaysa aralık çağıran iş parçacığında biter. `n >= 50000` ise donanım iş parçacığı sayısı (en çok 8 parça) kadar `std::thread` açılır, çağıran ilk parçayı işler ve dönüşten önce `join` edilir. Donanım iş parçacığı sayısı 2'den küçükse yine tek iş parçacığı kalır.

Üst deponun Python cephesi `nexus.optim.AdamW` ve `nexus.optim.SGD` bağlar. `RAdam` ve `NAdam` çekirdekte durur ve Python'a bağlanmamıştır.

## Bu makinedeki ölçüm

Ryzen 7 5800H üzerinde `Linear(1000, 1000)` Adam adımı ölçüldü. Tek iş parçacığı 1.92 ms, katılan iş parçacıkları 1.81 ms. Ağırlık tamponu 1 000 000 elemandır; bu sayı `kParallelGrain` eşiğinin üstündedir, bu yüzden Python yolu parçalayıp katılabilir. İki süre yalnızca bu adım içindir.

C++ çalıştırılabilir dosyaları kalıcı havuzu kullanmaya devam eder. Havuz, kurucuda `hardware_concurrency() - 1` işçi açar (üst sınır 63), işçiler koşul değişkeninde bekler ve `n >= 50000` iken aynı işçiler yeniden kullanılır. Yukarıdaki 1.92 ms ve 1.81 ms çifti bu kalıcı havuzun ölçümü değildir.

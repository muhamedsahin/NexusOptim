#pragma once

/// @file half.hpp
/// Software IEEE FP16 and BF16. Moments stay in FP32; these types are storage only.

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace nexus_optim {

struct alignas(2) float16 {
  std::uint16_t bits = 0;

  float16() = default;
  explicit float16(std::uint16_t raw) : bits(raw) {}

  static float16 from_float(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
    std::uint32_t mant = x & 0x7FFFFFu;
    if ((x & 0x7FFFFFFFu) == 0) {
      return float16{static_cast<std::uint16_t>(sign)};
    }
    if (exp <= 0) {
      if (exp < -10) {
        return float16{static_cast<std::uint16_t>(sign)};
      }
      mant |= 0x800000u;
      const std::uint32_t shift = static_cast<std::uint32_t>(1 - exp);
      std::uint32_t half_mant = mant >> (shift + 13);
      const std::uint32_t remainder = mant & ((1u << (shift + 13)) - 1u);
      const std::uint32_t halfway = 1u << (shift + 12);
      if (remainder > halfway || (remainder == halfway && (half_mant & 1u))) {
        ++half_mant;
      }
      return float16{static_cast<std::uint16_t>(sign | half_mant)};
    }
    if (exp >= 31) {
      if (exp == 128 + 15 && mant != 0) {
        return float16{static_cast<std::uint16_t>(sign | 0x7E00u)};
      }
      return float16{static_cast<std::uint16_t>(sign | 0x7C00u)};
    }
    std::uint32_t half_mant = mant >> 13;
    const std::uint32_t remainder = mant & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mant & 1u))) {
      ++half_mant;
      if (half_mant == 0x400u) {
        half_mant = 0;
        ++exp;
        if (exp >= 31) {
          return float16{static_cast<std::uint16_t>(sign | 0x7C00u)};
        }
      }
    }
    return float16{static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | (half_mant & 0x3FFu))};
  }

  float to_float() const {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exp = (bits >> 10) & 0x1Fu;
    const std::uint32_t mant = bits & 0x03FFu;
    std::uint32_t out = 0;
    if (exp == 0) {
      if (mant == 0) {
        out = sign;
      } else {
        std::uint32_t m = mant;
        std::int32_t e = -1;
        do {
          ++e;
          m <<= 1;
        } while ((m & 0x400u) == 0);
        m &= 0x3FFu;
        const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 15 - e);
        out = sign | (exp32 << 23) | (m << 13);
      }
    } else if (exp == 31) {
      out = sign | 0x7F800000u | (mant << 13);
    } else {
      const std::uint32_t exp32 = exp + (127 - 15);
      out = sign | (exp32 << 23) | (mant << 13);
    }
    float value = 0;
    std::memcpy(&value, &out, sizeof(value));
    return value;
  }
};

struct alignas(2) bfloat16 {
  std::uint16_t bits = 0;

  bfloat16() = default;
  explicit bfloat16(std::uint16_t raw) : bits(raw) {}

  static bfloat16 from_float(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof(x));
    const std::uint32_t lsb = (x >> 16) & 1u;
    const std::uint32_t rounding_bias = 0x7FFFu + lsb;
    x += rounding_bias;
    return bfloat16{static_cast<std::uint16_t>(x >> 16)};
  }

  float to_float() const {
    const std::uint32_t out = static_cast<std::uint32_t>(bits) << 16;
    float value = 0;
    std::memcpy(&value, &out, sizeof(value));
    return value;
  }
};

template <typename T>
struct is_narrow_storage : std::false_type {};

template <>
struct is_narrow_storage<float16> : std::true_type {};

template <>
struct is_narrow_storage<bfloat16> : std::true_type {};

template <typename T>
inline float load_as_float(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, double>) {
    return static_cast<float>(value);
  } else {
    return value.to_float();
  }
}

template <typename T>
inline T store_from_float(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, double>) {
    return static_cast<double>(value);
  } else {
    return T::from_float(value);
  }
}

}  // namespace nexus_optim

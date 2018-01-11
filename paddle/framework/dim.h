#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "paddle/platform/assert.h"
#include "paddle/platform/hostdevice.h"

#ifdef __HIPCC__
#define POSTHOSTDEVICE restrict(amp, cpu)
#define POSTDEVICE restrict(amp)
#define POSTHOST restrict(cpu)
#else
#define POSTHOSTDEVICE
#define POSTDEVICE
#define POSTHOST
#endif


namespace paddle {
namespace framework {

// Statically sized, statically indexed dimension
template <int i>
struct Dim {
  static constexpr int dimensions = i;

  template <typename... Args>
  Dim(int64_t _head, Args... _tail) POSTHOSTDEVICE : head(_head), tail(_tail...) {
    static_assert(sizeof...(_tail) == i - 1,
                  "Dim initialized with the wrong number of parameters");
  }

  Dim(int64_t _head, const Dim<i - 1>& _tail) POSTHOSTDEVICE : head(_head), tail(_tail) {}

  Dim() POSTHOSTDEVICE : head(0), tail() {}

  /** Construct a Dim from a linear index and size.  Uses Fortran order
   * indexing. */
  Dim(int64_t idx, const Dim<i>& size) POSTHOSTDEVICE
      : head(idx % size.head), tail(idx / size.head, size.tail) {}

  /** Construct a Dim with each dimension set to the given index */
  Dim(int64_t idx) POSTHOSTDEVICE : head(idx), tail(idx) {}

  bool operator==(const Dim<i>& o) const POSTHOSTDEVICE {
    return (head == o.head) && (tail == o.tail);
  }

  bool operator!=(const Dim<i>& o) const POSTHOSTDEVICE { return !(*this == o); }

  int64_t& operator[](int idx) POSTHOSTDEVICE;
  int64_t operator[](int idx) const POSTHOSTDEVICE;

  std::string to_string() const POSTHOST;

  int64_t head;
  Dim<i - 1> tail;
};

// Base case specialization
template <>
struct Dim<1> {
  static constexpr int dimensions = 1;

  Dim(int64_t _head) POSTHOSTDEVICE : head(_head) {}

  Dim() POSTHOSTDEVICE : head(0) {}

  Dim(int idx, const Dim<1>& size) POSTHOSTDEVICE : head(idx) {
#ifndef __HIP_DEVICE_COMPILE__
    if (idx >= size.head) {
      ;//throw std::invalid_argument("Index out of range.");
    }
#else
    PADDLE_ASSERT(idx < size.head);
#endif
  }

  bool operator==(const Dim<1>& o) const POSTHOSTDEVICE { return (head == o.head); }

  bool operator!=(const Dim<1>& o) const POSTHOSTDEVICE { return !(*this == o); }

  int64_t& operator[](int idx) POSTHOSTDEVICE;
  int64_t operator[](int idx) const POSTHOSTDEVICE;

  int64_t head;
};

namespace {

// Helper for accessing Dim classes
template <int i>
struct DimGetter {
  // Return a copy if Dim is const
  template <typename D>
  static int64_t impl(const D& d) POSTHOSTDEVICE {
    return DimGetter<i - 1>::impl(d.tail);
  }
  // Return a reference if Dim is mutable
  template <typename D>
  static int64_t& impl(D& d) POSTHOSTDEVICE {
    return DimGetter<i - 1>::impl(d.tail);
  }
};

// Eureka! We found the element!
template <>
struct DimGetter<0> {
  // Return a copy if Dim is const
  template <typename D>
  static int64_t impl(const D& d) POSTHOSTDEVICE {
    return d.head;
  }
  // Return a reference if Dim is mutable
  template <typename D>
  static int64_t& impl(D& d) POSTHOSTDEVICE {
    return d.head;
  }
};

template <int D>
int64_t& indexer(Dim<D>& dim, int idx) POSTHOSTDEVICE {
#ifndef __HIP_DEVICE_COMPILE__
  if (idx < 0) {
    ;//throw std::invalid_argument("Tried to access a negative dimension");
  }
#else
  PADDLE_ASSERT(idx >= 0);
#endif
  if (idx == 0) {
    return dim.head;
  }
  return indexer(dim.tail, idx - 1);
}

template <>
int64_t& indexer<1>(Dim<1>& dim, int idx) POSTHOSTDEVICE {
#ifndef __HIP_DEVICE_COMPILE__
  if (idx != 0) {
    ;//throw std::invalid_argument("Invalid index");
  }
#else
  PADDLE_ASSERT(idx == 0);
#endif
  return dim.head;
}

template <int D>
int64_t indexer(const Dim<D>& dim, int idx) POSTHOSTDEVICE {
#ifndef __HIP_DEVICE_COMPILE__
  if (idx < 0) {
    ;//throw std::invalid_argument("Tried to access a negative dimension");
  }
#else
  PADDLE_ASSERT(idx >= 0);
#endif
  if (idx == 0) {
    return dim.head;
  }
  return indexer(dim.tail, idx - 1);
}

template <>
int64_t indexer<1>(const Dim<1>& dim, int idx) POSTHOSTDEVICE {
#ifndef __HIP_DEVICE_COMPILE__
  if (idx != 0) {
    ;//throw std::invalid_argument("Invalid index");
  }
#else
  PADDLE_ASSERT(idx == 0);
#endif
  return dim.head;
}

}  // namespace
// Static access to constant Dim
template <int i, int l>
int64_t get(const Dim<l>& d) POSTHOSTDEVICE {
  return DimGetter<i>::impl(d);
}

// Static access to mutable Dim
template <int i, int l>
int64_t& get(Dim<l>& d) POSTHOSTDEVICE {
  return DimGetter<i>::impl(d);
}

// Dynamic access to constant Dim
template <int l>
int64_t Dim<l>::operator[](int i) const POSTHOSTDEVICE {
  return indexer(*this, i);
}

// Dynamic access to mutable Dim
template <int l>
int64_t& Dim<l>::operator[](int i) POSTHOSTDEVICE {
  return indexer(*this, i);
}

// Dynamic access to constant Dim
inline int64_t Dim<1>::operator[](int i) const POSTHOSTDEVICE {
  return indexer(*this, i);
}

// Dynamic access to mutable Dim
inline int64_t& Dim<1>::operator[](int i) POSTHOSTDEVICE {
  return indexer(*this, i);
}

// Dynamic access to constant Dim
// without std::enable_if will try to instantiate this on get<0>(d)
template <int l>
typename std::enable_if<(l > 0), int64_t>::type get(const Dim<l>& d,
                                                               int i) POSTHOSTDEVICE {
  return d[i];
}

// Dynamic access to mutable Dim
template <int l>
typename std::enable_if<(l > 0), int64_t&>::type get(Dim<l>& d,
                                                                int i) POSTHOSTDEVICE {
  return d[i];
}

// Dot product of two dims
template <int i>
int64_t linearize(const Dim<i>& a, const Dim<i>& b) POSTHOSTDEVICE {
  return a.head * b.head + linearize(a.tail, b.tail);
}

// Base case dot product of two Dims
// Notice it is inline because it is no longer a template
template <>
inline int64_t linearize(const Dim<1>& a, const Dim<1>& b) POSTHOSTDEVICE {
  return a.head * b.head;
}

// Product of a Dim
template <int i>
int64_t product(const Dim<i>& a, int prod = 1) POSTHOSTDEVICE {
  return prod * a.head * product(a.tail);
}

// Base case product of a Dim
// Notice it is inline because it is no longer a template
template <>
inline int64_t product(const Dim<1>& a, int prod) POSTHOSTDEVICE {
  return prod * a.head;
}

// Is 0 <= idx_i < size_i for all i?
template <int i>
bool contained(const Dim<i>& idx, const Dim<i>& size) POSTHOSTDEVICE {
  return ((0 <= idx.head) && (idx.head < size.head) &&
          contained(idx.tail, size.tail));
}

// Base case of is 0 <= idx_i < size_i ?
// Notice it is inline because it is no longer a template
template <>
inline bool contained(const Dim<1>& idx, const Dim<1>& size) POSTHOSTDEVICE {
  return ((0 <= idx.head) && (idx.head < size.head));
}

/**
 * \brief Compute exclusive prefix-multiply of a Dim.
 */
template <int i>
Dim<i> ex_prefix_mul(const Dim<i>& src, int mul = 1) POSTHOSTDEVICE {
  return Dim<i>(mul, ex_prefix_mul(src.tail, mul * src.head));
}

///\cond HIDDEN
// Base case of ex_prefix_mul
// Notice it is inline because it is no longer a template
template <>
inline Dim<1> ex_prefix_mul(const Dim<1>& src, int mul) POSTHOSTDEVICE {
  return Dim<1>(mul);
}
///\endcond

/**
 * Add two dimensions together
 */
template <int i>
Dim<i> dim_plus(const Dim<i>& a, const Dim<i>& b) POSTHOSTDEVICE {
  return Dim<i>(a.head + b.head, dim_plus(a.tail, b.tail));
}

// Base case
template <>
inline Dim<1> dim_plus(const Dim<1>& a, const Dim<1>& b) POSTHOSTDEVICE {
  return Dim<1>(a.head + b.head);
}

template <int i>
Dim<i> operator+(const Dim<i>& lhs, const Dim<i>& rhs) POSTHOSTDEVICE {
  return dim_plus(lhs, rhs);
}

/**
 * Multiply two dimensions together
 */
template <int i>
Dim<i> dim_mult(const Dim<i>& a, const Dim<i>& b) POSTHOSTDEVICE {
  return Dim<i>(a.head * b.head, dim_mult(a.tail, b.tail));
}

// Base case
template <>
inline Dim<1> dim_mult(const Dim<1>& a, const Dim<1>& b) POSTHOSTDEVICE {
  return Dim<1>(a.head * b.head);
}

template <int i>
Dim<i> operator*(const Dim<i>& lhs, const Dim<i>& rhs) {
  return dim_mult(lhs, rhs);
}

/**
 * \brief Normalize strides to ensure any dimension with extent 1
 * has stride 0.
 *
 * \param size Dim object containing the size of an array
 * \param stride Dim object containing stride of an array
 * \return Dim object the same size as \p size with normalized strides
 *
 */

template <int i>
Dim<i> normalize_strides(const Dim<i>& size, const Dim<i>& stride) POSTHOSTDEVICE {
  int norm_stride = size.head == 1 ? 0 : stride.head;
  return Dim<i>(norm_stride, normalize_strides(size.tail, stride.tail));
}

///\cond HIDDEN

template <>
inline Dim<1> normalize_strides(const Dim<1>& size,
                                           const Dim<1>& stride) POSTHOSTDEVICE {
  int norm_stride = size.head == 1 ? 0 : stride.head;
  return Dim<1>(norm_stride);
}

///\endcond

/**
 * Helper function to create a Dim
 *
 * \param idxes The type of Dim constructed depends on the number of params
 *
 */

template <typename... Args>
Dim<sizeof...(Args)> make_dim(Args... idxes) POSTHOSTDEVICE {
  return Dim<sizeof...(Args)>(idxes...);
}

// Allows us to output a Dim
// XXX For some reason, overloading fails to resolve this correctly
template <int i>
typename std::enable_if<(i > 1), std::ostream&>::type operator<<(
    std::ostream& os, const Dim<i>& d) {
  os << d.head << ", " << d.tail;
  return os;
}

// Base case that allows us to output a Dim
// XXX I wish this could be an overload instead of a template
template <int i>
typename std::enable_if<(i == 1), std::ostream&>::type operator<<(
    std::ostream& os, const Dim<i>& d) {
  os << d.head;
  return os;
}

template <int i>
std::string Dim<i>::to_string() const POSTHOST {
  std::stringstream stream;

  stream << *this;

  return stream.str();
}

template <int D>
Dim<D> linear_to_dimension(int linear_index, Dim<D> extents) POSTHOSTDEVICE {
  Dim<D> result;

  for (int i = 0; i < D - 1; ++i) {
    result[i] = linear_index % extents[i];
    linear_index /= extents[i];
  }

  result[D - 1] = linear_index;

  return result;
}

}  // namespace framework
}  // namespace paddle

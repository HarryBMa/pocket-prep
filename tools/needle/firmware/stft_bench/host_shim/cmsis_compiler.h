/* Host-only stand-in for CMSIS-Core's cmsis_compiler.h, so CMSIS-DSP builds with a PC compiler
   for correctness tests. Not used on the RA8P1 (FSP provides the real one). */
#ifndef NEEDLE_HOST_CMSIS_COMPILER_H
#define NEEDLE_HOST_CMSIS_COMPILER_H
#include <stdint.h>
#define __ASM __asm__
#define __INLINE inline
#define __STATIC_INLINE static inline
#define __STATIC_FORCEINLINE static inline __attribute__((always_inline))
#define __NO_RETURN __attribute__((__noreturn__))
#define __USED __attribute__((used))
#define __WEAK __attribute__((weak))
#define __PACKED __attribute__((packed, aligned(1)))
#define __PACKED_STRUCT struct __attribute__((packed, aligned(1)))
#define __ALIGNED(x) __attribute__((aligned(x)))
#define __RESTRICT __restrict
#define __UNALIGNED_UINT32_READ(addr) (*((const uint32_t *)(addr)))
#endif

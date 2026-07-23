#ifndef SMX_CCE_SHIM_H
#define SMX_CCE_SHIM_H

#include <cstdint>
#include "smx_tiling_data.h"

#ifndef __CPU_SIM
#define SMX_AICORE [aicore]
#define AICORE [aicore]
#define SMX_INTERNAL SMX_AICORE inline __attribute__((always_inline))
#else
#define SMX_AICORE
#define AICORE
#define SMX_INTERNAL inline
#endif

namespace smx_cce {

using MaskReg = vector_bool;

constexpr uint16_t BLOCK_BYTE_32 = 32;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t CONST_2 = 2;
constexpr int64_t CONST_4 = 4;
constexpr int64_t CONST_32 = 32;
constexpr int64_t CONST_64 = 64;
constexpr uint32_t UB_SIZE = 262144;
constexpr int64_t RESERVED_UB_SIZE = 32;
constexpr int64_t RESERVED_UB_FOR_ALIGN = 128;
constexpr int64_t VECTOR_CORE_NUM = 64;
constexpr int64_t BYTES_OF_FP16 = 2;
constexpr int64_t BYTES_OF_BF16 = 2;
constexpr int64_t BYTES_OF_FP8 = 1;
constexpr int64_t BYTES_OF_INT16 = 2;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t X_ONCE_NUM = 512;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t BASE_LAST_FACTOR_DIM1 = 256;

constexpr int OUT_E4M3 = 0;
constexpr int OUT_E5M2 = 1;
constexpr int OUT_E2M1 = 2;
constexpr int OUT_E1M2 = 3;

constexpr int TPL_RINT = 1;
constexpr int TPL_ROUND = 0;
constexpr int TPL_FLOOR = 4;

constexpr int TPL_SCALE_ALG_0 = 0;
constexpr int TPL_SCALE_ALG_1 = 1;

constexpr int FLOAT_OVERFLOW_MODE_CTRL = 60;

constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t BF16_EXP_MASK = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t FP16_EXP_MASK = 0x7c00;
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint16_t FP4_E2M1_MAX_EXP = 0x0100;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t INVALID_FP16 = 0x7c00;
constexpr uint16_t ABS_MASK_16 = 0x7fff;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0f;
constexpr float ONE_FOURTH = 0.25f;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t ZERO_FOR_ALL = 0x00000000;
constexpr uint32_t EXP_254 = 0x000000fe;
constexpr uint32_t HALF_FOR_MAN = 0x00400000;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925;
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t THRESHOLD_E2M1 = 0x0100;
constexpr uint16_t THRESHOLD_E1M2 = 0x0080;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t ONE_BIT_FOR_MANTISSA = 0x00c1;
constexpr uint16_t TWO_BIT_FOR_MANTISSA = 0x00e1;

constexpr uint16_t VL_B16 = 128;
constexpr uint16_t VL_B32 = 64;
constexpr uint16_t VL_B8 = 256;

template <typename T>
AICORE inline constexpr int64_t CeilDiv(int64_t a, T b) { return (a + (int64_t)b - 1) / (int64_t)b; }
AICORE inline constexpr int64_t FloorDiv(int64_t a, int64_t b) { return a / b; }
AICORE inline constexpr uint32_t AlignUp32(uint32_t v) { return (v + 31U) / 32U * 32U; }

template <int OK>
AICORE constexpr bool IsFp4Out() { return OK == OUT_E2M1 || OK == OUT_E1M2; }

template <int OK>
AICORE constexpr uint16_t YMaxExpForOutKind() {
    if constexpr (OK == OUT_E4M3) return FP8_E4M3_MAX_EXP;
    else if constexpr (OK == OUT_E5M2) return FP8_E5M2_MAX_EXP;
    else if constexpr (OK == OUT_E2M1) return FP4_E2M1_MAX_EXP;
    else return 0;
}

template <int OK>
AICORE constexpr uint32_t DtypeMaxForOutKind() {
    if constexpr (OK == OUT_E4M3) return FP8_E4M3_MAX;
    else if constexpr (OK == OUT_E5M2) return FP8_E5M2_MAX;
    else return 0;
}

template <int OK>
AICORE constexpr uint16_t F4EmaxForOutKind() {
    if constexpr (OK == OUT_E2M1) return FP4_E2M1_MAX_EXP;
    else if constexpr (OK == OUT_E1M2) return FP4_E1M2_MAX_EXP;
    else return 0;
}

#ifndef __CPU_SIM
#define SMX_INTERNAL AICORE inline __attribute__((always_inline))
#else
#define SMX_INTERNAL inline
#endif

#ifdef SMX_SIMD_CALLEE
#define SMX_VEC_SCOPE
#define SMX_INTERNAL_SIMD __attribute__((always_inline)) __simd_callee__ inline
#define SMX_INTERNAL_VF __attribute__((always_inline)) __simd_vf__ inline
#else
#define SMX_VEC_SCOPE __VEC_SCOPE__
#define SMX_INTERNAL_SIMD SMX_INTERNAL
#define SMX_INTERNAL_VF SMX_INTERNAL
#endif

constexpr uint8_t BUF_ID_IN0 = 0;
constexpr uint8_t BUF_ID_IN1 = 1;
constexpr uint8_t BUF_ID_OUT0 = 2;
constexpr uint8_t BUF_ID_OUT1 = 3;
constexpr uint8_t BUF_ID_SCALE0 = 4;
constexpr uint8_t BUF_ID_SCALE1 = 5;

SMX_INTERNAL void DmaGm2Ub2D(
    __ubuf__ uint16_t *dst, __gm__ uint16_t *src,
    uint16_t rowBlockSize, uint32_t burstLenBytes,
    uint32_t srcStride310, uint32_t dstStride310,
    uint8_t bufId)
{
    get_buf(PIPE_MTE2, bufId, true);
    copy_gm_to_ubuf_align_v2(dst, src, 0, rowBlockSize, burstLenBytes, 0, 0, true, 0, srcStride310, dstStride310);
    rls_buf(PIPE_MTE2, bufId, true);
}

SMX_INTERNAL void DmaGm2Ub1D(
    __ubuf__ uint16_t *dst, __gm__ uint16_t *src,
    uint32_t totalBytes, uint8_t bufId)
{
    get_buf(PIPE_MTE2, bufId, true);
    copy_gm_to_ubuf_align_v2(dst, src, 0, 1, totalBytes, 0, 0, false, 0, 0, 0);
    rls_buf(PIPE_MTE2, bufId, true);
}

SMX_INTERNAL void DmaUb2GmY(
    __gm__ uint8_t *dst, __ubuf__ uint8_t *src,
    uint32_t nRows, uint32_t rowBytes,
    uint32_t dstRowStride, uint32_t srcRowStride,
    uint8_t bufId)
{
    get_buf(PIPE_MTE3, bufId, false);
    // SMX_FIXB: same dropped-burst issue as DmaUb2GmScale -- a single
    // copy_ubuf_to_gm_align_v2 with nBurst>1 only lands the first burst on this
    // target, so batched (>1 row/core) stores wrote only row 0 of each core and
    // rows 1+ read back as zero (y_match collapsed for >1 row/core). Issue one
    // nBurst=1 burst per row instead: rowBytes valid bytes from the contiguous UB
    // row (src advances by srcRowStride = aligned UB row bytes), landing at the GM
    // row (dst advances by dstRowStride = the full GM row pitch).
    for (uint32_t r = 0; r < nRows; ++r) {
        copy_ubuf_to_gm_align_v2(dst + (uint64_t)r * dstRowStride,
                                 src + (uint64_t)r * srcRowStride,
                                 0, 1, rowBytes, 0, 0, 0);
    }
    rls_buf(PIPE_MTE3, bufId, false);
}

SMX_INTERNAL void DmaUb2GmScale(
    __gm__ uint8_t *dst, __ubuf__ uint8_t *src,
    uint32_t nRows, uint32_t rowBytes,
    uint32_t dstRowStride, uint32_t srcRowStride,
    uint8_t bufId)
{
    get_buf(PIPE_MTE3, bufId, false);
    // SMX_FIXB: the e8m0 scale bytes are stored PK_B16 CONTIGUOUSLY in UB
    // (row r's rowBytes scale bytes at src[r*srcRowStride], packed back-to-back).
    // A single copy_ubuf_to_gm_align_v2 with nBurst>1 does not emit the trailing
    // bursts on this target (only the first burst lands, so every row past row 0 read
    // as zero -> scale_eq collapsed for >1 row/core). Issue one nBurst=1 burst per row
    // instead, advancing the UB source by srcRowStride and the GM dest by dstRowStride
    // (= outputScaleRowBytes, which also absorbs any GM row padding).
    for (uint32_t r = 0; r < nRows; ++r) {
        copy_ubuf_to_gm_align_v2(dst + (uint64_t)r * dstRowStride,
                                 src + (uint64_t)r * srcRowStride,
                                 0, 1, rowBytes, 0, 0, 0);
    }
    rls_buf(PIPE_MTE3, bufId, false);
}

SMX_INTERNAL void ZeroUb16(__ubuf__ uint16_t *ptr, int32_t count)
{
    int32_t remaining = count;
    while (remaining > 0) {
        int32_t toStore = remaining > 256 ? 256 : remaining;
        uint32_t cnt = static_cast<uint32_t>(toStore);
        __VEC_SCOPE__ {
            vector_u16 zeroReg;
            vbr(zeroReg, (uint16_t)0);
            MaskReg preg = plt_b16(cnt, POST_UPDATE);
            vsts(zeroReg, ptr, 0, NORM_B16, preg);
        }
        remaining -= toStore;
        ptr += toStore;
    }
}

SMX_INTERNAL void ZeroUb8(__ubuf__ uint8_t *ptr, int32_t count)
{
    int32_t remaining = count;
    while (remaining > 0) {
        int32_t toStore = remaining > 256 ? 256 : remaining;
        uint32_t cnt = static_cast<uint32_t>(toStore);
        __VEC_SCOPE__ {
            vector_u8 zeroReg;
            vbr(zeroReg, (uint8_t)0);
            MaskReg preg = plt_b8(cnt, POST_UPDATE);
            vsts(zeroReg, ptr, 0, NORM_B8, preg);
        }
        remaining -= toStore;
        ptr += toStore;
    }
}

#define SMX_TRUNCATE_FP32_FOR_FP4(reg, preg, RM) \
    do { \
        vector_bf16 tmpBf16; \
        vector_f32 tmpF32; \
        if constexpr (RM == TPL_RINT) { \
            vcvt(tmpBf16, (reg), preg, ROUND_R, RS_ENABLE, PART_EVEN); \
        } else if constexpr (RM == TPL_ROUND) { \
            vcvt(tmpBf16, (reg), preg, ROUND_A, RS_ENABLE, PART_EVEN); \
        } else { \
            vcvt(tmpBf16, (reg), preg, ROUND_F, RS_ENABLE, PART_EVEN); \
        } \
        vcvt(tmpF32, tmpBf16, preg, PART_EVEN, MODE_ZEROING); \
        (reg) = tmpF32; \
    } while (0)

#define SMX_FP4_FROM_HALF(reg, pregAll32, OK, RM) \
    do { \
        MaskReg zeroMask; \
        MaskReg specialMask; \
        MaskReg negInfMask; \
        vector_s32 negZero; \
        vector_s32 maxExpFP32, exp0FP32, exp1FP32; \
        vbr(negZero, NEG_ZERO); \
        vcmps_eq(negInfMask, (vector_s32 &)(reg), NEG_ZERO, pregAll32); \
        if constexpr (OK == OUT_E1M2) { \
            vmuls((reg), (reg), FOUR, pregAll32, MODE_ZEROING); \
            vcmps_lt(specialMask, (reg), (float)0, pregAll32); \
            SMX_TRUNCATE_FP32_FOR_FP4(reg, pregAll32, RM); \
            vmuls((reg), (reg), ONE_FOURTH, pregAll32, MODE_ZEROING); \
        } else { \
            vbr(maxExpFP32, (int32_t)MAX_EXP_FOR_FP32); \
            vand(exp0FP32, (vector_s32 &)(reg), maxExpFP32, pregAll32, MODE_ZEROING); \
            vshrs(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32, MODE_ZEROING); \
            vadds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32, MODE_ZEROING); \
            vmaxs(exp0FP32, exp0FP32, 0, pregAll32, MODE_ZEROING); \
            vadds(exp0FP32, exp0FP32, NEG_ONE, pregAll32, MODE_ZEROING); \
            vmuls(exp1FP32, exp0FP32, NEG_ONE, pregAll32, MODE_ZEROING); \
            vadds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32, MODE_ZEROING); \
            vshls(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32, MODE_ZEROING); \
            vmul((reg), (reg), (vector_f32 &)exp1FP32, pregAll32, MODE_ZEROING); \
            vadds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32, MODE_ZEROING); \
            vshls(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32, MODE_ZEROING); \
            vcmps_lt(specialMask, (reg), (float)0, pregAll32); \
            SMX_TRUNCATE_FP32_FOR_FP4(reg, pregAll32, RM); \
            vmul((reg), (reg), (vector_f32 &)exp0FP32, pregAll32, MODE_ZEROING); \
        } \
        vcmps_eq(zeroMask, (reg), (float)0, pregAll32); \
        pand(zeroMask, specialMask, zeroMask, pregAll32); \
        por(zeroMask, negInfMask, zeroMask, pregAll32); \
        vsel((vector_s32 &)(reg), negZero, (vector_s32 &)(reg), zeroMask); \
    } while (0)

#define SMX_VCVT_BF16_TO_FP4(dst, src, preg, OK, RM) \
    do { \
        if constexpr (OK == OUT_E2M1) { \
            if constexpr (RM == TPL_RINT) vcvt((dst), (src), preg, ROUND_R, PART_P0, MODE_ZEROING); \
            else if constexpr (RM == TPL_ROUND) vcvt((dst), (src), preg, ROUND_A, PART_P0, MODE_ZEROING); \
            else vcvt((dst), (src), preg, ROUND_F, PART_P0, MODE_ZEROING); \
        } else { \
            if constexpr (RM == TPL_RINT) vcvt((dst), (src), preg, ROUND_R, PART_P0, MODE_ZEROING); \
            else if constexpr (RM == TPL_ROUND) vcvt((dst), (src), preg, ROUND_A, PART_P0, MODE_ZEROING); \
            else vcvt((dst), (src), preg, ROUND_F, PART_P0, MODE_ZEROING); \
        } \
    } while (0)

#define SMX_VCVT_FP32_TO_BF16(dst, src, preg, RM) \
    do { \
        if constexpr (RM == TPL_RINT) vcvt((dst), (src), preg, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING); \
        else if constexpr (RM == TPL_ROUND) vcvt((dst), (src), preg, ROUND_A, RS_DISABLE, PART_EVEN, MODE_ZEROING); \
        else vcvt((dst), (src), preg, ROUND_F, RS_DISABLE, PART_EVEN, MODE_ZEROING); \
    } while (0)

namespace simd_inlined {

template <typename BDst, typename SrcPtr>
__simd_callee__ inline void vlds_x2(BDst &d0, BDst &d1, SrcPtr &src, int32_t stride)
{
    vlds((vector_bf16 &)d0, (vector_bf16 &)d1, src, stride, DINTLV_B16, POST_UPDATE);
}

template <typename U16Dst>
__simd_callee__ inline void vlds_e2b_b16(U16Dst &dst, __ubuf__ uint16_t *src, int32_t off)
{
    vlds((vector_u16 &)dst, src, off, E2B_B16);
}

template <typename BDst, typename BSrc, typename USrc>
__simd_callee__ inline void vmul(BDst &dst, BSrc src, USrc scale, MaskReg mask)
{
    vmul((vector_bf16 &)dst, (vector_bf16 &)src, (vector_bf16 &)scale, mask, MODE_ZEROING);
}

template <typename Dst, typename Src>
__simd_callee__ inline void vintlv_x2(Dst &d0, Dst &d1, Src s0, Src s1)
{
    vintlv(d0, d1, s0, s1);
}

template <typename F32Dst, typename BF16Src>
__simd_callee__ inline void vcvt_bf16_to_fp32_even(F32Dst &dst, BF16Src src, MaskReg mask)
{
    vcvt((vector_f32 &)dst, (vector_bf16 &)src, mask, PART_EVEN, MODE_ZEROING);
}

template <typename F32Dst, typename BF16Src>
__simd_callee__ inline void vcvt_bf16_to_fp32_odd(F32Dst &dst, BF16Src src, MaskReg mask)
{
    vcvt((vector_f32 &)dst, (vector_bf16 &)src, mask, PART_ODD, MODE_ZEROING);
}

template <typename U8Src>
__simd_callee__ inline void vsts_pk4_b32(U8Src src, __ubuf__ uint8_t *&dst, int32_t stride, MaskReg mask)
{
    vsts((vector_u8 &)src, dst, stride, PK4_B32, mask, POST_UPDATE);
}

// --- SwiGLU F32 scalar wrappers ---

template <typename F32Dst>
__simd_callee__ inline void vmuls_f32(F32Dst &dst, F32Dst src, float scalar, MaskReg mask)
{
    vmuls((vector_f32 &)dst, (vector_f32 &)src, scalar, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vadds_f32(F32Dst &dst, F32Dst src, float scalar, MaskReg mask)
{
    vadds((vector_f32 &)dst, (vector_f32 &)src, scalar, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vexp_f32(F32Dst &dst, F32Dst src, MaskReg mask)
{
    vexp((vector_f32 &)dst, (vector_f32 &)src, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vdiv_f32(F32Dst &dst, F32Dst num, F32Dst den, MaskReg mask)
{
    vdiv((vector_f32 &)dst, (vector_f32 &)num, (vector_f32 &)den, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vmul_f32(F32Dst &dst, F32Dst a, F32Dst b, MaskReg mask)
{
    vmul((vector_f32 &)dst, (vector_f32 &)a, (vector_f32 &)b, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vmins_f32(F32Dst &dst, F32Dst src, float scalar, MaskReg mask)
{
    vmins((vector_f32 &)dst, (vector_f32 &)src, scalar, mask, MODE_ZEROING);
}

template <typename F32Dst>
__simd_callee__ inline void vmaxs_f32(F32Dst &dst, F32Dst src, float scalar, MaskReg mask)
{
    vmaxs((vector_f32 &)dst, (vector_f32 &)src, scalar, mask, MODE_ZEROING);
}

template <typename FDst, typename FSrc>
__simd_callee__ inline void vdintlv_f32(FDst &dstEven, FDst &dstOdd, FSrc src0, FSrc src1)
{
    vdintlv((vector_f32 &)dstEven, (vector_f32 &)dstOdd, (vector_f32 &)src0, (vector_f32 &)src1);
}

template <typename BF16Src>
__simd_callee__ inline void vsts_pk_b32(BF16Src src, __ubuf__ bfloat16_t *ub, int32_t off, MaskReg mask)
{
    vsts((vector_bf16 &)src, ub, off, PK_B32, mask);
}

template <typename Src>
__simd_callee__ inline void vlds_bf16_unpk(Src &dst, __ubuf__ bfloat16_t *ub, int32_t off)
{
    vlds((vector_bf16 &)dst, ub, off, UNPK_B16);
}

// --- U16 wrappers (MaxExp, Scale) ---

template <typename U16Dst>
__simd_callee__ inline void vbr_u16(U16Dst &dst, uint16_t val)
{
    vbr((vector_u16 &)dst, val);
}

template <typename U16Dst, typename U16Src>
__simd_callee__ inline void vand_u16(U16Dst &dst, U16Src src, U16Src mask_val, MaskReg preg)
{
    vand((vector_u16 &)dst, (vector_u16 &)src, (vector_u16 &)mask_val, preg, MODE_ZEROING);
}

template <typename U16Dst, typename U16Src>
__simd_callee__ inline void vmax_u16(U16Dst &dst, U16Src a, U16Src b, MaskReg preg)
{
    vmax((vector_u16 &)dst, (vector_u16 &)a, (vector_u16 &)b, preg, MODE_ZEROING);
}

template <typename U16Dst>
__simd_callee__ inline void vcgmax_u16(U16Dst &dst, U16Dst src, MaskReg preg)
{
    vcgmax((vector_u16 &)dst, (vector_u16 &)src, preg, MODE_ZEROING);
}

template <typename U16Src>
__simd_callee__ inline void vstus_u16(vector_align &dst, uint32_t scaleNum, U16Src src, __ubuf__ uint16_t *ub)
{
    vstus(dst, scaleNum, (vector_u16 &)src, ub, POST_UPDATE);
}

__simd_callee__ inline void vstas_u16(vector_align &dst, __ubuf__ uint16_t *ub)
{
    vstas(dst, ub, 0, POST_UPDATE);
}

template <typename U16Src>
__simd_callee__ inline void vcmp_ne_u16(MaskReg &result, U16Src a, U16Src b, MaskReg preg)
{
    vcmp_ne(result, (vector_u16 &)a, (vector_u16 &)b, preg);
}

template <typename U16Src>
__simd_callee__ inline void vcmp_le_u16(MaskReg &result, U16Src a, U16Src b, MaskReg preg)
{
    vcmp_le(result, (vector_u16 &)a, (vector_u16 &)b, preg);
}

template <typename U16Src>
__simd_callee__ inline void vcmp_eq_u16(MaskReg &result, U16Src a, U16Src b, MaskReg preg)
{
    vcmp_eq(result, (vector_u16 &)a, (vector_u16 &)b, preg);
}

template <typename U16Dst, typename U16Src>
__simd_callee__ inline void vsel_u16(U16Dst &dst, U16Src a, U16Src b, MaskReg preg)
{
    vsel((vector_u16 &)dst, (vector_u16 &)a, (vector_u16 &)b, preg);
}

template <typename U16Dst, typename U16Src>
__simd_callee__ inline void vsub_u16(U16Dst &dst, U16Src a, U16Src b, MaskReg preg)
{
    vsub((vector_u16 &)dst, (vector_u16 &)a, (vector_u16 &)b, preg, MODE_ZEROING);
}

template <typename U16Dst>
__simd_callee__ inline void vshrs_u16(U16Dst &dst, U16Dst src, int16_t shift, MaskReg preg)
{
    vshrs((vector_u16 &)dst, (vector_u16 &)src, shift, preg, MODE_ZEROING);
}

template <typename U16Src>
__simd_callee__ inline void vsts_pk_b16(U16Src src, __ubuf__ uint16_t *ub, int32_t off, MaskReg preg)
{
    vsts((vector_u16 &)src, ub, off, PK_B16, preg);
}

template <typename U16Src>
__simd_callee__ inline void vsts_norm_b16_u16(U16Src src, __ubuf__ uint16_t *ub, int32_t off, MaskReg preg)
{
    vsts((vector_u16 &)src, ub, off, NORM_B16, preg);
}

template <typename U16Dst>
__simd_callee__ inline void vlds_norm_u16(U16Dst &dst, __ubuf__ uint16_t *ub, int32_t off)
{
    vlds((vector_u16 &)dst, ub, off, NORM);
}

template <typename F16Dst, typename F16Src>
__simd_callee__ inline void vcvt_f16_to_bf16(F16Dst &dst, F16Src src, MaskReg preg)
{
    vcvt((vector_bf16 &)dst, (vector_f16 &)src, preg, ROUND_Z);
}

template <typename F16Dst, typename HalfSrcPtr>
__simd_callee__ inline void vlds_x2_f16(F16Dst &d0, F16Dst &d1, HalfSrcPtr &src, int32_t stride)
{
    vlds((vector_f16 &)d0, (vector_f16 &)d1, src, stride, DINTLV_B16, POST_UPDATE);
}


template <int RM, typename FP4Dst, typename BF16Src>
__simd_callee__ inline void vcvt_bf16_to_fp4_e2m1(FP4Dst &dst, BF16Src src, MaskReg mask)
{
    if constexpr (RM == TPL_RINT) vcvt((vector_f4e2m1x2 &)dst, (vector_bf16 &)src, mask, ROUND_R, PART_P0, MODE_ZEROING);
    else if constexpr (RM == TPL_ROUND) vcvt((vector_f4e2m1x2 &)dst, (vector_bf16 &)src, mask, ROUND_A, PART_P0, MODE_ZEROING);
    else vcvt((vector_f4e2m1x2 &)dst, (vector_bf16 &)src, mask, ROUND_F, PART_P0, MODE_ZEROING);
}

template <int RM, typename FP4Dst, typename BF16Src>
__simd_callee__ inline void vcvt_bf16_to_fp4_e1m2(FP4Dst &dst, BF16Src src, MaskReg mask)
{
    if constexpr (RM == TPL_RINT) vcvt((vector_f4e1m2x2 &)dst, (vector_bf16 &)src, mask, ROUND_R, PART_P0, MODE_ZEROING);
    else if constexpr (RM == TPL_ROUND) vcvt((vector_f4e1m2x2 &)dst, (vector_bf16 &)src, mask, ROUND_A, PART_P0, MODE_ZEROING);
    else vcvt((vector_f4e1m2x2 &)dst, (vector_bf16 &)src, mask, ROUND_F, PART_P0, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e4m3_p0(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e4m3 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e4m3_p1(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e4m3 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e4m3_p2(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e4m3 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P2, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e4m3_p3(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e4m3 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P3, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e5m2_p0(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e5m2 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e5m2_p1(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e5m2 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e5m2_p2(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e5m2 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P2, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e5m2_p3(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt((vector_f8e5m2 &)dst, (vector_f32 &)src, mask, ROUND_R, RS_ENABLE, PART_P3, MODE_ZEROING);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e4m3(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt_fp32_to_fp8e4m3_p0(dst, src, mask);
}

template <typename FP8Dst, typename F32Src>
__simd_callee__ inline void vcvt_fp32_to_fp8e5m2(FP8Dst &dst, F32Src src, MaskReg mask)
{
    vcvt_fp32_to_fp8e5m2_p0(dst, src, mask);
}

__simd_callee__ inline void MergeAndStoreFp8NormB8(
    vector_u8 &p0Reg, vector_u8 &p1Reg, vector_u8 &p2Reg, vector_u8 &p3Reg,
    __ubuf__ uint8_t *dst, MaskReg maskB8)
{
    vadd(p0Reg, p0Reg, p2Reg, maskB8, MODE_ZEROING);
    vadd(p0Reg, p0Reg, p1Reg, maskB8, MODE_ZEROING);
    vadd(p0Reg, p0Reg, p3Reg, maskB8, MODE_ZEROING);
    vsts(p0Reg, dst, 0, NORM_B8, maskB8);
}

template <typename TSrc>
__simd_callee__ inline void vlds_norm(TSrc &dst, __ubuf__ bfloat16_t *src, int32_t off)
{
    vlds((vector_bf16 &)dst, src, off, NORM);
}

template <typename TSrc>
__simd_callee__ inline void vlds_norm_half(TSrc &dst, __ubuf__ half *src, int32_t off)
{
    vlds((vector_f16 &)dst, src, off, NORM);
}

template <typename TDst>
__simd_callee__ inline void vsts_norm_b16(TDst src, __ubuf__ bfloat16_t *dst, int32_t off, MaskReg mask)
{
    vsts((vector_bf16 &)src, dst, off, NORM_B16, mask);
}

template <typename TDst>
__simd_callee__ inline void vsts_norm_b8(TDst src, __ubuf__ uint8_t *dst, int32_t off, MaskReg mask)
{
    vsts((vector_u8 &)src, dst, off, NORM_B8, mask);
}

} // namespace simd_inlined

} // namespace smx_cce

#endif

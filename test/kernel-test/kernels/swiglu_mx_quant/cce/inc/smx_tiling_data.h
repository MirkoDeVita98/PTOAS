#ifndef SMX_TILING_DATA_H
#define SMX_TILING_DATA_H

#include <cstdint>

struct SwigluMxQuantTilingData {
    int64_t usedCoreNum;
    int64_t inputDim1;
    int64_t inputDim2;
    int64_t outputDim2;
    int64_t basicDim2;
    int64_t basicDim1;
    int64_t maxBasicNumUbDim2;
    int64_t maxBasicNumUbDim1;
    int64_t ubLoopPerRow;
    int64_t ubTailPerRow;
    int64_t frontCoreNum;
    int64_t frontCoreBasicNumDim1;
    int64_t frontCoreLoopTimes;
    int64_t frontCoreLastLoopBasicNum;
    int64_t tailCoreBasicNumDim1;
    int64_t tailCoreLoopTimes;
    int64_t tailCoreLastLoopBasicNum;
    int64_t activateLeft;
    int64_t swigluMode;
    int64_t roundMode;
    int64_t scaleAlg;
    int64_t groupMode;
    int64_t groupIndexNum;
    int64_t useDoubleBuffer;
    float clampLimit;
    float gluAlpha;
    float gluBias;
    float maxDtypeValue;
};

#endif

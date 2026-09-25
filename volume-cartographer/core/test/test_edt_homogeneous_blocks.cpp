// The space-line loss (lineLossDistance::compute in GrowPatch.cpp) skips the
// distance transform for a block that is entirely foreground or has no
// foreground at all, and writes a constant instead. That is only valid if the
// transform's answer for those two blocks IS that constant after compute()'s
// own copy-out (non-finite -> 255, clamp to [0, 255], round). This test checks
// the premise against the real transform, at the real block size, so an edt
// upgrade that changed either case would fail here instead of silently
// changing growth output.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "edt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace
{
constexpr int kBorder = 16;      // lineLossDistance::BORDER
constexpr int kChunk = 64;       // lineLossDistance::CHUNK_SIZE
constexpr int kBlock = kChunk + 2 * kBorder;

// compute()'s copy-out applied to one transform value.
uint8_t copyOut(float d)
{
    if (!std::isfinite(d)) {
        d = 255.0f;
    }
    d = std::clamp(d, 0.0f, 255.0f);
    return static_cast<uint8_t>(std::lround(d));
}

// Transform of a uniform block; `foreground` selects all-foreground (0 = source)
// or no-foreground (1 = background everywhere), exactly as compute() binarises.
std::vector<uint8_t> transformUniform(bool foreground)
{
    const size_t voxels = static_cast<size_t>(kBlock) * kBlock * kBlock;
    std::vector<uint8_t> binary(voxels, foreground ? 0 : 1);
    std::unique_ptr<float[]> edt(edt::binary_edt<uint8_t>(
        binary.data(), kBlock, kBlock, kBlock, 1.0f, 1.0f, 1.0f, false, 1));
    std::vector<uint8_t> kept;
    for (int z = 0; z < kChunk; ++z)
        for (int y = 0; y < kChunk; ++y)
            for (int x = 0; x < kChunk; ++x) {
                const size_t idx = static_cast<size_t>(z + kBorder) +
                                   static_cast<size_t>(y + kBorder) * kBlock +
                                   static_cast<size_t>(x + kBorder) * kBlock * kBlock;
                kept.push_back(copyOut(edt[idx]));
            }
    return kept;
}
}  // namespace

TEST_CASE("an all-foreground block transforms to exactly 0 everywhere")
{
    const auto kept = transformUniform(/*foreground=*/true);
    REQUIRE(kept.size() == static_cast<size_t>(kChunk) * kChunk * kChunk);
    CHECK(std::all_of(kept.begin(), kept.end(), [](uint8_t v) { return v == 0; }));
}

TEST_CASE("a block with no foreground transforms to exactly 255 everywhere")
{
    const auto kept = transformUniform(/*foreground=*/false);
    REQUIRE(kept.size() == static_cast<size_t>(kChunk) * kChunk * kChunk);
    CHECK(std::all_of(kept.begin(), kept.end(), [](uint8_t v) { return v == 255; }));
}

TEST_CASE("a single foreground voxel is NOT homogeneous: the shortcut must not apply")
{
    // Guard against the shortcut condition widening: one source voxel gives a
    // real gradient, distinct from both constants.
    const size_t voxels = static_cast<size_t>(kBlock) * kBlock * kBlock;
    std::vector<uint8_t> binary(voxels, 1);
    binary[voxels / 2] = 0;
    std::unique_ptr<float[]> edt(edt::binary_edt<uint8_t>(
        binary.data(), kBlock, kBlock, kBlock, 1.0f, 1.0f, 1.0f, false, 1));
    float lo = 1e30f, hi = -1e30f;
    for (size_t i = 0; i < voxels; ++i) {
        lo = std::min(lo, edt[i]);
        hi = std::max(hi, edt[i]);
    }
    CHECK(lo == 0.0f);
    CHECK(hi > 1.0f);
    CHECK(std::isfinite(hi));
}

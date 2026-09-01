#pragma once

#include <filesystem>

// Replace one directory with another, giving the strongest guarantee the
// underlying filesystem actually offers.
//
// Segment directories (tifxyz: x.tif / y.tif / z.tif / meta.json) are saved by
// writing a complete tree to a staging directory and then putting it in place
// of the old one. That final step must never expose a half-replaced segment to
// a concurrent reader, and must never be able to lose the old segment if it
// fails partway. Which mechanism can deliver that depends on the filesystem,
// so callers should not have to know or care what they are sitting on.

// Which mechanism was used. Both leave an identical result on disk; the
// distinction is reported for logging and for tests.
enum class DirReplaceMode {
    AtomicExchange,  // single kernel step; no observable intermediate state
    StagedSwap       // two renames; old tree preserved throughout
};

// Normally Auto. StagedSwapOnly forces the portable path so it can be
// exercised on a filesystem that does support the atomic exchange.
enum class DirReplaceStrategy {
    Auto,
    StagedSwapOnly
};

// Replace `target` with `staged`. Both must live on the same filesystem
// (callers stage into a sibling directory, so they do).
//
// On success `staged` no longer exists and `target` holds what `staged` held.
// On failure this throws and `target` still holds its original contents.
DirReplaceMode replaceDirectory(
    const std::filesystem::path& staged,
    const std::filesystem::path& target,
    DirReplaceStrategy strategy = DirReplaceStrategy::Auto);

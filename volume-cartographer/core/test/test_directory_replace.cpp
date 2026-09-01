#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/DirectoryReplace.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace
{

// A scratch area on whatever filesystem the build tree lives on. Both
// strategies are exercised there, so the portable path is covered even when
// the local filesystem does support the atomic exchange.
struct Scratch {
    fs::path root;
    Scratch()
        : root(fs::temp_directory_path()
               / ("vc_dirreplace_" + std::to_string(
                     std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        fs::remove_all(root);
        fs::create_directories(root);
    }
    ~Scratch() { fs::remove_all(root); }
};

void makeTree(const fs::path& dir, const std::string& marker)
{
    fs::create_directories(dir);
    std::ofstream(dir / marker) << marker;
}

bool hasFile(const fs::path& dir, const std::string& name)
{
    return fs::exists(dir / name);
}

// Nothing may be left lying around next to the target.
bool hasAsideLeftovers(const fs::path& parent)
{
    for (const auto& e : fs::directory_iterator(parent)) {
        if (e.path().filename().string().find(".replaced-") != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST_CASE("replaceDirectory puts the staged tree in place")
{
    for (auto strategy : {DirReplaceStrategy::Auto, DirReplaceStrategy::StagedSwapOnly}) {
        Scratch scratch;
        const fs::path target = scratch.root / "segment";
        const fs::path staged = scratch.root / "segment.staged";
        makeTree(target, "old");
        makeTree(staged, "new");

        replaceDirectory(staged, target, strategy);

        CHECK(hasFile(target, "new"));
        CHECK_FALSE(hasFile(target, "old"));
        CHECK_FALSE(fs::exists(staged));
        CHECK_FALSE(hasAsideLeftovers(scratch.root));
    }
}

TEST_CASE("the portable path is used when the atomic exchange is skipped")
{
    Scratch scratch;
    const fs::path target = scratch.root / "segment";
    const fs::path staged = scratch.root / "segment.staged";
    makeTree(target, "old");
    makeTree(staged, "new");

    CHECK(replaceDirectory(staged, target, DirReplaceStrategy::StagedSwapOnly)
          == DirReplaceMode::StagedSwap);
}

// The regression this file exists for. The previous implementation ran
// remove_all(target) before renaming the staged tree in, so a failure in
// between destroyed the old segment and left nothing at all behind. A segment
// can represent hours of growth, so a failed replacement must be a no-op.
TEST_CASE("a failed replacement leaves the previous tree intact")
{
    for (auto strategy : {DirReplaceStrategy::Auto, DirReplaceStrategy::StagedSwapOnly}) {
        Scratch scratch;
        const fs::path target = scratch.root / "segment";
        const fs::path staged = scratch.root / "segment.staged";
        makeTree(target, "old");
        // staged never exists, so the replacement cannot succeed
        CHECK_THROWS(replaceDirectory(staged, target, strategy));

        CHECK(fs::exists(target));
        CHECK(hasFile(target, "old"));
        CHECK_FALSE(hasAsideLeftovers(scratch.root));
    }
}

#include "vc/core/util/ThreadBudget.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <string>
#include <system_error>

namespace
{

// -1 = not yet resolved, 0 = unlimited, >0 = cap.
std::atomic<int> gThreadBudget{-1};

int budgetFromEnvironment()
{
    const char* configured = std::getenv("VC_MAX_THREADS");
    if (!configured || configured[0] == '\0')
        return 0;

    int parsed = 0;
    const char* end = configured + std::char_traits<char>::length(configured);
    const auto result = std::from_chars(configured, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || parsed < 1)
        return 0;  // Unparseable or nonsensical: behave as if unset.
    return parsed;
}

}  // namespace

namespace vc::core::util
{

void setThreadBudget(int threads)
{
    gThreadBudget.store(threads > 0 ? threads : 0, std::memory_order_relaxed);
}

int threadBudget()
{
    int budget = gThreadBudget.load(std::memory_order_relaxed);
    if (budget < 0) {
        budget = budgetFromEnvironment();
        // Benign race: concurrent first callers resolve to the same value.
        gThreadBudget.store(budget, std::memory_order_relaxed);
    }
    return budget;
}

std::size_t clampWorkerCount(std::size_t desired)
{
    const int budget = threadBudget();
    if (budget <= 0)
        return desired;
    return std::min(desired, static_cast<std::size_t>(budget));
}

}  // namespace vc::core::util

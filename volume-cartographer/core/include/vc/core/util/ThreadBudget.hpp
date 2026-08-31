#pragma once

#include <cstddef>

namespace vc::core::util
{

/**
 * Process-wide soft cap on the size of worker-thread pools.
 *
 * Several long-lived pools (chunk fetch/decode/probe schedulers, the corner
 * batch sampler) size themselves from hardware_concurrency() or from fixed
 * constants tuned for an interactive GUI session. When many tracing processes
 * run concurrently on one machine -- the batch-tracing case -- that
 * oversubscribes the CPU badly: a single vc_grow_seg_from_seed was measured
 * creating 172 OS threads on a 32-core host even with "thread_limit": 1 in the
 * params JSON, because thread_limit only reached omp_set_num_threads().
 *
 * This budget is advisory and only ever *lowers* a pool's worker count. It
 * changes how much work runs concurrently, never what work is performed, so
 * results are bit-identical -- pools here are work queues whose outputs do not
 * depend on the number of workers draining them.
 *
 * Resolution order (first wins):
 *   1. setThreadBudget() -- e.g. from the "thread_limit" params key.
 *   2. The VC_MAX_THREADS environment variable.
 *   3. Unlimited (0), preserving historical behaviour.
 */

/** Set the cap. Values < 1 clear it (unlimited). */
void setThreadBudget(int threads);

/** Current cap, or 0 when unlimited. */
int threadBudget();

/** @p desired lowered to the budget, or unchanged when unlimited. */
std::size_t clampWorkerCount(std::size_t desired);

}  // namespace vc::core::util

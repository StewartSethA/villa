#pragma once

/**
 * @file AbfSolver.hpp
 * @brief The linear solver ABF++ uses, with an optional GPU path.
 *
 * WHAT THIS REPLACES
 * ------------------
 * OpenABF's ABFPlusPlus takes its solver as a template parameter, defaulting to
 * Eigen::SparseLU. Each ABF++ iteration factorises and solves one reduced
 * system; instrumented at OpenABF.hpp's ABF++ solve site, that factorisation is
 * 39 % of flatten time at 7,688 rows rising to 59 % at 154,568, while the
 * triangular solve is 0.000 s. So the factorisation is the only part worth
 * moving, and a perfect free one caps the whole flatten at about 2.4x.
 *
 * WHY ILU(0)+BiCGSTAB AND NOT A DIRECT GPU FACTORISATION
 * -----------------------------------------------------
 * cuSOLVER's device sparse API is exactly two solvers -- QR and Cholesky.
 * ABF's reduced system is a SADDLE-POINT matrix, hence indefinite, so Cholesky
 * is illegal; and QR is the wrong factorisation for a 2D-mesh matrix at ~14
 * nnz/row: measured on the same CPU at 10 k rows, Eigen SparseLU 0.073 s
 * against Eigen SparseQR 119.473 s, with nnz(R) = 27,078,208, a 305x fill-in.
 * `csrlsvlu` exists but is HOST ONLY (the name ends in `Host`; a lowercase-only
 * grep once truncated before the capital H and manufactured a device LU that
 * does not exist). ILU(0) instead carries A's sparsity by construction, so
 * fill-in is exactly zero and the 305x cannot occur.
 *
 * MEASURED, on a real ABF++ system dumped from this flattener
 * (49,928 rows, 693,944 nnz, 13.9/row):
 *     CPU Eigen::SparseLU   2.600 s   true rel.resid 5.85e-14
 *     GPU ILU(0)+BiCGSTAB   0.377 s   true rel.resid 9.82e-09   (6.9x)
 *                           = 0.017 s ILU(0) + 133 iterations
 * ILU(0) factorisation is FLAT in problem size (0.018-0.022 s from 22.5 k to
 * 122.5 k rows) where SparseLU runs 0.43 -> 4.85 s. It took 133 iterations on
 * the real system against 4 on a diagonally-dominant proxy, so iteration count
 * is a property of this matrix class and must never be inherited from a
 * synthetic benchmark.
 *
 * THE ACCEPTANCE BAR IS THE TRUE RESIDUAL, COMPUTED FROM A
 * --------------------------------------------------------
 * This solver once reported CONVERGED at a recurrence residual of 4.7e-09 while
 * the true ||Ax-b||/||b|| was 7.87e-02. Every internal signal said it had
 * worked. The cause was Eigen's SparseMatrix being COLUMN-major by default:
 * handing its arrays to cusparseCreateCsr describes A-transpose, and BiCGSTAB
 * then converges beautifully onto A^-T b -- a fixed point, so the wrong answer
 * was invariant to three separate "fixes". Two guards are therefore permanent
 * and must not be removed:
 *   1. an SpMV self-test against Eigen on first factorisation, which catches
 *      any layout or indexing slip loudly rather than silently;
 *   2. an explicit ||Ax-b||/||b|| computed from the ORIGINAL matrix after every
 *      solve. A solver benchmark without that is measuring the speed of a wrong
 *      answer.
 * If either fails, this class silently falls back to Eigen::SparseLU for that
 * solve. The GPU path is therefore an optimisation that cannot change results:
 * the worst case is CPU speed, never a wrong flatten.
 *
 * SELECTION
 *   VC_ABF_SOLVER=auto (default)  GPU when a device is usable, else CPU
 *   VC_ABF_SOLVER=cpu             force Eigen::SparseLU
 *   VC_ABF_SOLVER=gpu             force GPU; still falls back on a bad residual
 *   VC_ABF_GPU_TOL=1e-8           BiCGSTAB target, and the residual bar
 * Built only when VC_ENABLE_GPU_ABF=ON; otherwise this is Eigen::SparseLU with
 * no overhead and the same API, so callers need no conditional code.
 */

#include <Eigen/Sparse>
#include <memory>

namespace volcart::flattening
{

/// Statistics from the most recent solve, for logging and for tests that must
/// assert the GPU path actually ran rather than assuming it did.
struct AbfSolveStats {
    bool used_gpu{false};       ///< false means the CPU fallback produced this
    int iterations{0};          ///< BiCGSTAB iterations (0 on the CPU path)
    double true_residual{0.0};  ///< ||Ax-b||/||b|| from the ORIGINAL matrix
    double factor_seconds{0.0};
    double solve_seconds{0.0};
    const char* fallback_reason{nullptr};  ///< non-null iff we fell back
};

/**
 * @brief Eigen-concept sparse solver for ABF++, GPU-accelerated when available.
 *
 * Models the subset of the Eigen sparse-solver concept OpenABF uses:
 * `compute(A)`, `solve(b)`, `info()`. Drop-in by typedef.
 */
class AbfSolver
{
public:
    using Scalar = double;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;

    AbfSolver();
    ~AbfSolver();
    AbfSolver(AbfSolver&&) noexcept;
    AbfSolver& operator=(AbfSolver&&) noexcept;
    AbfSolver(const AbfSolver&) = delete;
    AbfSolver& operator=(const AbfSolver&) = delete;

    /// Analyse and factorise. Never throws; failure is reported via info().
    void compute(const SparseMatrix& A);

    /// Solve A x = rhs. Returns sparse because OpenABF consumes the result in a
    /// sparse expression (`bstar1 - JstarT * deltaLambda2`), where a dense
    /// return would not compile.
    SparseMatrix solve(const SparseMatrix& rhs);

    Eigen::ComputationInfo info() const;

    /// Statistics for the last solve. Also what the regression test reads.
    const AbfSolveStats& stats() const;

    /// True when this build can use a GPU at all (VC_ENABLE_GPU_ABF and a
    /// device present). Tests skip rather than fail when false.
    static bool gpuAvailable();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace volcart::flattening

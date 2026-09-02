#include "vc/flattening/AbfSolver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef VC_HAVE_GPU_ABF
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#endif

namespace volcart::flattening
{

namespace
{
using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

/// auto | cpu | gpu. Read once; an env lookup per ABF iteration is pointless.
enum class Mode { Auto, Cpu, Gpu };
Mode selectedMode()
{
    static const Mode m = [] {
        const char* e = std::getenv("VC_ABF_SOLVER");
        if (e == nullptr) { return Mode::Auto; }
        std::string v{e};
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (v == "cpu") { return Mode::Cpu; }
        if (v == "gpu") { return Mode::Gpu; }
        return Mode::Auto;
    }();
    return m;
}

double gpuTolerance()
{
    static const double t = [] {
        const char* e = std::getenv("VC_ABF_GPU_TOL");
        double v = (e != nullptr) ? std::atof(e) : 0.0;
        return (v > 0.0) ? v : 1e-8;
    }();
    return t;
}
}  // namespace

// ---------------------------------------------------------------------------

struct AbfSolver::Impl {
    Eigen::SparseMatrix<double> A;  // kept: the true residual must be computed
                                    // against the ORIGINAL matrix, never
                                    // against whatever the device holds.
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> lu;
    bool lu_ready{false};
    Eigen::ComputationInfo info{Eigen::Success};
    AbfSolveStats stats;

#ifdef VC_HAVE_GPU_ABF
    // Row-major copy is THE fix for the CSC-as-CSR bug described in the header.
    // Eigen is column-major by default and the mistake is silent on a
    // structurally symmetric matrix, which this one is.
    Eigen::SparseMatrix<double, Eigen::RowMajor> Arow;
    bool gpu_ready{false};
    int n{0}, nnz{0};

    cusparseHandle_t H{nullptr};
    cublasHandle_t CB{nullptr};
    double *dA{nullptr}, *dM{nullptr};
    int *dRow{nullptr}, *dCol{nullptr};
    double *dB{nullptr}, *dX{nullptr}, *dR{nullptr}, *dR0{nullptr}, *dP{nullptr},
        *dV{nullptr}, *dS{nullptr}, *dT{nullptr}, *dY{nullptr}, *dZ{nullptr},
        *dTmp{nullptr}, *dPh{nullptr}, *dSh{nullptr};
    cusparseSpMatDescr_t mA{nullptr}, mL{nullptr}, mU{nullptr};
    cusparseDnVecDescr_t vX{nullptr}, vY{nullptr}, vZ{nullptr}, vP{nullptr},
        vV{nullptr}, vS{nullptr}, vT{nullptr}, vTmp{nullptr}, vR{nullptr},
        vPh{nullptr}, vSh{nullptr};
    cusparseSpSVDescr_t sL{nullptr}, sU{nullptr};
    void *spmvBuf{nullptr}, *iluBuf{nullptr}, *bufL{nullptr}, *bufU{nullptr};

    bool setupGpu();
    void teardownGpu();
    bool selfTestSpMV();
    bool bicgstab(const double* b, double* x, int& iters);
    void precondition(double* din, double* dout);
    ~Impl() { teardownGpu(); }
#else
    ~Impl() = default;
#endif
};

// ---------------------------------------------------------------------------
#ifdef VC_HAVE_GPU_ABF

#define CUDA_OK(x)                                                             \
    do {                                                                       \
        if ((x) != cudaSuccess) { return false; }                              \
    } while (0)
#define CUSP_OK(x)                                                             \
    do {                                                                       \
        if ((x) != CUSPARSE_STATUS_SUCCESS) { return false; }                  \
    } while (0)

void AbfSolver::Impl::teardownGpu()
{
    if (sL != nullptr) { cusparseSpSV_destroyDescr(sL); sL = nullptr; }
    if (sU != nullptr) { cusparseSpSV_destroyDescr(sU); sU = nullptr; }
    for (auto* d : {&mA, &mL, &mU}) {
        if (*d != nullptr) { cusparseDestroySpMat(*d); *d = nullptr; }
    }
    for (auto* d : {&vX, &vY, &vZ, &vP, &vV, &vS, &vT, &vTmp, &vR, &vPh, &vSh}) {
        if (*d != nullptr) { cusparseDestroyDnVec(*d); *d = nullptr; }
    }
    for (auto** p : {&spmvBuf, &iluBuf, &bufL, &bufU}) {
        if (*p != nullptr) { cudaFree(*p); *p = nullptr; }
    }
    for (double** p : {&dA, &dM, &dB, &dX, &dR, &dR0, &dP, &dV, &dS, &dT, &dY,
                       &dZ, &dTmp, &dPh, &dSh}) {
        if (*p != nullptr) { cudaFree(*p); *p = nullptr; }
    }
    for (int** p : {&dRow, &dCol}) {
        if (*p != nullptr) { cudaFree(*p); *p = nullptr; }
    }
    if (CB != nullptr) { cublasDestroy(CB); CB = nullptr; }
    if (H != nullptr) { cusparseDestroy(H); H = nullptr; }
    gpu_ready = false;
}

bool AbfSolver::Impl::setupGpu()
{
    n = static_cast<int>(Arow.rows());
    nnz = static_cast<int>(Arow.nonZeros());
    if (n <= 0 || nnz <= 0) { return false; }

    CUSP_OK(cusparseCreate(&H));
    if (cublasCreate(&CB) != CUBLAS_STATUS_SUCCESS) { return false; }

    CUDA_OK(cudaMalloc(&dA, sizeof(double) * nnz));
    CUDA_OK(cudaMalloc(&dM, sizeof(double) * nnz));
    CUDA_OK(cudaMalloc(&dRow, sizeof(int) * (n + 1)));
    CUDA_OK(cudaMalloc(&dCol, sizeof(int) * nnz));
    for (double** p : {&dB, &dX, &dR, &dR0, &dP, &dV, &dS, &dT, &dY, &dZ, &dTmp,
                       &dPh, &dSh}) {
        CUDA_OK(cudaMalloc(p, sizeof(double) * n));
    }
    CUDA_OK(cudaMemcpy(dA, Arow.valuePtr(), sizeof(double) * nnz,
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dM, Arow.valuePtr(), sizeof(double) * nnz,
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dRow, Arow.outerIndexPtr(), sizeof(int) * (n + 1),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dCol, Arow.innerIndexPtr(), sizeof(int) * nnz,
                       cudaMemcpyHostToDevice));

    // ---- ILU(0), in place on dM. Fill-in is zero by construction. ----
    csrilu02Info_t ilu = nullptr;
    cusparseMatDescr_t dm = nullptr;
    CUSP_OK(cusparseCreateCsrilu02Info(&ilu));
    CUSP_OK(cusparseCreateMatDescr(&dm));
    cusparseSetMatType(dm, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(dm, CUSPARSE_INDEX_BASE_ZERO);
    int bs = 0;
    CUSP_OK(cusparseDcsrilu02_bufferSize(H, n, nnz, dm, dM, dRow, dCol, ilu, &bs));
    CUDA_OK(cudaMalloc(&iluBuf, bs > 0 ? bs : 1));
    CUSP_OK(cusparseDcsrilu02_analysis(H, n, nnz, dm, dM, dRow, dCol, ilu,
                                       CUSPARSE_SOLVE_POLICY_USE_LEVEL, iluBuf));
    CUSP_OK(cusparseDcsrilu02(H, n, nnz, dm, dM, dRow, dCol, ilu,
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, iluBuf));
    int pivot = -1;
    const bool zeroPivot =
        cusparseXcsrilu02_zeroPivot(H, ilu, &pivot) == CUSPARSE_STATUS_ZERO_PIVOT;
    cusparseDestroyCsrilu02Info(ilu);
    cusparseDestroyMatDescr(dm);
    // A structurally singular preconditioner is not fatal on its own, but on a
    // saddle-point matrix it reliably stalls BiCGSTAB. Refuse here rather than
    // burn maxit iterations and fall back afterwards.
    if (zeroPivot) { return false; }

    CUSP_OK(cusparseCreateCsr(&mA, n, n, nnz, dRow, dCol, dA, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F));
    CUSP_OK(cusparseCreateCsr(&mL, n, n, nnz, dRow, dCol, dM, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F));
    CUSP_OK(cusparseCreateCsr(&mU, n, n, nnz, dRow, dCol, dM, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F));
    cusparseFillMode_t lo = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t up = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(mL, CUSPARSE_SPMAT_FILL_MODE, &lo, sizeof(lo));
    cusparseSpMatSetAttribute(mL, CUSPARSE_SPMAT_DIAG_TYPE, &unit, sizeof(unit));
    cusparseSpMatSetAttribute(mU, CUSPARSE_SPMAT_FILL_MODE, &up, sizeof(up));
    cusparseSpMatSetAttribute(mU, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit,
                              sizeof(nonunit));

    double* dv[] = {dX, dY, dZ, dP, dV, dS, dT, dTmp, dR, dPh, dSh};
    cusparseDnVecDescr_t* vv[] = {&vX, &vY, &vZ, &vP, &vV, &vS,
                                  &vT, &vTmp, &vR, &vPh, &vSh};
    for (int i = 0; i < 11; ++i) {
        CUSP_OK(cusparseCreateDnVec(vv[i], n, dv[i], CUDA_R_64F));
    }

    const double one = 1.0, zero = 0.0;
    size_t bsz = 0;
    CUSP_OK(cusparseSpMV_bufferSize(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    mA, vX, &zero, vR, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &bsz));
    CUDA_OK(cudaMalloc(&spmvBuf, bsz > 0 ? bsz : 1));

    CUSP_OK(cusparseSpSV_createDescr(&sL));
    CUSP_OK(cusparseSpSV_createDescr(&sU));
    size_t bl = 0, bu = 0;
    CUSP_OK(cusparseSpSV_bufferSize(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    mL, vTmp, vY, CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT, sL, &bl));
    CUSP_OK(cusparseSpSV_bufferSize(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    mU, vY, vZ, CUDA_R_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT, sU, &bu));
    CUDA_OK(cudaMalloc(&bufL, bl > 0 ? bl : 1));
    CUDA_OK(cudaMalloc(&bufU, bu > 0 ? bu : 1));
    CUSP_OK(cusparseSpSV_analysis(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mL,
                                  vTmp, vY, CUDA_R_64F,
                                  CUSPARSE_SPSV_ALG_DEFAULT, sL, bufL));
    CUSP_OK(cusparseSpSV_analysis(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mU,
                                  vY, vZ, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                                  sU, bufU));
    CUDA_OK(cudaDeviceSynchronize());

    if (!selfTestSpMV()) { return false; }
    gpu_ready = true;
    return true;
}

/// One SpMV against Eigen. This is the guard that would have caught the
/// CSC-as-CSR bug on the first run instead of after an afternoon of plausible
/// convergence onto A^-T b. Cost is one matrix-vector product per factorisation.
bool AbfSolver::Impl::selfTestSpMV()
{
    Eigen::VectorXd w(n);
    for (int i = 0; i < n; ++i) { w[i] = std::cos(i * 0.003) + 0.1; }
    CUDA_OK(cudaMemcpy(dTmp, w.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    CUSP_OK(cusparseSpMV(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mA, vTmp,
                         &zero, vV, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                         spmvBuf));
    Eigen::VectorXd got(n);
    CUDA_OK(cudaMemcpy(got.data(), dV, sizeof(double) * n, cudaMemcpyDeviceToHost));
    const Eigen::VectorXd want = A * w;
    const double denom = std::max(1e-30, want.norm());
    return (got - want).norm() / denom < 1e-12;
}

void AbfSolver::Impl::precondition(double* din, double* dout)
{
    const double one = 1.0;
    cudaMemcpy(dTmp, din, sizeof(double) * n, cudaMemcpyDeviceToDevice);
    cusparseSpSV_solve(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mL, vTmp, vY,
                       CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, sL);
    cusparseSpSV_solve(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mU, vY, vZ,
                       CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, sU);
    cudaMemcpy(dout, dZ, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}

/// BiCGSTAB (Van der Vorst 1992; Barrett et al. "Templates"; NVIDIA's bicgstab
/// sample), left-preconditioned by the ILU(0) factors.
bool AbfSolver::Impl::bicgstab(const double* b, double* x, int& iters)
{
    const double one = 1.0;
    const double tol = gpuTolerance();
    const int maxit = 3000;

    CUDA_OK(cudaMemcpy(dB, b, sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dX, 0, sizeof(double) * n));
    CUDA_OK(cudaMemcpy(dR, dB, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(dR0, dR, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemset(dP, 0, sizeof(double) * n));
    CUDA_OK(cudaMemset(dV, 0, sizeof(double) * n));

    double bnorm = 0;
    cublasDnrm2(CB, n, dB, 1, &bnorm);
    if (bnorm == 0) { bnorm = 1; }

    double rho = 1, alpha = 1, omega = 1, rho_new = 0, rn = 0;
    int it = 0;
    bool converged = false;
    for (; it < maxit; ++it) {
        cublasDdot(CB, n, dR0, 1, dR, 1, &rho_new);
        if (std::fabs(rho_new) < 1e-300) { break; }  // breakdown
        const double beta = (rho_new / rho) * (alpha / omega);
        // p = r + beta*(p - omega*v). Subtract omega*v FIRST, then scale: doing
        // it the other way round scales a term that has not been corrected yet.
        const double nomega = -omega;
        cublasDaxpy(CB, n, &nomega, dV, 1, dP, 1);
        cublasDscal(CB, n, &beta, dP, 1);
        cublasDaxpy(CB, n, &one, dR, 1, dP, 1);

        precondition(dP, dPh);
        const double zero = 0.0;
        cusparseSpMV(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mA, vPh, &zero,
                     vV, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmvBuf);

        double r0v = 0;
        cublasDdot(CB, n, dR0, 1, dV, 1, &r0v);
        if (std::fabs(r0v) < 1e-300) { break; }
        alpha = rho_new / r0v;

        cudaMemcpy(dS, dR, sizeof(double) * n, cudaMemcpyDeviceToDevice);
        const double na = -alpha;
        cublasDaxpy(CB, n, &na, dV, 1, dS, 1);
        cublasDnrm2(CB, n, dS, 1, &rn);
        if (rn / bnorm < tol) {
            cublasDaxpy(CB, n, &alpha, dPh, 1, dX, 1);
            converged = true;
            ++it;
            break;
        }

        precondition(dS, dSh);
        cusparseSpMV(H, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mA, vSh, &zero,
                     vT, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmvBuf);

        double ts = 0, tt = 0;
        cublasDdot(CB, n, dT, 1, dS, 1, &ts);
        cublasDdot(CB, n, dT, 1, dT, 1, &tt);
        if (tt == 0) { break; }
        omega = ts / tt;

        cublasDaxpy(CB, n, &alpha, dPh, 1, dX, 1);
        cublasDaxpy(CB, n, &omega, dSh, 1, dX, 1);
        cudaMemcpy(dR, dS, sizeof(double) * n, cudaMemcpyDeviceToDevice);
        const double no = -omega;
        cublasDaxpy(CB, n, &no, dT, 1, dR, 1);
        cublasDnrm2(CB, n, dR, 1, &rn);
        if (rn / bnorm < tol) { converged = true; ++it; break; }
        if (std::fabs(omega) < 1e-300) { break; }
        rho = rho_new;
    }
    cudaDeviceSynchronize();
    CUDA_OK(cudaMemcpy(x, dX, sizeof(double) * n, cudaMemcpyDeviceToHost));
    iters = it;
    return converged;
}

#undef CUDA_OK
#undef CUSP_OK
#endif  // VC_HAVE_GPU_ABF

// ---------------------------------------------------------------------------

AbfSolver::AbfSolver() : impl_(std::make_unique<Impl>()) {}
AbfSolver::~AbfSolver() = default;
AbfSolver::AbfSolver(AbfSolver&&) noexcept = default;
AbfSolver& AbfSolver::operator=(AbfSolver&&) noexcept = default;

bool AbfSolver::gpuAvailable()
{
#ifdef VC_HAVE_GPU_ABF
    static const bool ok = [] {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }();
    return ok;
#else
    return false;
#endif
}

void AbfSolver::compute(const SparseMatrix& A)
{
    auto& d = *impl_;
    d.A = A;
    d.A.makeCompressed();
    d.info = Eigen::Success;
    d.stats = AbfSolveStats{};
    d.lu_ready = false;

    const auto t0 = Clock::now();
#ifdef VC_HAVE_GPU_ABF
    if (selectedMode() != Mode::Cpu && gpuAvailable()) {
        // Row-major conversion is THE fix, and it is silent when wrong: see the
        // header. Never hand a default Eigen SparseMatrix to cusparseCreateCsr.
        d.Arow = d.A;
        d.Arow.makeCompressed();
        if (d.setupGpu()) {
            d.stats.factor_seconds = secs(t0, Clock::now());
            d.stats.used_gpu = true;
            return;
        }
        d.teardownGpu();
        d.stats.fallback_reason = "GPU setup, ILU(0) or SpMV self-test failed";
    } else if (selectedMode() == Mode::Gpu) {
        d.stats.fallback_reason = "VC_ABF_SOLVER=gpu but no usable CUDA device";
    }
#endif
    d.lu.compute(d.A);
    d.lu_ready = d.lu.info() == Eigen::Success;
    d.info = d.lu.info();
    d.stats.factor_seconds = secs(t0, Clock::now());
}

AbfSolver::SparseMatrix AbfSolver::solve(const SparseMatrix& rhs)
{
    auto& d = *impl_;
    const auto t0 = Clock::now();
    const Eigen::VectorXd b = Eigen::VectorXd(rhs.col(0));

#ifdef VC_HAVE_GPU_ABF
    if (d.gpu_ready) {
        Eigen::VectorXd x(d.n);
        int iters = 0;
        const bool conv = d.bicgstab(b.data(), x.data(), iters);
        // THE ACCEPTANCE BAR: ||Ax-b||/||b|| against the ORIGINAL matrix. The
        // recurrence residual inside BiCGSTAB once read 4.7e-09 while this read
        // 7.87e-02, so the internal signal is not evidence of anything.
        const double denom = std::max(1e-30, b.norm());
        const double resid = (d.A * x - b).norm() / denom;
        d.stats.iterations = iters;
        d.stats.true_residual = resid;
        if (conv && std::isfinite(resid) && resid < 1e-6) {
            d.stats.used_gpu = true;
            d.stats.solve_seconds = secs(t0, Clock::now());
            d.info = Eigen::Success;
            return x.sparseView();
        }
        // Not good enough -- fall through to the CPU. A flatten must never be
        // made wrong by an optimisation; the worst case here is CPU speed.
        d.stats.fallback_reason = "GPU true residual above 1e-6";
        d.stats.used_gpu = false;
    }
#endif
    if (!d.lu_ready) {
        d.lu.compute(d.A);
        d.lu_ready = d.lu.info() == Eigen::Success;
        if (!d.lu_ready) {
            d.info = d.lu.info();
            return SparseMatrix(rhs.rows(), 1);
        }
    }
    // Forward the SPARSE rhs straight to SparseLU rather than round-tripping
    // through a dense vector. Eigen's sparse-rhs path inserts only exact
    // nonzeros; a dense solve followed by sparseView() is a different sequence
    // of operations, and on a 10-iteration ABF++ loop that difference compounds
    // -- measured 0.41-0.58 voxels mean displacement, which is a real change to
    // the surface and not rounding. The CPU path must be bit-identical to
    // upstream or the fallback is not a fallback.
    SparseMatrix x = d.lu.solve(rhs);
    d.info = d.lu.info();
    const Eigen::VectorXd xd = Eigen::VectorXd(x.col(0));
    const double denom = std::max(1e-30, b.norm());
    d.stats.true_residual = (d.A * xd - b).norm() / denom;
    d.stats.solve_seconds = secs(t0, Clock::now());
    return x;
}

Eigen::ComputationInfo AbfSolver::info() const { return impl_->info; }
const AbfSolveStats& AbfSolver::stats() const { return impl_->stats; }

}  // namespace volcart::flattening

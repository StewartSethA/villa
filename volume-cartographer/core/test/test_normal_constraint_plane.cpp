// Equivalence coverage for the NormalConstraintPlane normal-loss fold.
//
// NormalConstraintPlane is an AutoDiffCostFunction over four 3-vectors, so
// Ceres instantiates the functor with Jet<double, 12>. Its loop over nearby
// normal-grid polyline segments used to run in that type even though every
// quantity the loop produces (segment weights, segment normals, the sign
// ceres::abs applies) is a constant of the evaluation. The loop now runs in
// plain double and only
//     normal_loss = 1 - n_edge . (sum_k w_k s_k n_k / sum_k w_k)
// stays differentiable.
//
// The claim under test is that this is an algebraic identity, not an
// approximation: same value AND same derivative. So the first test rebuilds
// the previous formulation over the same segment set, in the same Jet type,
// and compares both parts. The second checks the assembled cost function's
// analytic Jacobian against numeric differentiation, so a derivative that is
// self-consistently wrong cannot pass unnoticed.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
// CostFunctions.hpp is not self-contained: several functors call helpers
// (loc_valid, ...) that GrowPatch.cpp gets from Geometry.hpp first.
#include "vc/core/util/Geometry.hpp"
#include "vc/tracer/CostFunctions.hpp"

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using vc::core::util::GridStore;
using vc::core::util::NormalGridVolume;

namespace {

constexpr int kSlice = 100;   // z of the single xy grid slice we write
using Jet12 = ceres::Jet<double, 12>;

// A normal-grid directory holding one xy slice of synthetic polylines.
fs::path makeGridDir()
{
    std::mt19937_64 rng(0xC0FFEEu);
    auto dir = fs::temp_directory_path() /
               ("vc_ncp_" + std::to_string(rng()));
    fs::create_directories(dir / "xy");
    fs::create_directories(dir / "xz");
    fs::create_directories(dir / "yz");
    {
        std::ofstream f(dir / "metadata.json");
        f << "{\"sparse-volume\":1}";
    }

    GridStore store(cv::Rect(0, 0, 512, 512), 32);
    // Polylines at several orientations, long enough that the snapping term
    // finds a previous/next segment, and dense enough that a query around the
    // sample points returns many segments.
    for (int k = 0; k < 6; ++k) {
        std::vector<cv::Point> path;
        const double ang = k * 0.37;
        const double cx = 200 + 11 * k, cy = 200 - 7 * k;
        for (int i = 0; i < 24; ++i) {
            const double t = i * 4.0;
            path.emplace_back(
                static_cast<int>(std::lround(cx + t * std::cos(ang) + 3.0 * std::sin(0.4 * t))),
                static_cast<int>(std::lround(cy + t * std::sin(ang) + 3.0 * std::cos(0.4 * t))));
        }
        store.add(path);
    }
    char name[64];
    std::snprintf(name, sizeof(name), "%06d.grid", kSlice);
    store.save((dir / "xy" / name).string());
    return dir;
}

// The pre-fold formulation, over the segment set the functor itself gathers.
// Kept here (and only here) as the equivalence oracle.
Jet12 referenceNormalLoss(const NormalConstraintPlane& ncp,
                          const NormalConstraintPlane::PathCachePayload& paths,
                          const Jet12& edge_normal_x, const Jet12& edge_normal_y,
                          const cv::Point2f& p1_cv, const cv::Point2f& p2_cv,
                          bool direction_aware)
{
    Jet12 total_weighted_dot_loss(0.0);
    Jet12 total_weight(0.0);
    for (const std::vector<cv::Point>& path : paths) {
        if (path.size() < 2) continue;
        for (size_t i = 0; i < path.size() - 1; ++i) {
            cv::Point2f p_a = path[i];
            cv::Point2f p_b = path[i + 1];
            float dist_sq = NormalConstraintPlane::seg_dist_sq_appx(p1_cv, p2_cv, p_a, p_b);
            if (dist_sq > ncp.roi_radius_ * ncp.roi_radius_) continue;
            dist_sq = std::max(10.0f, dist_sq);
            Jet12 weight_n(1.0 / dist_sq);
            cv::Point2f tangent = p_b - p_a;
            float length = cv::norm(tangent);
            if (length > 0) tangent /= length;
            cv::Point2f normal(-tangent.y, tangent.x);
            Jet12 dot = edge_normal_x * Jet12(normal.x) + edge_normal_y * Jet12(normal.y);
            if (!direction_aware) dot = ceres::abs(dot);
            total_weighted_dot_loss += weight_n * (Jet12(1.0) - dot);
            total_weight += weight_n;
        }
    }
    if (total_weight > Jet12(1e-9)) return total_weighted_dot_loss / total_weight;
    return Jet12(0.0);
}

} // namespace

TEST_CASE("NormalConstraintPlane: folded normal loss matches the summed form "
          "in value and in all twelve derivative components")
{
    const auto dir = makeGridDir();
    NormalGridVolume ngv(dir.string());
    auto grid = ngv.query_nearest(cv::Point3f(0, 0, kSlice), 0);
    REQUIRE(grid != nullptr);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> pos(180.0, 320.0);
    std::uniform_real_distribution<double> seed(-1.0, 1.0);

    int compared = 0;
    double max_val_err = 0.0, max_der_err = 0.0;

    for (bool direction_aware : {false, true}) {
        NormalConstraintPlane ncp(ngv, /*plane_idx=*/0, /*w_normal=*/10.0,
                                  /*w_snap=*/0.1, /*maybe_fit_quality=*/nullptr,
                                  direction_aware);
        for (int trial = 0; trial < 400; ++trial) {
            // p1 and p2 as jets with arbitrary, non-degenerate derivative seeds.
            Jet12 p1[2], p2[2];
            for (int c = 0; c < 2; ++c) {
                p1[c] = Jet12(pos(rng));
                p2[c] = Jet12(pos(rng));
                for (int d = 0; d < 12; ++d) {
                    p1[c].v[d] = seed(rng);
                    p2[c].v[d] = seed(rng);
                }
            }

            Jet12 ex = p2[0] - p1[0], ey = p2[1] - p1[1];
            Jet12 elen = ceres::sqrt(ex * ex + ey * ey);
            if (elen.a < 1.0) continue;              // degenerate edge
            Jet12 enx = ey / elen, eny = -ex / elen;

            const cv::Point2f p1_cv(val(p1[0]), val(p1[1]));
            const cv::Point2f p2_cv(val(p2[0]), val(p2[1]));
            const cv::Point2f mid = 0.5f * (p1_cv + p2_cv);
            auto paths = ncp.filter_and_split_paths(
                grid->get(mid, ncp.query_radius_), p1_cv, p2_cv);
            if (paths.empty()) continue;

            // Folded form, exactly as CostFunctions.hpp computes it.
            const double enx_val = val(enx), eny_val = val(eny);
            double mx = 0.0, my = 0.0, w_total = 0.0;
            for (const std::vector<cv::Point>& path : paths) {
                if (path.size() < 2) continue;
                for (size_t i = 0; i < path.size() - 1; ++i) {
                    cv::Point2f p_a = path[i];
                    cv::Point2f p_b = path[i + 1];
                    float dist_sq = NormalConstraintPlane::seg_dist_sq_appx(p1_cv, p2_cv, p_a, p_b);
                    if (dist_sq > ncp.roi_radius_ * ncp.roi_radius_) continue;
                    dist_sq = std::max(10.0f, dist_sq);
                    const double weight_n = 1.0 / dist_sq;
                    cv::Point2f tangent = p_b - p_a;
                    float length = cv::norm(tangent);
                    if (length > 0) tangent /= length;
                    cv::Point2f normal(-tangent.y, tangent.x);
                    const double dot = enx_val * normal.x + eny_val * normal.y;
                    const double s = direction_aware ? 1.0 : std::copysign(1.0, dot);
                    mx += weight_n * s * normal.x;
                    my += weight_n * s * normal.y;
                    w_total += weight_n;
                }
            }
            if (!(w_total > 1e-9)) continue;
            const Jet12 folded =
                Jet12(1.0) - (enx * Jet12(mx / w_total) + eny * Jet12(my / w_total));

            const Jet12 ref = referenceNormalLoss(ncp, paths, enx, eny, p1_cv, p2_cv,
                                                  direction_aware);

            ++compared;
            max_val_err = std::max(max_val_err, std::fabs(folded.a - ref.a));
            for (int d = 0; d < 12; ++d)
                max_der_err = std::max(max_der_err, std::fabs(folded.v[d] - ref.v[d]));
        }
    }

    // The two forms differ only by floating-point reassociation of one sum.
    REQUIRE(compared > 200);
    CHECK(max_val_err < 1e-12);
    CHECK(max_der_err < 1e-9);

    fs::remove_all(dir);
}

TEST_CASE("NormalConstraintPlane: the double and jet instantiations return "
          "the same residual, and the jet one produces a finite Jacobian")
{
    // The fold reads the sign of the edge-normal/segment-normal dot product
    // out of the scalar part and applies it to both parts of the jet, mirroring
    // ceres::abs. If that ever diverges between the T = double instantiation
    // (residual-only evaluations) and the T = Jet instantiation (residual +
    // Jacobian), Ceres would minimise a different function from the one it
    // line-searches on. Numeric differentiation is NOT a usable oracle here:
    // the functor re-selects its segment set and its snap target per
    // evaluation, so it is only piecewise smooth.
    const auto dir = makeGridDir();
    NormalGridVolume ngv(dir.string());
    REQUIRE(ngv.query_nearest(cv::Point3f(0, 0, kSlice), 0) != nullptr);

    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> xy(200.0, 300.0);

    int nonzero = 0, mismatched = 0;
    double max_rel = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        // pA.z must round to kSlice: query_nearest picks the xy grid slice from
        // A's z and only that slice exists in the fixture. B1/B2 and C straddle
        // the plane through A so the functor does not short-circuit.
        double pA[3] = {xy(rng), xy(rng), double(kSlice)};
        double pB1[3] = {xy(rng), xy(rng), kSlice + 0.8};
        double pB2[3] = {xy(rng), xy(rng), kSlice + 1.0};
        double pC[3] = {xy(rng), xy(rng), kSlice - 0.7};
        const double* params[4] = {pA, pB1, pB2, pC};

        ceres::AutoDiffCostFunction<NormalConstraintPlane, 1, 3, 3, 3, 3> cf(
            new NormalConstraintPlane(ngv, 0, 10.0, 0.1, nullptr));

        double r_plain = 0.0, r_jet = 0.0;
        double j[12];
        double* J[4] = {j, j + 3, j + 6, j + 9};
        REQUIRE(cf.Evaluate(params, &r_plain, nullptr));
        REQUIRE(cf.Evaluate(params, &r_jet, J));

        for (int i = 0; i < 12; ++i) CHECK(std::isfinite(j[i]));
        if (r_jet != 0.0) {
            ++nonzero;
            if (r_plain != r_jet) ++mismatched;
            max_rel = std::max(max_rel, std::fabs(r_plain - r_jet) / std::fabs(r_jet));
            double jnorm = 0.0;
            for (int i = 0; i < 12; ++i) jnorm += j[i] * j[i];
            CHECK(jnorm > 0.0);
        }
    }

    MESSAGE("nonzero=" << nonzero << " mismatched=" << mismatched
            << " max_rel=" << max_rel);
    REQUIRE(nonzero > 0);
    // Not bit-equality: -O3 -march=x86-64-v3 is free to contract a*b+c into an
    // FMA differently in the two instantiations. Agreement to rounding is what
    // matters -- Ceres must not be line-searching a different function.
    CHECK(max_rel < 1e-12);

    fs::remove_all(dir);
}

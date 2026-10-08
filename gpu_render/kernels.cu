// GPU reimplementation of the vc_render_tifxyz sampling path.
//
// Mirrors, op for op:
//   QuadSurface::gen()            (warpBilinearReplicateVec3f + warpNearestConstU8
//                                  + warpNearestConstVec3f + invalidation pass)
//   prepareBaseAndDirs()          (scale_seg, normalizeNormals, ds_scale)
//   sampleTileSlicesImpl() /      (ChunkSampler::inBounds + sampleTrilinear,
//   readMultiSliceImpl()           clamp, uint8 truncation)
//
// Compiled with --fmad=false so every FMA is explicit and matches std::fma().

#define CHUNK_VOX (128 * 128 * 128)

// ---------------------------------------------------------------- gen_band
extern "C" __global__ void gen_band(
    const float* __restrict__ pts,       // (srows, scols, 3)
    const unsigned char* __restrict__ valid,  // (srows, scols)
    const float* __restrict__ ncache,    // (srows, scols, 3)
    int srows, int scols,
    float ox, float oy, float sxw, float syw,
    int W, int H,
    float scaleSeg, float dsScale,
    float* __restrict__ base,            // (H, W, 3)
    float* __restrict__ dirs)            // (H, W, 3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    int i = idx % W, j = idx / W;

    const float qnan = __int_as_float(0x7fc00000);

    // ---- bilinear warp with replicate border (coords) ----
    // dst pixel (j,i) of the cropped image == coords_big(j+4, i+4)
#if WARP_FMA
    float fx = __fmaf_rn((float)(i + 4), sxw, ox);
    float fy = __fmaf_rn((float)(j + 4), syw, oy);
#else
    float fx = ox + (float)(i + 4) * sxw;
    float fy = oy + (float)(j + 4) * syw;
#endif
    const float sxmax = (float)(scols - 1);
    const float symax = (float)(srows - 1);
    float cfx = fx < 0.0f ? 0.0f : (fx > sxmax ? sxmax : fx);
    float cfy = fy < 0.0f ? 0.0f : (fy > symax ? symax : fy);
    int x0 = (int)cfx; int x1 = x0 + 1; if (x1 > scols - 1) x1 = scols - 1;
    int y0 = (int)cfy; int y1 = y0 + 1; if (y1 > srows - 1) y1 = srows - 1;
    float wx = cfx - (float)x0, wy = cfy - (float)y0;
    float iwx = 1.0f - wx, iwy = 1.0f - wy;

    float c[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        float p00 = pts[((size_t)y0 * scols + x0) * 3 + k];
        float p01 = pts[((size_t)y0 * scols + x1) * 3 + k];
        float p10 = pts[((size_t)y1 * scols + x0) * 3 + k];
        float p11 = pts[((size_t)y1 * scols + x1) * 3 + k];
#if WARP_FMA
        // -ffp-contract=fast (GCC default) contracts the CPU expression like this
        float t0 = __fmaf_rn(p01, wx, p00 * iwx);
        float t1 = __fmaf_rn(p11, wx, p10 * iwx);
        c[k] = __fmaf_rn(t1, wy, t0 * iwy);
#else
        c[k] = (p00 * iwx + p01 * wx) * iwy + (p10 * iwx + p11 * wx) * wy;
#endif
    }

    // ---- nearest warp (validity, border 0) and (normals, border NaN) ----
    int nsx = (int)lroundf(fx);
    int nsy = (int)lroundf(fy);
    unsigned char v = 0;
    float n[3] = {qnan, qnan, qnan};
    if (nsy >= 0 && nsy < srows) {
        if (nsx >= 0 && nsx < scols) {
            v = valid[(size_t)nsy * scols + nsx];
#pragma unroll
            for (int k = 0; k < 3; k++) n[k] = ncache[((size_t)nsy * scols + nsx) * 3 + k];
        }
    }

    // ---- invalidation pass ----
    if (!v) {
        c[0] = c[1] = c[2] = qnan;
        n[0] = n[1] = n[2] = qnan;
    }

    // ---- prepareBaseAndDirs ----
    float b[3];
#pragma unroll
    for (int k = 0; k < 3; k++) b[k] = c[k] * scaleSeg;
    // normalizeNormals: skip NaN, skip L2 <= 0
    if (!isnan(n[0])) {
#if NORM_FMA
        float L2 = __fmaf_rn(n[2], n[2], __fmaf_rn(n[1], n[1], n[0] * n[0]));
#else
        float L2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
#endif
        if (L2 > 0.0f) {
            float s = sqrtf(L2);
            n[0] /= s; n[1] /= s; n[2] /= s;
        }
    }
#pragma unroll
    for (int k = 0; k < 3; k++) b[k] *= dsScale;

#pragma unroll
    for (int k = 0; k < 3; k++) {
        base[(size_t)idx * 3 + k] = b[k];
        dirs[(size_t)idx * 3 + k] = n[k];
    }
}

// ------------------------------------------------------------ mark_chunks
extern "C" __global__ void mark_chunks(
    const float* __restrict__ base,
    const float* __restrict__ dirs,
    const float* __restrict__ offsets, int nOff,
    int W, int H,
    int SZ, int SY, int SX,
    int CNZ, int CNY, int CNX,
    unsigned char* __restrict__ marks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    float b0 = base[(size_t)idx * 3 + 0], b1 = base[(size_t)idx * 3 + 1], b2 = base[(size_t)idx * 3 + 2];
    if (!isfinite(b0) || !isfinite(b1) || !isfinite(b2)) return;
    float d0 = dirs[(size_t)idx * 3 + 0], d1 = dirs[(size_t)idx * 3 + 1], d2 = dirs[(size_t)idx * 3 + 2];
    if (!isfinite(d0) || !isfinite(d1) || !isfinite(d2)) return;

    for (int t = 0; t < nOff; t++) {
        float off = offsets[t];
#if STEP_FMA
        float vx = __fmaf_rn(d0, off, b0);
        float vy = __fmaf_rn(d1, off, b1);
        float vz = __fmaf_rn(d2, off, b2);
#else
        float vx = b0 + d0 * off;
        float vy = b1 + d1 * off;
        float vz = b2 + d2 * off;
#endif
        if (!(vz >= 0.f && vy >= 0.f && vx >= 0.f && vz < (float)SZ && vy < (float)SY && vx < (float)SX))
            continue;
        int iz = (int)vz, iy = (int)vy, ix = (int)vx;
        int cz0 = iz >> 7, cy0 = iy >> 7, cx0 = ix >> 7;
        int cz1 = (iz + 1 < SZ) ? ((iz + 1) >> 7) : cz0;
        int cy1 = (iy + 1 < SY) ? ((iy + 1) >> 7) : cy0;
        int cx1 = (ix + 1 < SX) ? ((ix + 1) >> 7) : cx0;
        for (int cz = cz0; cz <= cz1; cz++)
            for (int cy = cy0; cy <= cy1; cy++)
                for (int cx = cx0; cx <= cx1; cx++)
                    marks[((size_t)cz * CNY + cy) * CNX + cx] = 1;
    }
}

// ------------------------------------------------------------ sample_band
__device__ __forceinline__ float sampleInt(
    int iz, int iy, int ix,
    int SZ, int SY, int SX, int CNY, int CNX,
    const int* __restrict__ lut, const unsigned char* __restrict__ pool)
{
    if ((unsigned)iz >= (unsigned)SZ || (unsigned)iy >= (unsigned)SY || (unsigned)ix >= (unsigned)SX)
        return 0.f;
    int slot = lut[(((size_t)(iz >> 7)) * CNY + (iy >> 7)) * CNX + (ix >> 7)];
    if (slot < 0) return 0.f;   // AllFill / missing chunk -> 0, as on CPU
    return (float)pool[(size_t)slot * CHUNK_VOX
                       + (((size_t)(iz & 127) << 14) | ((iy & 127) << 7) | (ix & 127))];
}

extern "C" __global__ void sample_band(
    const float* __restrict__ base,
    const float* __restrict__ dirs,
    const float* __restrict__ offsets, int nOff,
    int W, int H,
    int SZ, int SY, int SX, int CNY, int CNX,
    const int* __restrict__ lut,
    const unsigned char* __restrict__ pool,
    unsigned char* __restrict__ out)     // (nOff, H, W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    float b0 = base[(size_t)idx * 3 + 0], b1 = base[(size_t)idx * 3 + 1], b2 = base[(size_t)idx * 3 + 2];
    float d0 = dirs[(size_t)idx * 3 + 0], d1 = dirs[(size_t)idx * 3 + 1], d2 = dirs[(size_t)idx * 3 + 2];
    const size_t plane = (size_t)W * H;

    for (int t = 0; t < nOff; t++) {
        float off = offsets[t];
#if STEP_FMA
        float vx = __fmaf_rn(d0, off, b0);
        float vy = __fmaf_rn(d1, off, b1);
        float vz = __fmaf_rn(d2, off, b2);
#else
        float vx = b0 + d0 * off;
        float vy = b1 + d1 * off;
        float vz = b2 + d2 * off;
#endif
        unsigned char val = 0;
        if (vz >= 0.f && vy >= 0.f && vx >= 0.f && vz < (float)SZ && vy < (float)SY && vx < (float)SX) {
            int iz = (int)vz, iy = (int)vy, ix = (int)vx;
            float c000 = sampleInt(iz,     iy,     ix,     SZ, SY, SX, CNY, CNX, lut, pool);
            float c100 = sampleInt(iz + 1, iy,     ix,     SZ, SY, SX, CNY, CNX, lut, pool);
            float c010 = sampleInt(iz,     iy + 1, ix,     SZ, SY, SX, CNY, CNX, lut, pool);
            float c110 = sampleInt(iz + 1, iy + 1, ix,     SZ, SY, SX, CNY, CNX, lut, pool);
            float c001 = sampleInt(iz,     iy,     ix + 1, SZ, SY, SX, CNY, CNX, lut, pool);
            float c101 = sampleInt(iz + 1, iy,     ix + 1, SZ, SY, SX, CNY, CNX, lut, pool);
            float c011 = sampleInt(iz,     iy + 1, ix + 1, SZ, SY, SX, CNY, CNX, lut, pool);
            float c111 = sampleInt(iz + 1, iy + 1, ix + 1, SZ, SY, SX, CNY, CNX, lut, pool);
            float fz = vz - (float)iz, fy = vy - (float)iy, fx = vx - (float)ix;
            float c00 = __fmaf_rn(fx, c001 - c000, c000);
            float c01 = __fmaf_rn(fx, c011 - c010, c010);
            float c10 = __fmaf_rn(fx, c101 - c100, c100);
            float c11 = __fmaf_rn(fx, c111 - c110, c110);
            float cc0 = __fmaf_rn(fy, c01 - c00, c00);
            float cc1 = __fmaf_rn(fy, c11 - c10, c10);
            float vv = __fmaf_rn(fz, cc1 - cc0, cc0);
            if (vv < 0.f) vv = 0.f;
            if (vv > 255.f) vv = 255.f;
            val = (unsigned char)vv;   // truncation, matching T(v) for uint8
        }
        out[(size_t)t * plane + idx] = val;
    }
}

// ---------------------------------------------------- build_normals (_normalCache)
// QuadSurface::gen's normal cache build via grid_normal_int(), with the FMA
// contraction pattern GCC emits at -O3 -ffp-contract=fast.
extern "C" __global__ void build_normals(
    const float* __restrict__ pts, int rows, int cols,
    float* __restrict__ nc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int c = idx % cols, r = idx / cols;
    const float qn = __int_as_float(0x7fc00000);
    float o0 = qn, o1 = qn, o2 = qn;
    if (r >= 1 && r <= rows - 2 && c >= 1 && c <= cols - 2 &&
        pts[((size_t)r * cols + c) * 3 + 0] != -1.0f) {
        const float* xl = pts + ((size_t)r * cols + (c - 1)) * 3;
        const float* xr = pts + ((size_t)r * cols + (c + 1)) * 3;
        const float* yu = pts + ((size_t)(r - 1) * cols + c) * 3;
        const float* yd = pts + ((size_t)(r + 1) * cols + c) * 3;
        if (!(xl[0] == -1.0f || xr[0] == -1.0f || yu[0] == -1.0f || yd[0] == -1.0f)) {
            float xv0 = xr[0] - xl[0], xv1 = xr[1] - xl[1], xv2 = xr[2] - xl[2];
            float yv0 = yd[0] - yu[0], yv1 = yd[1] - yu[1], yv2 = yd[2] - yu[2];
#if NORM_FMA
#if NORM_FMA == 2
            float n0 = -__fmaf_rn(xv2, yv1, -(xv1 * yv2));
            float n1 = -__fmaf_rn(xv0, yv2, -(xv2 * yv0));
            float n2 = -__fmaf_rn(xv1, yv0, -(xv0 * yv1));
#else
            float n0 = __fmaf_rn(xv1, yv2, -(xv2 * yv1));
            float n1 = __fmaf_rn(xv2, yv0, -(xv0 * yv2));
            float n2 = __fmaf_rn(xv0, yv1, -(xv1 * yv0));
#endif
            float len2 = __fmaf_rn(n2, n2, __fmaf_rn(n1, n1, n0 * n0));
#else
            float n0 = xv1 * yv2 - xv2 * yv1;
            float n1 = xv2 * yv0 - xv0 * yv2;
            float n2 = xv0 * yv1 - xv1 * yv0;
            float len2 = n0 * n0 + n1 * n1 + n2 * n2;
#endif
            if (!(len2 == 0.0f || len2 != len2)) {
                float inv = 1.0f / sqrtf(len2);
                o0 = n0 * inv; o1 = n1 * inv; o2 = n2 * inv;
            }
        }
    }
    nc[(size_t)idx * 3 + 0] = o0;
    nc[(size_t)idx * 3 + 1] = o1;
    nc[(size_t)idx * 3 + 2] = o2;
}

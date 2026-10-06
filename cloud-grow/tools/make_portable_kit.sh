#!/usr/bin/env bash
# Build a self-contained, relocatable kit of VC3D tools.
#
# Why: a binary straight out of a build tree carries a DT_RPATH naming
# absolute paths on the machine that built it. If any of those paths is on a
# network mount that later stops responding, the dynamic loader blocks there
# BEFORE main() -- the process cannot start, cannot be killed, and shows up as
# a running job doing nothing. Measured on this fleet: every tool hung that way
# until the RPATH was removed.
#
# So the kit copies each tool with every non-system library it needs
# (resolved recursively), and launches through a wrapper that sets
# LD_LIBRARY_PATH to the kit's own lib directory. Nothing outside the kit is
# consulted, so it runs the same on any machine of the same architecture.
#
# Usage:
#   make_portable_kit.sh --build DIR --out DIR [tool ...]
#
# With no tools named, the pipeline stages are used.
# No `set -e` / `pipefail`: this loops over tools and libraries, and a
# single missing dependency must not abandon the whole kit.
set -u

BUILD=""; OUT=""; TOOLS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD="$2"; shift 2 ;;
        --out)   OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) TOOLS+=("$1"); shift ;;
    esac
done
[ -n "$BUILD" ] && [ -n "$OUT" ] || { echo "make_portable_kit: --build and --out are required" >&2; exit 2; }
# 2026-09-29 (upstream-tools, coordinator-requested): added the three post-grow QC tools
# (stages/upstream_qc.py, growth_guard.py's ARM B) to the default list -- vc_tifxyz_selfcross
# and vc_tifxyz_winding were ALREADY present on every real kit checked (phiNN/28/29/30) despite
# never being in this default, meaning every one of those kits was already hand-built with an
# explicit tool list wider than this default; vc_seg_add_overlap was the one QC tool that
# NEVER made it onto any real kit until this session's manual fix on phiNN (identification +
# a real run verified before trusting it -- see scripts/upstream_inputs/STATE.md). Making the
# default match what a kit actually needs to be means the next kit built from scratch (a fresh
# host, a rebuild) carries all four QC tools without anyone having to remember the extra names.
# 2026-09-29 (coordinator, general merge queue): added vc_merge_patch -- found missing from
# EVERY real kit checked this session (phiNN had 7 binaries total, none of them merge tools;
# the same gap this file's own comment above already documented for vc_seg_add_overlap before
# it was added). Live-verified on phiNN: the hub's own compiled binary, copied and chmod +x
# alongside the existing kit, ran correctly against that kit's ALREADY-PRESENT shared libs
# (`file` confirmed a real ELF, not a wrapper; `--help` and 3 real merges succeeded).
[ "${#TOOLS[@]}" -gt 0 ] || TOOLS=(vc_grow_seg_from_seed vc_flatten vc_render_tifxyz vc_gen_normalgrids
                                   vc_tifxyz_selfcross vc_tifxyz_winding vc_seg_add_overlap vc_merge_patch)

BINDIR="$BUILD/bin"
[ -d "$BINDIR" ] || { echo "make_portable_kit: no bin/ under $BUILD" >&2; exit 2; }

rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/lib"

# Resolve the full transitive closure of shared libraries. ldd already reports
# it transitively, but only for libraries it can find -- so run it from the
# build tree, where the tool's own RPATH still resolves its siblings.
collect_deps() {
    local t="$1"
    ldd "$t" 2>/dev/null | awk '/=> \//{print $3}'
}

copied=0
for t in "${TOOLS[@]}"; do
    src="$BINDIR/$t"
    [ -x "$src" ] || { echo "  skip $t (not built)"; continue; }
    cp -a "$src" "$OUT/bin/$t"
    while IFS= read -r so; do
        [ -n "$so" ] || continue
        # Leave the core system libraries to the host: glibc and its direct
        # companions must match the running kernel and loader, and bundling
        # them is how a kit breaks rather than how it travels.
        case "$so" in
            /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;;
        esac
        b="$(basename "$so")"
        [ -e "$OUT/lib/$b" ] && continue
        cp -aL "$so" "$OUT/lib/$b" && copied=$((copied+1))
    done < <(collect_deps "$src")
done

# Second pass: the bundled libraries have dependencies of their own.
for _ in 1 2 3 4 5; do
    added=0
    for so in "$OUT"/lib/*; do
        [ -e "$so" ] || continue
        while IFS= read -r dep; do
            [ -n "$dep" ] || continue
            case "$dep" in /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*) continue ;; esac
            b="$(basename "$dep")"
            [ -e "$OUT/lib/$b" ] && continue
            cp -aL "$dep" "$OUT/lib/$b" && added=$((added+1)) && copied=$((copied+1))
        done < <(collect_deps "$so")
    done
    [ "$added" -eq 0 ] && break
done

# Rewrite DT_RPATH on everything we copied, dropping any component that names
# an absolute path outside the kit.
#
# This is not cosmetic and the wrapper below cannot substitute for it:
# DT_RPATH is searched BEFORE LD_LIBRARY_PATH, so as long as a stale absolute
# path survives in the header, the loader consults it first -- and if it lives
# on an unresponsive mount, the process blocks there before main(). Verified on
# this fleet: a fully self-contained kit in tmpfs, launched through a wrapper,
# still hung on every tool until these entries were removed.
STRIP="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/lib/strip_rpath.py"
if [ -f "$STRIP" ]; then
    ABSOLUTE=1 APPLY=1 \
        python3 "$STRIP" "$OUT"/bin/* "$OUT"/lib/* >/dev/null 2>&1 || true
    # The stripper keeps a .rpath-bak beside each file it edits; a generated
    # kit has no use for them and they distort the audit below.
    find "$OUT" -name '*.rpath-bak' -delete
    left=0
    for f in "$OUT"/bin/* "$OUT"/lib/*; do
        if objdump -x "$f" 2>/dev/null | awk '/RPATH|RUNPATH/{print $2}' \
             | tr ':' '\n' | grep -q '^/'; then
            left=$((left+1))
            echo "  WARNING: $(basename "$f") still names an absolute rpath" >&2
        fi
    done
    echo "  rpath: $left artifact(s) still name an absolute path"
else
    echo "  WARNING: lib/strip_rpath.py not found; kit may hang on a dead mount" >&2
fi

# One wrapper per tool. The wrapper -- not a baked-in RPATH -- is what points
# at the kit's libraries, so the kit can be moved anywhere without patching
# any ELF header.
mkdir -p "$OUT/run"
for t in "$OUT"/bin/*; do
    n="$(basename "$t")"
    cat > "$OUT/run/$n" <<'WRAP'
#!/usr/bin/env bash
kit="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
export LD_LIBRARY_PATH="$kit/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# One OpenBLAS thread: these tools run many processes in parallel, and a
# nested BLAS pool oversubscribes every core.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$kit/bin/$(basename "${BASH_SOURCE[0]}")" "$@"
WRAP
    chmod +x "$OUT/run/$n"
done

echo "make_portable_kit: $(ls "$OUT/bin" | wc -l) tools, $copied libraries, $(du -sh "$OUT" | cut -f1)"
echo "  launch via $OUT/run/<tool>"

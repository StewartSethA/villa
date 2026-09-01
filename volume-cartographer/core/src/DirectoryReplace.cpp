#include "vc/core/util/DirectoryReplace.hpp"

#include <cerrno>
#include <chrono>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef __linux__
// renameat2(RENAME_EXCHANGE) needs the GNU prototypes. (g++ defines
// _GNU_SOURCE for every C++ translation unit, so this is belt-and-braces --
// but it must still precede the first libc header to have any effect.)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifndef RENAME_EXCHANGE
#include <linux/fs.h>
#endif
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)
#endif
#endif

namespace fs = std::filesystem;

namespace
{

#ifdef __linux__

// Issue renameat2(RENAME_EXCHANGE), preferring glibc's wrapper and falling
// back to the raw syscall.
//
// The wrapper arrived in glibc 2.28 (2018); the kernel has served the syscall
// since Linux 3.15 (2014). A build against an older sysroot -- the conda
// x86_64-conda-linux-gnu sysroot used for portable binaries is glibc 2.17 --
// therefore fails to link a call the running kernel would happily service.
// The syscall branch is the SAME kernel operation, so RENAME_EXCHANGE's
// atomicity holds identically on both: there is no silent downgrade to a
// non-atomic swap.
int renameExchange(const char* oldPath, const char* newPath) noexcept
{
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 28))
    return ::renameat2(AT_FDCWD, oldPath, AT_FDCWD, newPath, RENAME_EXCHANGE);
#elif defined(SYS_renameat2)
    return static_cast<int>(::syscall(SYS_renameat2, AT_FDCWD, oldPath,
                                      AT_FDCWD, newPath,
                                      static_cast<unsigned int>(RENAME_EXCHANGE)));
#else
    (void)oldPath;
    (void)newPath;
    errno = ENOSYS;
    return -1;
#endif
}

// Does this errno mean "this kernel or this filesystem cannot exchange at
// all", as opposed to "the exchange was attempted and genuinely failed"?
//
// Filesystems answer that in several dialects, and getting the set wrong
// turns an unsupported-flag report into a thrown exception on exactly the
// filesystems most likely to hold a shared segment directory:
//   ENOSYS      kernel older than 3.15
//   EINVAL      flag not recognised (NFS, CIFS, many FUSE filesystems)
//   EOPNOTSUPP  flag recognised but unimplemented for this superblock
//   ENOTSUP     spelling of the above on some libcs
//   EPERM       overlayfs and some FUSE filesystems refuse the flag this way
bool exchangeUnsupported(int err) noexcept
{
    return err == ENOSYS || err == EINVAL || err == EOPNOTSUPP
        || err == ENOTSUP || err == EPERM;
}

#endif  // __linux__

// Flush the directory entry itself, so a rename that has already returned is
// still there after a power loss. rename() is atomic with respect to
// concurrent readers the instant it returns, but that is a different
// guarantee from durability: without this, a save that reported success can
// still be lost. Best-effort by design -- a filesystem that will not hand out
// a directory fd is not a reason to fail a save that otherwise completed.
void fsyncDirectory(const fs::path& dir) noexcept
{
#ifdef __linux__
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#else
    (void)dir;
#endif
}

// Unique enough never to collide with a concurrent save of the same segment,
// and recognisable if a crash ever leaves one behind.
fs::path asideNameFor(const fs::path& target)
{
    fs::path aside = target;
    aside += ".replaced-";
#ifdef __linux__
    aside += std::to_string(static_cast<long long>(::getpid()));
    aside += "-";
#endif
    aside += std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return aside;
}

}  // namespace

DirReplaceMode replaceDirectory(const fs::path& staged, const fs::path& target,
                                DirReplaceStrategy strategy)
{
    const fs::path parent = target.parent_path();

#ifdef __linux__
    // Preferred path: RENAME_EXCHANGE swaps the two directory entries in a
    // single kernel step, so a concurrent reader of `target` sees either the
    // whole old tree or the whole new one -- never a mixture, never an
    // absence.
    if (strategy == DirReplaceStrategy::Auto) {
        if (renameExchange(staged.c_str(), target.c_str()) == 0) {
            // `staged` now names the OLD tree; drop it.
            std::error_code ec;
            fs::remove_all(staged, ec);
            if (ec) {
                throw std::runtime_error(
                    "failed to clean up previous data at " + staged.string()
                    + ": " + ec.message());
            }
            fsyncDirectory(parent);
            return DirReplaceMode::AtomicExchange;
        }
        const int err = errno;
        if (!exchangeUnsupported(err)) {
            const std::error_code ec(err, std::generic_category());
            throw std::runtime_error("atomic exchange failed for "
                + staged.string() + " and " + target.string() + ": "
                + ec.message());
        }
        // else: this filesystem has no exchange; fall through.
    }
#endif

    // Portable path, for filesystems without the flag -- NFS is the case that
    // matters in practice, and shared segment directories very often live
    // there. Move the old tree ASIDE first, then move the new tree in. Each
    // step is an ordinary same-directory rename and so is itself atomic.
    //
    // The ordering is the whole point. Doing remove_all(target) first and
    // renaming afterwards opens a window in which the old tree is already
    // destroyed and the new one is not yet in place; a crash or an error
    // inside that window leaves NO directory at `target` at all, losing a
    // segment that may represent hours of growth. Here the old tree survives
    // under the aside name for the entire operation, a failed second rename
    // is rolled back, and the aside copy is removed only once the new tree is
    // committed. The residual exposure is `target` being briefly absent --
    // recoverable -- rather than permanently empty.
    const fs::path aside = asideNameFor(target);

    std::error_code ec;
    fs::rename(target, aside, ec);
    if (ec) {
        throw std::runtime_error("could not move previous data "
            + target.string() + " aside to " + aside.string() + ": "
            + ec.message());
    }

    fs::rename(staged, target, ec);
    if (ec) {
        // Put the old tree back before reporting. A failed replacement must
        // never be the reason a segment goes missing.
        std::error_code rollbackEc;
        fs::rename(aside, target, rollbackEc);
        const std::string note = rollbackEc
            ? " (rollback ALSO failed; previous data is at " + aside.string() + ")"
            : " (previous data restored)";
        throw std::runtime_error("could not move new data into "
            + target.string() + ": " + ec.message() + note);
    }

    fsyncDirectory(parent);

    std::error_code cleanupEc;
    fs::remove_all(aside, cleanupEc);
    if (cleanupEc) {
        throw std::runtime_error("failed to clean up previous data at "
            + aside.string() + ": " + cleanupEc.message());
    }
    return DirReplaceMode::StagedSwap;
}

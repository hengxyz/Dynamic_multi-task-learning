#!/usr/bin/env python
import os
import sys
import ctypes
import struct
from ctypes.util import find_library

# struct linux_dirent64 {
#     ino64_t        d_ino;    /* 64-bit inode number */
#     off64_t        d_off;    /* 64-bit offset to next structure */
#     unsigned short d_reclen; /* Size of this dirent */
#     unsigned char  d_type;   /* File type */
#     char           d_name[]; /* Filename (null-terminated) */
# };


class linux_dirent64(ctypes.Structure):
    _fields_ = (
        ('d_ino', ctypes.c_uint64),
        ('d_off', ctypes.c_int64),
        ('d_reclen', ctypes.c_ushort),
        ('d_type', ctypes.c_ubyte),
        ('d_name', ctypes.c_char * 256),
    )


SYS_GETDENTS64 = 217
DEFAULT_BUF_SIZE = 1 * 1024 * 1024  # 1MB
D_NAME_OFFSET = \
    ctypes.sizeof(ctypes.c_uint64) + ctypes.sizeof(ctypes.c_int64) + \
    ctypes.sizeof(ctypes.c_ushort) + ctypes.sizeof(ctypes.c_ubyte)


c_linux_dirent64_p = ctypes.POINTER(linux_dirent64)
libc = ctypes.CDLL(find_library('c'), use_errno=True)
syscall = libc.syscall
syscall.argtypes = [ctypes.c_int, ctypes.c_uint,
                    c_linux_dirent64_p, ctypes.c_uint]
syscall.restype = ctypes.c_int


def handle_error(path):
    errno = ctypes.get_errno()
    err = OSError(errno, os.strerror(errno))
    err.filename = path
    raise err


def getdents64(path, buf=None, buf_size=DEFAULT_BUF_SIZE):
    """
    Yields a tuple: (d_ino, d_off, d_reclen, d_type, d_name)
    """
    fd = None
    nread = 0
    if not buf:
        buf = ctypes.create_string_buffer(buf_size)
    bufp = ctypes.cast(buf, c_linux_dirent64_p)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        while True:
            nread = syscall(SYS_GETDENTS64, fd, bufp, buf_size)
            if nread < 0:
                handle_error(path)
            if nread == 0:
                break
            pos = 0
            while pos < nread:
                d_name_start_pos = pos + D_NAME_OFFSET
                d_ino, d_off, d_reclen, d_type = \
                    struct.unpack('QqHB', buf.raw[pos:d_name_start_pos])
                d_name_end_pos = pos + d_reclen
                d_name_raw = buf.raw[d_name_start_pos:d_name_end_pos]
                d_name = ctypes.create_string_buffer(d_name_raw).value
                pos += d_reclen
                yield (d_ino, d_off, d_reclen, d_type, d_name)
    finally:
        if fd:
            os.close(fd)


if __name__ == '__main__':
    path_arg = '.' if len(sys.argv) == 1 else sys.argv[1]
    # Example usage
    for (d_ino, d_off, d_reclen, d_type, d_name) in getdents64(path_arg):
        print(d_type, d_name)
# rutentiny - Ada <sarahsooup@protonmail.com> CC0-1.0
# vim: et:ts=4:sw=4
import gzip
from struct import pack, unpack, calcsize
from typing import NamedTuple


class Byte(NamedTuple):
    key: str | None
    val: int

class Short(NamedTuple):
    key: str | None
    val: int

class Int(NamedTuple):
    key: str | None
    val: int

class Long(NamedTuple):
    key: str | None
    val: int

class Float(NamedTuple):
    key: str | None
    val: float

class Double(NamedTuple):
    key: str | None
    val: float

class Binary(NamedTuple):
    key: str | None
    val: bytes

class String(NamedTuple):
    key: str | None
    val: str

class List(NamedTuple):
    key: str | None
    val: list
    type: int

class Compound(NamedTuple):
    key: str | None
    val: list


def _runpack(s, st):
    buf = unpack(st, s.read(calcsize(st)))

    if len(buf) < 2:
        return buf[0]
    else:
        return buf


def _reads(s) -> str:
    return s.read(_runpack(s, "!h")).decode()


def _read_list(s, length: int, type: int) -> list:
    buf = []

    # why does nbt have so much recursion
    for i in range(length):
        match type:
            case 1:
                buf.append(Byte(None, _runpack(s, "b")))
            case 2:
                buf.append(Short(None, _runpack(s, "!h")))
            case 3:
                buf.append(Int(None, _runpack(s, "!i")))
            case 4:
                buf.append(Long(None, _runpack(s, "!q")))
            case 5:
                buf.append(Float(None, _runpack(s, "!f")))
            case 6:
                buf.append(Double(None, _runpack(s, "!d")))
            case 7:
                buf.append(Binary(None, s.read(_runpack(s, "!i"))))
            case 8:
                buf.append(Binary(None, _reads(s)))
            case 9:
                mytype = _runpack(s, "b")
                mylen = _runpack(s, "!i")
                myarr = _read_list(s, mylen, mytype)
                buf.append(List(None, myarr, mytype))
            case 10:
                this = Compound(None, [])
                while not _read_nbt(s, this.val):
                    pass
                buf.append(this)

    return buf

def _read_nbt(s, l: list) -> bool:
    try:
        ntype = _runpack(s, "b")
    except:
        return True

    match ntype:
        case 0: # End
            return True

        case 1: # Byte
            l.append(Byte(_reads(s), _runpack(s, "b")))

        case 2: # Short
            l.append(Short(_reads(s), _runpack(s, "!h")))

        case 3: # Int
            l.append(Int(_reads(s), _runpack(s, "!i")))

        case 4: # Long
            l.append(Long(_reads(s), _runpack(s, "!q")))

        case 5: # Float
            l.append(Float(_reads(s), _runpack(s, "!f")))

        case 6: # Double
            l.append(Double(_reads(s), _runpack(s, "!d")))

        case 7: # Binary
            l.append(Binary(_reads(s), s.read(_runpack(s, "!i"))))

        case 8: # String
            l.append(String(_reads(s), _reads(s)))

        case 9: # List
            mykey = _reads(s)
            mytype = _runpack(s, "b")
            mylen = _runpack(s, "!i")
            myarr = _read_list(s, mylen, mytype)
            l.append(List(mykey, myarr, mytype))

        case 10: # Compound
            l.append(_read_compound(s))

        case _:
            raise Exception(f"Unknown NBT type {ntype}")

    return False

def _read_compound(s) -> Compound:
    this = Compound(_reads(s), [])

    while not _read_nbt(s, this.val):
        pass

    return this


def open(path: str) -> Compound:
    import builtins

    def is_gzip():
        with builtins.open(path, "rb") as f:
            return f.read(2) == b'\x1f\x8b'

    # try to open as gzipped and fall back to uncompressed
    if is_gzip():
        file = gzip.open(path, "rb")
    else:
        file = builtins.open(path, "rb")

    with file:
        if (tag := _runpack(file, "b")) != 10:
            raise Exception(f"First tag is {tag}, 10 expected")

        return _read_compound(file)

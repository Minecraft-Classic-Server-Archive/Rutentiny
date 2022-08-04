#!/usr/bin/env python3
# rutentiny - Ada <sarahsooup@protonmail.com> CC0-1.0
# vim: et:ts=4:sw=4
import sys

if sys.version_info < (3, 10):
    print("Rutentiny requires Python 3.10 or later")
    exit(1)

import zlib
import gzip
import time
import typing
import threading
import socket
import socketserver
import textwrap
from enum import IntEnum
from struct import pack, unpack, calcsize
from typing import NamedTuple, Optional, Iterator

try:
    import opensimplex
    simplex_available = True
except:
    print("opensimplex unavailable, falling back to bad noise algorithm")


def pad(msg: str) -> bytes:
    return msg.ljust(64).encode("cp437")


msg_pad_wrapper = textwrap.TextWrapper(
        width=64,
        subsequent_indent="  ",
        expand_tabs=False,
        tabsize=1,
        replace_whitespace=False)
def pad_msg(msg: str) -> list[bytes]:
    from textwrap import wrap
    buf = []
    for line in msg_pad_wrapper.wrap(msg):
        buf.append(pad(line))
    return buf


def pad_data(data: bytes) -> bytes:
    trim = data[:1024]
    return trim + (b"\0" * (1024 - len(trim)))


def chunk_iter(data: bytes) -> Iterator[bytes]:
    return (data[pos:pos + 1024] for pos in range(0, len(data), 1024))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] " + msg)


def clamp(v: float, a: float, b: float) -> float:
    return max(min(v, b), a)


class Vec3(NamedTuple):
    x: int
    y: int
    z: int

    def __str__(self) -> str:
        return f"{self.x, self.y, self.z}"

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_fixed(x: float, y: float, z: float) -> "Vec3":
        return Vec3(
                int(clamp(x * 32, -32768, 32767)),
                int(clamp((y * 32) + 51, -32768, 32767)),
                int(clamp(z * 32, -32768, 32767)))

    def dist_to(self, other: "Vec3") -> float:
        from math import sqrt
        return sqrt(
            ((self.x - other.x) ** 2)
            + ((self.y - other.y) ** 2)
            + ((self.z - other.z) ** 2))


class Vec2(NamedTuple):
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x, self.y}"


class BlockID(IntEnum):
    NONE = 0
    STONE = 1
    GRASS = 2
    DIRT = 3
    COBBLESTONE = 4
    PLANKS = 5
    SAPLING = 6
    BEDROCK = 7
    WATER = 8
    WATER_STILL = 9
    LAVA = 10
    LAVA_STILL = 11
    SAND = 12
    GRAVEL = 13
    GOLD_ORE = 14
    IRON_ORE = 15
    COAL_ORE = 16
    LOG = 17
    LEAVES = 18
    SPONGE = 19
    GLASS = 20
    CLOTH_RED = 21
    CLOTH_ORANGE = 22
    CLOTH_YELLOW = 23
    CLOTH_LIME = 24
    CLOTH_GREEN = 25
    CLOTH_TEAL = 26
    CLOTH_AQUA = 27
    CLOTH_CYAN = 28
    CLOTH_BLUE = 29
    CLOTH_INDIGO = 30
    CLOTH_VIOLET = 31
    CLOTH_MAGENTA = 32
    CLOTH_PINK = 33
    CLOTH_BLACK = 34
    CLOTH_GRAY = 35
    CLOTH_WHITE = 36
    DANDELION = 37
    ROSE = 38
    MUSHROOM_BROWN = 39
    MUSHROOM_RED = 40
    GOLD = 41
    IRON = 42
    SMOOTH_STONE = 43
    SMOOTH_STONE_SLAB = 44
    BRICKS = 45
    TNT = 46
    BOOKSHELF = 47
    MOSSY_COBBLESTONE = 48
    OBSIDIAN = 49

    # cpe level 1 blocks
    COBBLESTONE_SLAB = 50,
    ROPE = 51,
    SANDSTONE = 52,
    SNOW = 53,
    FIRE = 54,
    CLOTH_LIGHTPINK = 55,
    CLOTH_FOREST = 56,
    CLOTH_BROWN = 57,
    CLOTH_NAVY = 58,
    CLOTH_DARKTEAL = 59,
    ICE = 60,
    TILES = 61,
    MAGMA = 62,
    PILLAR = 63,
    CRATE = 64,
    STONE_BRICKS = 65,

    # example blocks to replace still water and lava
    TAR = 9,
    ACID = 11,
    JUMP_PAD = 66,

    def cpe_fallback(block: "BlockID") -> "BlockID":
        match block:
            case BlockID.COBBLESTONE_SLAB: return BlockID.SMOOTH_STONE_SLAB
            case BlockID.ROPE: return BlockID.MUSHROOM_BROWN
            case BlockID.SANDSTONE: return BlockID.SAND
            case BlockID.SNOW: return BlockID.NONE
            case BlockID.FIRE: return BlockID.LAVA
            case BlockID.CLOTH_LIGHTPINK: return BlockID.CLOTH_PINK
            case BlockID.CLOTH_FOREST: return BlockID.CLOTH_GREEN
            case BlockID.CLOTH_BROWN: return BlockID.DIRT
            case BlockID.CLOTH_NAVY: return BlockID.CLOTH_BLUE
            case BlockID.CLOTH_DARKTEAL: return BlockID.CLOTH_CYAN
            case BlockID.ICE: return BlockID.GLASS
            case BlockID.TILES: return BlockID.IRON
            case BlockID.MAGMA: return BlockID.OBSIDIAN
            case BlockID.PILLAR: return BlockID.CLOTH_WHITE
            case BlockID.CRATE: return BlockID.PLANKS
            case BlockID.STONE_BRICKS: return BlockID.STONE
            case BlockID.ACID: return BlockID.LAVA_STILL
            case BlockID.TAR: return BlockID.WATER_STILL
            case BlockID.JUMP_PAD: return BlockID.TNT
            case _: return block


class Client:
    """
    Represents the player state of a connected client.
    """

    def __init__(self, socket, server: "ServerState"):
        self.socket = socket
        self.server: "ServerState" = server
        self.name: str = "unnamed"
        self.key: str = ""
        self.pos: Vec3 = Vec3(0, 0, 0)
        self.angle: Vec2 = Vec2(0, 0)
        self.old_pos: Vec3 = self.pos
        self.old_angle: Vec2 = self.angle
        self.health: int = 20
        self.air: int = -1
        self.armor: int = -1
        self.gamemode: int = 1
        self.ticks: int = 0
        self.uses_cpe: bool = False
        self.cpe_exts: list[tuple[str, int]] = []
        self.oper: bool = False
        self.disconnected: bool = False
        self.msg_buffer: bytearray = bytearray()
        self.dmg_queue: list[int] = []
        self.held_block: int = 0
        self.model: str = "humanoid"

    def recv(self, size: int) -> bytes:
        try:
            buffer = bytearray()
            remain = size

            while remain > 0:
                tb = self.socket.recv(remain)

                if len(tb) < 1:
                    raise EOFError

                remain -= len(tb)
                buffer += tb

            return buffer
        except:
            self.disconnected = True
            return b""

    def send(self, data: bytes) -> None:
        if self.disconnected: return
        try:
            self.socket.sendall(data)
        except:
            self.disconnected = True

    def tick(self):
        if self.disconnected: return
        self.ticks += 1

        # ping every 5 seconds
        if self.ticks % (20 * 5):
            self.send(pack("B", 1))

        if self.health < 1:
            self.server.spawn_effect(1,
                    self.pos - Vec3(0, 24, 0),
                    Vec3(0, 0, 0))

            self.health = 20
            self.armor = -1
            self.air = -1
            self.dmg_queue.clear()
            self.health_update()
            spawn_pos = self.server.level.get_spawnpoint()
            self.spawn(spawn_pos[0], spawn_pos[1])

            if self.gamemode == 0:
                self.message("&cYou died!", 100)

        x, y, z = self.pos
        feet_block = self.server.level.get_block(x//32, (y-52)//32, z//32)

        if feet_block == BlockID.JUMP_PAD \
                and ("VelocityControl", 1) in self.cpe_exts:
            self.send(pack("!BiiiBBB", 47, 0, 50000, 0, 0, 1, 0))

        if self.gamemode == 1 or self.gamemode == 2:
            return

        old_health = self.health
        old_armor = self.armor
        old_air = self.air
        head_block = self.server.level.get_block(x//32, y//32, z//32)
        body_block = self.server.level.get_block(x//32, (y-51)//32, z//32)

        for h in self.dmg_queue:
            self.health -= h
        self.dmg_queue.clear()

        if head_block == BlockID.WATER or head_block == BlockID.TAR:
            if self.air < 0:
                self.air = 21

            if (self.ticks % 10) == 0:
                if self.air > 0:
                    self.air -= 1
                elif (self.ticks % 20) == 0:
                    self.health -= 2
        else:
            self.air = -1

        if (head_block == BlockID.LAVA
                or body_block == BlockID.LAVA):
            if (self.ticks % 10) == 0:
                self.health -= 4
        elif (head_block == BlockID.ACID
                or body_block == BlockID.ACID):
            if (self.ticks % 10) == 0:
                self.health -= 2
        elif (head_block == BlockID.FIRE
                or body_block == BlockID.FIRE):
            if (self.ticks % 10) == 0:
                self.health -= 2

        self.health = max(self.health, 0)

        if old_health != self.health \
                or old_air != self.air \
                or old_armor != self.armor:
            self.health_update()

    def packet(self):
        try:
            op, = unpack("B", self.socket.recv(1))
        except:
            self.disconnected = True
            return

        match op:
            # login
            case 0:
                if not self.server.level or not self.server.level.ready:
                    self.kick("Level is not ready yet")
                    return

                try:
                    ver, name, key, pad = unpack("B64s64sB",
                            self.recv(130))
                    self.name = name.strip().decode("cp437")
                    self.key = key.strip().decode("ascii")
                except:
                    log(f"client sent bad login packet")
                    self.kick("Corrupted login packet")
                    return

                if len(self.server.clients) >= self.server.max_clients:
                    self.kick("Server is full")
                    return

                if self.server.key:
                    if self.server.key != self.key:
                        self.kick("Invalid passkey")
                        return
                elif self.server.config.get("heartbeat_url", None):
                    from hashlib import md5
                    if key != md5(self.server.key.encode() + name):
                        self.kick("Invalid login key")
                        return

                if pad == 0x42:
                    self.uses_cpe = True
                    self.server.cpe_handshake(self)

                self.server.send_id(self)
                self.server.level.send_to_client(self)
                spawnpos = self.server.level.get_spawnpoint()
                self.spawn(spawnpos[0], spawnpos[1])
                self.set_gamemode(self.server.level.gamemode)
                self.server.add_client(self)

            # block set
            case 5:
                x, y, z, mode, block = unpack("!hhhBB", self.recv(8))

                # only survival and creative allow block placing
                if self.gamemode != 0 and self.gamemode != 1:
                    self.send(pack("!BhhhB", 6, x, y, z,
                        self.server.level.get_block(x, y, z)))
                else:
                    if mode == 0:
                        self.server.set_block(self, Vec3(x, y, z), BlockID.NONE)
                    else:
                        self.server.set_block(self, Vec3(x, y, z), block)

            # position update
            case 8:
                old_feet = self.server.level.get_block(
                        self.pos.x // 32,
                        (self.pos.y - 52) // 32,
                        self.pos.z // 32)
                old_pos = self.pos

                block, x, y, z, yaw, pitch = unpack("!BhhhBB", self.recv(9))
                self.old_pos = self.pos
                self.old_angle = self.angle
                self.pos = Vec3(x, y, z)
                self.angle = Vec2(pitch, yaw)

                if block != self.held_block \
                        and (self.model == "hold" or self.model == "humanoid"):
                    if block != BlockID.NONE:
                        self.server.set_playermodel(self, "hold", 1000 + block)
                    else:
                        self.server.set_playermodel(self, "humanoid", 1000)

                new_feet = self.server.level.get_block(
                        self.pos.x // 32,
                        (self.pos.y - 52) // 32,
                        self.pos.z // 32)

                if old_feet != BlockID.WATER and new_feet == BlockID.WATER:
                    self.server.spawn_effect(2,
                            self.pos - Vec3(0, 51, 0),
                            Vec3(0, 100, 0))

                self.held_block = block

            # message
            case 13:
                import re
                ext, msg = unpack("B64s", self.recv(65))
                self.msg_buffer += msg

                if ext == 0:
                    msg = self.msg_buffer.strip().decode("cp437")
                    msg = re.sub(r'%([0-9a-f])', r'&\1', msg)
                    self.msg_buffer = bytearray()
                    self.server.handle_message(self, msg)

            # click
            case 34:
                from math import tau, sin, cos
                data = unpack("!BBhhBhhhB", self.recv(14))

                if data[4] > 127:
                    return

                try:
                    target = self.server.clients[data[4]]
                except:
                    return

                block = Vec3(data[5], data[6], data[7])

                # FIXME: classicube doesnt check for line-of-sight so we
                # will have to do it instead at some point
                if self.gamemode == 4:
                    if data[0] == 0 and data[1] == 0:
                        target.dmg_queue.append(200)
                        self.server.message(f"&c{self.name} fragged {target.name}")
                elif self.gamemode == 0:
                    dist = self.pos.dist_to(target.pos)
                    yaw = ((self.angle.y / 255) * tau) + (tau / 4)
                    punch = (int(cos(yaw) * -5000), int(sin(yaw) * -5000))
                    if data[0] == 0 and data[1] == 0 and dist < 96:
                        target.dmg_queue.append(1)

                        if ("VelocityControl", 1) in target.cpe_exts:
                            target.send(pack("!BiiiBBB", 47,
                                punch[0], 5000, punch[1], 1, 1, 1))

            case _:
                log(f"{self.name}: Unknown opcode {hex(op)}")

    def health_update(self) -> None:
        if ("RutentoyGamemode", 1) in self.cpe_exts:
            self.send(pack("Bbbbx", 0xa1, self.health,
                self.armor, self.air))
        elif ("MessageTypes", 1) in self.cpe_exts:
            if self.gamemode != 1:
                self.message(f"&cHealth: {self.health}", 1)
            else:
                self.message(" ", 1)

            if self.armor > -1:
                self.message(f"&7Armor: {self.armor}", 2)
            else:
                self.message(" ", 2)

            if self.air > -1 and self.air < 21:
                self.message(f"&bAir: {self.air}", 3)
            else:
                self.message(" ", 3)

    def set_gamemode(self, mode: str | int) -> None:
        ruten = ("RutentoyGamemode", 1) in self.cpe_exts

        self.health = 20

        speed = 1.0
        hax = "-hax"

        match mode:
            case "survival" | "0" | 0:
                if ruten:
                    self.send(pack("BBB", 0xa0, 0, 1))
                self.gamemode = 0

            case "creative" | "1" | 1:
                if ruten:
                    self.send(pack("BBB", 0xa0, 1, 1))
                self.gamemode = 1
                hax = "+hax"

            case "explore" | "2" | 2:
                if ruten:
                    self.send(pack("BBB", 0xa0, 1, 0))
                self.gamemode = 2

            case "instagib" | "4" | 4:
                if ruten:
                    self.send(pack("BBB", 0xa0, 0, 0))
                self.gamemode = 4
                speed = 2.0

            case _:
                self.message("&cInvalid gamemode")
                return

        self.health_update()

        if ("InstantMOTD", 1) in self.cpe_exts:
            oper = 0
            if self.oper:
                oper = 100

            self.send(pack("BB64s64sB", 0, 7,
                pad(self.server.config.get("name", "Rutentiny")),
                pad(f"{hax} horspeed={speed}"), oper))

    # NOTE: expects a fixed-point Vec3, unlike teleport
    def spawn(self, pos: Vec3, angle: Vec2) -> None:
        self.send(pack("!BB64shhhBB", 7, 255, pad(self.name),
            pos.x, pos.y, pos.z, angle.y, angle.x))

    def teleport(self, pos: Vec3, angle: Vec2) -> None:
        self.pos = pos
        self.angle = angle
        fpos = Vec3.to_fixed(pos.x + .5, pos.y, pos.z + .5)
        self.send(pack("!BBhhhBB", 8, 255,
            fpos.x, fpos.y, fpos.z,
            angle.y, angle.x))

    def kick(self, reason: str) -> None:
        self.send(b"\x0e" + pad(reason))
        self.server.remove_client(self)
        self.disconnected = True

    def message(self, msg: str, type: int = 0) -> None:
        # NOTE: wiki.vg doesn't document all of CC's capabilities,
        # maybe 'big announcement' and 'small announcement' aren't standard CPE
        # 0: normal chat
        # 1: top-right 1
        # 2: top-right 2
        # 3: top-right 3
        # 11: bottom-right 1
        # 12: bottom-right 2
        # 13: bottom-right 3
        # 100: announcement
        # 101: big announcement
        # 102: announcement subtitle
        if ("MessageTypes", 1) not in self.cpe_exts:
            type = 0

        chunks = pad_msg(msg)
        for chunk in chunks:
            self.send(pack("BB64s", 13, type, chunk))


class Level:
    """
    Stores block data and provides functions for disk and network IO.
    """

    def __init__(self, server: "ServerState", size: Optional[Vec3] = None):
        self.xs: int
        self.ys: int
        self.zs: int
        self.seed: int
        self.map: bytearray
        self.clouds: tuple[int, int]
        self.colors: list[tuple[int, int, int]]
        self.server: "ServerState" = server

        if size is None:
            self.xs, self.ys, self.zs = self.server.config.get("map_size",
                    (128, 128, 128))
        else:
            self.xs, self.ys, self.zs = size

        self.spawn_points = []
        self.gamemode = "creative"

        self.map = bytearray(self.xs * self.ys * self.zs)
        self.edge = ((BlockID.WATER, BlockID.BEDROCK), (self.ys // 2, -2))
        self.clouds = (self.ys + 2, 256)
        self.colors = [
                (0x63, 0x9b, 0xff),	# sky
                (0xff, 0xff, 0xff), # clouds
                (0xcb, 0xdb, 0xfc), # fog
                (0x84, 0x7e, 0x87), # block ambient
                (0xff, 0xff, 0xff), # block diffuse
                (0xff, 0xff, 0xff), # skybox tint
                ]
        self.seed = int(time.time())
        if simplex_available:
            opensimplex.seed(self.seed)

    def save(self, path: str) -> None:
        log(f"saving map {path}")
        with gzip.open(path, "wb") as file:
            file.write(b"rtm\x01")
            file.write(pack("!iii", self.xs, self.ys, self.zs))
            file.write(self.map)

            # FIXME: sometimes this just breaks and corrupts the save
            try:
                file.write(pack("!q", self.seed))
            except:
                file.write(pack("!q", 0))
            file.write(pack("!BBii", self.edge[0][0], self.edge[0][1],
                self.edge[1][0], self.edge[1][1]))
            file.write(pack("!ii", self.clouds[0], self.clouds[1]))

            for i in range(0, 6):
                file.write(pack("!BBB", self.colors[i][0], self.colors[i][1],
                    self.colors[i][2]))

            file.write(pack("!i", len(self.spawn_points)))
            for point in self.spawn_points:
                file.write(pack("!iiiBB", point[0].x, point[0].y, point[0].z,
                    point[1].x, point[1].y))

            file.write(pad(self.gamemode))

    def load(self, path: str) -> None:
        self.ready = False
        log(f"loading map {path}")
        with gzip.open(path, "rb") as file:
            magic = file.read(4)

            # old map format
            if magic == b"rtm\x00":
                self.xs, self.ys, self.zs, self.seed = unpack("!iiiq",
                        file.read(calcsize("!iiiq")))
                self.edge = (unpack("BB", file.read(2)),
                        (unpack("!ii", file.read(calcsize("!ii")))))
                self.clouds = unpack("!ii", file.read(calcsize("!ii")))

                for i in range(0, 6):
                    self.colors.append(unpack("!BBB", file.read(3)))

                self.map = bytearray(file.read(self.xs * self.ys * self.zs))
            elif magic == b"rtm\x01":
                # put the size and map data before everything else so future
                # extensions to the rtm1 format can be ignored for back compat
                self.xs, self.ys, self.zs = unpack("!iii", file.read(12))
                self.map = bytearray(file.read(self.xs * self.ys * self.zs))

                try:
                    self.seed = unpack("!q", file.read(8))
                except:
                    self.seed = 0
                self.edge = (unpack("BB", file.read(2)),
                        unpack("!ii", file.read(8)))
                self.clouds = unpack("!ii", file.read(8))

                self.colors.clear()
                for i in range(0, 6):
                    self.colors.append(unpack("!BBB", file.read(3)))

                spawnpoint_count, = unpack("!i", file.read(4))

                if spawnpoint_count > 0:
                    self.spawn_points.clear()

                    for i in range(0, spawnpoint_count):
                        self.spawn_points.append((
                            Vec3(*unpack("!iii", file.read(12))),
                            Vec2(*unpack("BB", file.read(2)))))

                self.gamemode = file.read(64).strip().decode("cp437")
            else:
                log(f"Invalid map format {repr(magic)}")
                raise ValueError(f"Invalid map format {repr(magic)}")

        self.ready = True

    def get_spawnpoint(self) -> tuple[Vec3, Vec2]:
        from random import choice

        if len(self.spawn_points) < 1:
            return (Vec3.to_fixed(
                (self.xs//2),
                (self.ys+1),
                (self.zs//2)),
                    Vec2(0, 0))
        else:
            return choice(self.spawn_points)

    def _index(self, x: int, y: int, z: int) -> int:
        return (y * self.zs + z) * self.xs + x

    def _set_block(self, x: int, y:int, z: int, block: BlockID) -> None:
        i = self._index(x, y, z)
        if i < 0 or i >= len(self.map):
            return

        self.map[i] = block

    def set_block(self, x: int, y: int, z: int, block: BlockID) -> None:
        self._set_block(x, y, z, block)

        if self.ready:
            for cl in self.server.clients:
                cl.send(pack("!BhhhB", 6, x, y, z, block))

    def get_block(self, x: int, y: int, z: int) -> BlockID:
        i = self._index(x, y, z)
        if i < 0 or i >= len(self.map):
            return BlockID.NONE

        return typing.cast(BlockID, self.map[i])

    def noise(self, x: float, y: float) -> float:
        if simplex_available:
            t = opensimplex.noise2(x + 34.332, y + 4.444) * 1.0
            #t += opensimplex.noise2((x * 2) + 43.112, (y * 2) + 3.143) * 0.1
            t += opensimplex.noise2((x * 5) + 0.145, (y * 5) + 88.432) * 0.1
            return t
        else:
            # TODO: make a cc0 noise function that isnt trashgarbage
            # terrible awful fallback noise
            from random import random, seed
            from math import trunc

            def lerp(a, b, t):
                return a + (b - a) * t

            t = (x - trunc(x)) ** 2

            seed(int(x + self.seed))
            a = random()
            seed(int(x + 1) + self.seed)
            b = random()

            out = lerp(a, b, t)
            return (out * 2) - 1

    def tree(self, rx: int, ry: int, rz: int) -> None:
        from random import randrange
        height = randrange(4, 8)

        y = 1
        for x in range(-2, 3):
            for z in range(-2, 3):
                self.set_block(x+rx, y+ry+(height-3), z+rz, BlockID.LEAVES)
        y = 2
        for x in range(-2, 3):
            for z in range(-2, 3):
                self.set_block(x+rx, y+ry+(height-3), z+rz, BlockID.LEAVES)
        y = 3
        for x in range(-1, 2):
            for z in range(-1, 2):
                self.set_block(x+rx, y+ry+(height-3), z+rz, BlockID.LEAVES)

        for y in range(0, height):
            self.set_block(rx, ry + y, rz, BlockID.LOG)

    def boulder(self, rx: int, ry: int, rz: int) -> None:
        from random import randrange, uniform
        radius = randrange(4, 16)
        xr = round(radius * uniform(0.8, 3.0))
        yr = round(radius * uniform(0.8, 3.0))
        zr = round(radius * uniform(0.8, 3.0))
        xrs = xr**2
        yrs = yr**2
        zrs = zr**2
        trs = xr*yr*zr

        for x in range(-xr, xr):
            for z in range(-zr, zr):
                for y in range(-yr, yr):
                    if (((x**2)*xrs) + ((y**2)*yrs) + ((z**2)*zrs)) <= trs:
                        self.set_block(rx+x, ry+y, rz+z, BlockID.STONE)

    def generate_hills(self):
        from random import randrange

        self.ready = False
        log("Generating Hills map")
        self.edge = ((BlockID.WATER, BlockID.BEDROCK), (self.ys // 2, -2))

        for z in range(0, self.zs):
            for x in range(0, self.xs):
                height = max(0, self.noise(x / 128, z / 128) * 32)
                height += self.noise(x / 64, z / 64) * 4

                sink = abs(self.noise(x/256, z/256))
                spikes = self.noise(-x/256, -z/256)

                riverswirl = self.noise(z/32, x/32) * 2
                river = self.noise((z+riverswirl)/256, (x+riverswirl)/512)

                if height < 8 and sink > 0.3 and sink < 0.6:
                    height -= 4

                if river < 0.05 and river > -0.05:
                    height = -6

                height += max(0, (spikes - 0.5)) * 128

                height = int(height + 3 + (self.ys//2))

                if height < 0: height = 0
                if height > self.ys: height = self.ys - (height - self.ys)

                for y in range(0, self.ys//2):
                    self.set_block(x, y, z, BlockID.WATER)

                for y in range(0, height - 3):
                    self.set_block(x, y, z, BlockID.STONE)

                for y in range(height - 3, height - 1):
                    self.set_block(x, y, z, BlockID.DIRT)

                if height-1 < (self.ys//2):
                    self.set_block(x, height-1, z, BlockID.SAND)
                else:
                    self.set_block(x, height-1, z, BlockID.GRASS)

                if height > self.ys//2 and height < self.ys-16:
                    n = self.noise(x/64, z/64)

                    if n > 0.3:
                        if randrange(0, 64) == 0:
                            self.tree(x, height, z)

                    if n < -0.3:
                        if randrange(0, 1024) == 0:
                            self.boulder(x, height + randrange(-4, 0), z)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_islands(self):
        self.ready = False
        log("Generating Islands map")
        self.edge = ((BlockID.WATER, BlockID.SAND), (self.ys // 2, -2))

        for z in range(0, self.zs):
            for x in range(0, self.xs):
                height = abs(self.noise(x / 96, z / 96) * 16)
                height += self.noise(x / 24, z / 24) * 2
                height -= 8
                if height < 0 or height > 8: height *= .3
                height = int(height + (self.ys//2))

                if height < 0: height = 0
                if height > self.ys: height = self.ys

                for y in range(0, self.ys//2):
                    self.set_block(x, y, z, BlockID.WATER)

                for y in range(0, height - 3):
                    self.set_block(x, y, z, BlockID.STONE)
                for y in range(height - 3, height - 1):
                    self.set_block(x, y, z, BlockID.DIRT)

                if height-1 < (self.ys//2):
                    self.set_block(x, height-1, z, BlockID.SAND)
                else:
                    self.set_block(x, height-1, z, BlockID.GRASS)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_hell(self):
        from random import randrange
        self.ready = False
        log("Generating Hell map")
        self.edge = ((BlockID.LAVA, BlockID.DIRT), (self.ys // 2, -2))
        self.colors = [
                (0x84, 0x1b, 0x1b), # sky
                (0x72, 0x17, 0x17), # clouds
                (0x84, 0x1b, 0x1b), # fog
                (0x12, 0x04, 0x04), # block ambient
                (0x5f, 0x1f, 0x1f), # block diffuse
                (0xff, 0xff, 0xff), # skybox tint
                ]

        for z in range(0, self.zs):
            for x in range(0, self.xs):
                height = abs(self.noise(x / 96, z / 96) * 24)
                height += self.noise(x / 24, z / 24) * 2
                height -= 8
                height = int(height + (self.ys//2))

                if height < 0: height = 0
                if height > self.ys: height = self.ys

                for y in range(0, self.ys//2):
                    self.set_block(x, y, z, BlockID.LAVA)

                for y in range(0, height - 3):
                    self.set_block(x, y, z, BlockID.STONE)
                for y in range(height - 3, height - 1):
                    self.set_block(x, y, z, BlockID.DIRT)

                self.set_block(x, height-1, z, BlockID.DIRT)

                if height > self.ys//2:
                    if self.noise(x/64, z/64) > 0.3:
                        if randrange(0, 64) == 0:
                            self.tree(x, height, z)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_moon(self):
        from random import randrange
        self.ready = False
        log("Generating Moon map")
        self.edge = ((BlockID.GRAVEL, BlockID.BEDROCK), (self.ys // 2, 0))
        self.colors = [
                (0x00, 0x00, 0x00), # sky
                (0x00, 0x00, 0x00), # clouds
                (0x00, 0x00, 0x00), # fog
                (0x5f, 0x5f, 0x5f), # block ambient
                (0xff, 0xff, 0xff), # block diffuse
                (0xff, 0xff, 0xff), # skybox tint
                ]

        for z in range(0, self.zs):
            for x in range(0, self.xs):
                height = abs(self.noise(x / 96, z / 96) * 8)
                height += self.noise(x / 24, z / 24) * 2
                height -= max(16, self.noise(x/32, z/32) * 32)

                spikes = self.noise(-x/256, -z/256)
                height += max(0, (spikes - 0.5)) * 128

                height = int(height + 18 + (self.ys//2))

                if height < 0: height = 0
                if height > self.ys: height = self.ys - (height - self.ys)

                for y in range(0, height - 3):
                    self.set_block(x, y, z, BlockID.STONE)
                for y in range(height - 3, height - 1):
                    self.set_block(x, y, z, BlockID.GRAVEL)

                if height > self.ys//2:
                    if randrange(0, 1024) == 0:
                        self.boulder(x, height + randrange(-4, 0), z)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_flatgrass(self):
        self.ready = False
        log("Generating Flat Grass map")
        self.edge = ((BlockID.GRASS, BlockID.BEDROCK), (self.ys // 2, 0))
        height = self.ys//2
        for z in range(0, self.zs):
            for x in range(0, self.xs):
                for y in range(0, height - 3):
                    self.set_block(x, y, z, BlockID.STONE)
                for y in range(height - 3, height - 1):
                    self.set_block(x, y, z, BlockID.DIRT)

                if z == 0 or z == self.zs-1 or x == 0 or x == self.xs-1:
                    self.set_block(x, height-1, z, BlockID.CLOTH_LIME)
                else:
                    self.set_block(x, height-1, z, BlockID.GRASS)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_stone_checkers(self):
        self.ready = False
        log("Generating Stone Checkers map")
        self.edge = ((BlockID.STONE, BlockID.BEDROCK), (self.ys // 2, 0))
        for z in range(0, self.zs):
            for x in range(0, self.xs):
                xd = int(x / 16) % 2 == 0
                zd = int(z / 16) % 2 == 0
                if (xd and not zd) or (not xd and zd):
                    block = BlockID.SMOOTH_STONE
                else:
                    block = BlockID.STONE
                for y in range(0, self.ys // 2):
                    self.set_block(x, y, z, block)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True

    def generate_void(self):
        self.ready = False
        log("Generating Void map")
        self.edge = ((BlockID.NONE, BlockID.NONE), (0, 0))
        self.map = bytearray(self.xs * self.ys * self.zs)
        self.set_block(self.xs//2-1, self.ys//2-1, self.zs//2-1, BlockID.STONE)
        self.set_block(self.xs//2, self.ys//2-1, self.zs//2-1, BlockID.STONE)
        self.set_block(self.xs//2-1, self.ys//2-1, self.zs//2, BlockID.STONE)
        self.set_block(self.xs//2, self.ys//2-1, self.zs//2, BlockID.STONE)
        log("Map generation done")
        self.ready = True

    def send_to_client(self, c: Client) -> None:
        fast = ("FastMap", 1) in c.cpe_exts

        if fast:
            c.send(pack("!Bi", 2, len(self.map)))
        else:
            c.send(pack("B", 2))


        if ("BlockDefinitions", 1) in c.cpe_exts:
            c.send(pack("BB64sBBBBBBBBBBBBBB",
                35,             # DefineBlock packet
                BlockID.ACID,   # block id
                b"Acid",        # name
                5,              # collision mode
                96,             # speed modifier
                46, 46, 46,     # top, side, bottom textures
                0,              # translucent
                0,              # walk noise
                1,              # fullbright
                16,             # voxel height
                3,              # transparency mode
                191,            # fog density
                # fog rgb
                0x99, 0xe5, 0x50))

            c.send(pack("BB64sBBBBBBBBBBBBBB",
                35,             # DefineBlock packet
                BlockID.TAR,    # block id
                b"Tar",         # name
                6,              # collision mode
                15,             # speed modifier
                63, 63, 63,     # top, side, bottom textures
                0,              # translucent
                0,              # walk noise
                0,              # fullbright
                16,             # voxel height
                0,              # transparency mode
                255,            # fog density
                # fog rgb
                0x00, 0x00, 0x00))

            c.send(pack("BB64sBBBBBBBBBBBBBB",
                35,                 # DefineBlock packet
                BlockID.JUMP_PAD,   # block id
                b"Jump Pad",        # name
                2,                  # collision mode
                128,                # speed modifier
                10, 39, 55,         # top, side, bottom textures
                0,                  # translucent
                5,                  # walk noise
                0,                  # fullbright
                16,                 # voxel height
                0,                  # transparency mode
                0,                  # fog density
                # fog rgb
                0x00, 0x00, 0x00))

        if ("EnvMapAspect", 1) in c.cpe_exts:
            # edge blocks
            c.send(pack("!BBi", 41, 0, self.edge[0][1]))
            c.send(pack("!BBi", 41, 9, self.edge[1][1]))
            c.send(pack("!BBi", 41, 1, self.edge[0][0]))
            c.send(pack("!BBi", 41, 2, self.edge[1][0]))

            # clouds
            c.send(pack("!BBi", 41, 3, self.clouds[0]))
            c.send(pack("!BBi", 41, 5, self.clouds[1]))

        if fast:
            m = zlib.compressobj(wbits=-15)
            cb = m.compress(self.map) + m.flush(zlib.Z_FINISH)
            for b in chunk_iter(cb):
                c.send(pack("!BH", 3, len(b)) + pad_data(b) + b"\0")
        else:
            for b in chunk_iter(gzip.compress(pack("!i", len(self.map)) + self.map)):
                c.send(pack("!BH", 3, len(b)) + pad_data(b) + b"\0")

        if ("EnvColors", 1) in c.cpe_exts:
            for i in range(0, 5):
                col = self.colors[i]
                c.send(pack("!BBhhh", 25, i, col[0], col[1], col[2]))

        c.send(pack("!Bhhh", 4, self.xs, self.ys, self.zs))

        # allow any blocks because we dont have physics simulation
        if ("BlockPermissions", 1) in c.cpe_exts:
            c.send(pack("BBBB", 28, 7, 1, 1))
            c.send(pack("BBBB", 28, 8, 1, 1))
            c.send(pack("BBBB", 28, 9, 1, 1))
            c.send(pack("BBBB", 28, 10, 1, 1))
            c.send(pack("BBBB", 28, 11, 1, 1))

        if ("CustomParticles", 1) in c.cpe_exts:
            # cloud puff particle
            c.send(pack("!BBBBBBBBBBBBiHiiiiBB",
                48,                 # DefineEffect
                1,                  # EffectID
                0, 0, 15, 15,       # UV
                0xFF, 0xFF, 0xFF,   # Color
                8,                  # FrameCount
                16,                 # ParticleCount
                24,                 # Size (/32)
                5000,               # SizeVariation
                32,                 # Spread
                1000,               # Speed (/10k)
                -1000,              # Gravity (/10k)
                5000,               # Lifetime (/10k)
                5000,               # LifetimeVariation (/10k)
                0b01110000,         # CollisionFlags
                0                   # FullBright
                ))

            # water splash particle
            c.send(pack("!BBBBBBBBBBBBiHiiiiBB",
                48,                 # DefineEffect
                2,                  # EffectID
                0, 16, 15, 31,      # UV
                0xFF, 0xFF, 0xFF,   # Color
                2,                  # FrameCount
                6,                  # ParticleCount
                8,                  # Size (/32)
                5000,               # SizeVariation
                20,                 # Spread (/32)
                12000,              # Speed (/10k)
                60000,              # Gravity (/10k)
                5000,               # Lifetime (/10k)
                5000,               # LifetimeVariation (/10k)
                0b01110000,         # CollisionFlags
                0                   # FullBright
                ))


class ServerState:
    def __init__(self):
        self.level: Level = None
        self.clients: list[Client] = []
        self.config: dict = {}
        self.alive: bool = True
        self.key: str = ""
        self.max_clients: int = 8

    def load_config(self, path: str) -> None:
        from ast import literal_eval
        with open(path, "r", encoding="utf-8") as f:
            lineno = 0
            for line in f:
                line = line.strip()
                lineno += 1

                if len(line) < 1 or line[0] == "#":
                    continue

                if "=" not in line:
                    log(f"{path}:{lineno} is missing '=', ignoring")
                    continue

                key, _, value = line.partition("=")
                key = key.strip().lower()
                value = value.strip()

                if value == "":
                    value = "None"

                try:
                    self.config[key] = literal_eval(value)
                except:
                    log(f"config error on line {lineno}")

        self.max_clients = self.config.get("max_clients", 8)

        if key := self.config.get("key", ""):
            self.key = key
        elif self.config.get("heartbeat_url", None):
            from secrets import choice
            base62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            for i in range(0, 16):
                self.key += choice(base62)

    def add_client(self, c: Client) -> None:
        if c not in self.clients:
            self.clients.append(c)
            self.message(f"{c.name} joined")

        for cl in self.clients:
            if c != cl:
                c.send(pack("!BB64shhhBB", 7, self.clients.index(cl),
                    pad(cl.name), cl.pos.x, cl.pos.y, cl.pos.z,
                    cl.angle.y, cl.angle.x))

                cl.send(pack("!BB64shhhBB", 7, self.clients.index(c),
                    pad(c.name), c.pos.x, c.pos.y, c.pos.z,
                    c.angle.y, c.angle.x))

    def remove_client(self, c: Client) -> None:
        if c not in self.clients:
            return

        self.message(f"{c.name} left")
        i = self.clients.index(c)

        for c in self.clients:
            c.send(pack("BB", 12, i))

        self.clients.remove(c)

    def shutdown(self, reason: str) -> None:
        log("Server shutting down")

        if self.config.get("map_shutdown_autosave", False):
            path = f"autosave-{int(time.time())}.rtm"
            self.level.save(self.config.get("map_path", ".") + "/" + path)

        for c in self.clients:
            c.send(b"\x0e" + pad(reason))
            c.disconnected = True

        self.clients = []
        self.alive = False

    def send_id(self, c: Client) -> None:
        oper = 0

        if c.name in self.config.get("opers", []):
            c.oper = True
            oper = 100

        c.send(pack("BB64s64sB", 0, 7,
            pad(self.config.get("name", "Rutentiny")),
            pad(self.config.get("motd", "My Cool Server")), oper))

    def cpe_handshake(self, c: Client) -> None:
        c.send(pack("!B64sH", 16, b"Rutentiny 0.1.0", 11))
        c.send(pack("!B64sI", 17, b"RutentoyGamemode", 1))
        c.send(pack("!B64sI", 17, b"CustomBlocks", 1))
        c.send(pack("!B64sI", 17, b"EnvMapAspect", 1))
        c.send(pack("!B64sI", 17, b"HackControl", 1))
        c.send(pack("!B64sI", 17, b"FullCP437", 1))
        c.send(pack("!B64sI", 17, b"LongerMessages", 1))
        c.send(pack("!B64sI", 17, b"MessageTypes", 1))
        c.send(pack("!B64sI", 17, b"FastMap", 1))
        c.send(pack("!B64sI", 17, b"PlayerClick", 1))
        c.send(pack("!B64sI", 17, b"HeldBlock", 1))
        c.send(pack("!B64sI", 17, b"CustomParticles", 1))
        cname, extnum = unpack("!x64sH", c.recv(67))

        for i in range(0, extnum):
            extname, extver = unpack("!x64sI", c.recv(69))
            c.cpe_exts.append((extname.strip().decode("ascii"), extver))

        # TODO: send fallback blocks to clients that dont support this
        if ("CustomBlocks", 1) in c.cpe_exts:
            c.send(pack("BB", 19, 1))
            unpack("xB", c.recv(2))

    def load_map(self, path: str) -> None:
        self.level = Level(self)
        self.level.load(path)

        for c in self.clients:
            self.level.send_to_client(c)

            spawnpos = self.level.get_spawnpoint()
            c.spawn(spawnpos[0], spawnpos[1])
            c.set_gamemode(self.level.gamemode)
            self.add_client(c)

    def new_map(self, type: str, size: Vec3) -> None:
        self.level = Level(self, size)

        match type:
            case "stone_checkers":
                self.level.generate_stone_checkers()

            case "islands":
                self.level.generate_islands()

            case "hills":
                self.level.generate_hills()

            case "hell":
                self.level.generate_hell()

            case "moon":
                self.level.generate_moon()

            case "flatgrass":
                self.level.generate_flatgrass()

            case "void":
                self.level.generate_void()

            case _:
                log(f"Unknown generator {state.config.get('map_generator')}")
                self.level.generate_flatgrass()

        for c in self.clients:
            self.level.send_to_client(c)

            spawnpos = self.level.get_spawnpoint()
            c.spawn(spawnpos[0], spawnpos[1])
            c.set_gamemode(self.level.gamemode)
            self.add_client(c)

    def tick(self) -> None:
        for c in self.clients:
            if c.pos != c.old_pos or c.angle != c.old_angle:
                self.player_move(c)
            c.tick()
        timer = threading.Timer(1 / 20, self.tick)
        timer.daemon = True
        timer.start()

    def player_move(self, c: Client) -> None:
        i = self.clients.index(c)

        for cl in self.clients:
            if c == cl: continue
            cl.send(pack("!BBhhhBB", 8, i, c.pos.x, c.pos.y, c.pos.z,
                c.angle.y, c.angle.x))

    def set_playermodel(self, c: Client, model: str, scale: int = 1000) -> None:
        i = self.clients.index(c)
        models = pad(model)

        c.model = model

        for cl in self.clients:
            if ("ChangeModel", 1) not in c.cpe_exts:
                continue

            # ChangeModel
            if c == cl:
                cl.send(pack("!Bb64s", 29, -1, models))
            else:
                cl.send(pack("!Bb64s", 29, i, models))

            if ("EntityProperty", 1) not in c.cpe_exts:
                continue

            # SetEntityProperty
            # NOTE: CC supports property 3, 4, and 5 all for setting
            # the model scale; wiki.vg doesn't document this
            if c == cl:
                cl.send(pack("!BbBi", 42, -1, 3, scale))
            else:
                cl.send(pack("!BbBi", 42, i, 3, scale))

    def spawn_effect(self, id: int, pos: Vec3, vel: Vec3) -> None:
        for cl in self.clients:
            if ("CustomParticles", 1) not in cl.cpe_exts:
                continue

            # SpawnEffect
            cl.send(pack("!BBiiiiii", 49, id,
                pos.x, pos.y, pos.z,
                # NOTE: these are a world-space target, not a velocity
                pos.x - vel.x, pos.y - vel.y, pos.z - vel.z))

    def set_block(self, c: Client, pos: Vec3, block: BlockID) -> None:
        self.level._set_block(pos.x, pos.y, pos.z, block)

        for cl in self.clients:
            cl.send(pack("!BhhhB", 6, pos.x, pos.y, pos.z, block))

    def message(self, msg: str, type: int = 0) -> None:
        log(msg)
        chunks = pad_msg(msg)

        # don't use Client.message so we only have to pad_msg once
        for c in self.clients:
            for chunk in chunks:
                c.send(pack("BB64s", 13,
                    type if ("MessageTypes", 1) in c.cpe_exts else 0,
                    chunk))

    def sys_command(self, args: list[str]) -> None:
        if len(args) < 1:
            return

        def warn(msg: str) -> None:
            from sys import stderr
            print(msg, file=stderr)

        if args[0][0] == "/":
            args[0] = args[0][1:]

        match args[0]:
            case "say":
                if len(args) < 2: return
                self.message(" ".join(args[1:]))

            case "stop":
                if len(args) < 2:
                    self.shutdown("Server shutting down")
                else:
                    self.shutdown(" ".join(args[1:]))

            case "kick":
                if len(args) < 2:
                    warn("/kick needs at least one argument")
                    return

                # FIXME: this is ugly and slow
                for c in self.clients:
                    if c.name == args[1]:
                        if len(args) < 3:
                            c.kick("Kicked")
                        else:
                            c.kick("Kicked: " + " ".join(args[2:]))

            case "load":
                if len(args) < 2 or len(args) > 2:
                    warn("/load takes one argument")
                    return

                path = f"{args[1]}.rtm"
                try:
                    self.load_map(
                            self.config.get("map_path", ".") + "/" + path)
                    self.message(f"Loaded {path}")
                except:
                    self.message(f"Failed to load {path}")

            case "save":
                if len(args) < 2 or len(args) > 2:
                    warn("/save takes one argument")
                    return

                path = f"{args[1]}.rtm"
                try:
                    self.level.save(
                            self.config.get("map_path", ".") + "/" + path)
                    self.message(f"Saved as {path}")
                except:
                    self.message(f"Failed to save {path}")

            case _:
                warn(f"Unknown command {args[0]}")

    def handle_message(self, c: Client, msg: str) -> None:
        if msg[0] != "/":
            self.message(f"{c.name}: {msg}")
            return

        if msg.startswith("/me "):
            self.message(f"{c.name} {msg[4:]}")
        elif msg.startswith("/gamemode ") and c.oper:
            c.set_gamemode(msg[10:].strip())
            c.message(f"&eYour gamemode is set to {msg[10:].strip()}")
        elif msg.startswith("/load ") and c.oper:
            path = msg[6:].strip() + ".rtm"

            if "./" in path:
                c.message(f"&cInvalid path")
                return

            try:
                self.load_map(self.config.get("map_path", ".") + "/" + path)
                self.message(f"Loaded {path}")
            except Exception as e:
                c.message(f"&cFailed to load {path}: {e!r}")
        elif msg.startswith("/save ") and c.oper:
            path = msg[6:].strip() + ".rtm"

            if "./" in path:
                c.message(f"&cInvalid path")
                return

            try:
                self.level.save(self.config.get("map_path", ".") + "/" + path)
                self.message(f"Saved as {path}")
            except Exception as e:
                self.message(f"Failed to save {path}: {e!r}")
        elif msg.startswith("/new ") and c.oper:
            args = msg[5:].split(" ")
            if len(args) < 4:
                c.message("&cUsage: /new generator xsize ysize zsize")
                return

            if (int(args[1]) > 1024
                    or int(args[2]) > 1024
                    or int(args[3]) > 1024):
                c.message("&cMap extents cannot exceed 1024 in any direction")
                return

            self.message(f"Generating new {args[0]} map")
            self.new_map(args[0],
                    Vec3(int(args[1]), int(args[2]), int(args[3])))
            self.message(f"Done generating {args[0]} map")
        elif msg.startswith("/tp ") and (c.gamemode == 1 or c.oper):
            args = msg[4:].split(" ")

            try:
                x, y, z = int(args[0]), int(args[1]), int(args[2])
                c.teleport(Vec3(x, y, z), c.angle)
            except:
                c.message("&cUsage: /tp x y z")
        elif msg.startswith("/add-spawn") and (c.gamemode == 1 or c.oper):
            self.level.spawn_points.append((c.pos, c.angle))
            c.message(f"Added new spawn point at {c.pos}, {c.angle}")
        elif msg.startswith("/level-gamemode ") and c.oper:
            mode = msg[15:].strip()
            self.level.gamemode = mode
            c.message(f"Default gamemode set to {mode}")
        elif msg.startswith("/tree") and c.gamemode == 1:
            self.level.tree(c.pos.x//32, c.pos.y//32-1, c.pos.z//32)
        elif msg.startswith("/model "):
            self.set_playermodel(c, msg[7:])
        else:
            c.message("&cUnknown command")


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.client: Client = Client(self.request, self.server.state)

        while not self.client.disconnected:
            self.client.packet()
        self.server.state.remove_client(self.client)


class ThreadedServer(socketserver.ThreadingTCPServer):
    def __init__(self, address: tuple[str, int],
            request_handler: typing.Any, state: ServerState):
        from ipaddress import ip_address, IPv6Address

        if len(address[0]) > 0 \
                    and not isinstance(ip_address(address[0]), IPv6Address):
                self.address_family = socket.AF_INET
        else:
                self.address_family = socket.AF_INET6

        self.state: ServerState = state
        self.allow_reuse_address = True
        super().__init__(address, request_handler)


class HeartBeater:
    def __init__(self, state: ServerState):
        self.state = state

    def start_beating(self):
        import urllib.parse as parse
        import urllib.request as request
        from time import sleep
        cfg = self.state.config.get

        url = cfg("heartbeat_url") + "?"
        url += f"port={cfg('listen_ip', (0, 25565))[1]}&"
        url += f"max={cfg('max_clients', 8)}&"
        url += f"name={parse.quote(cfg('name', 'Rutentoy'))}&"
        url += f"public=False&"
        url += f"version=7&"
        url += f"salt={parse.quote(self.state.key)}&"
        url += f"users={len(self.state.clients)}"

        while True:
            try:
                req = request.urlopen(url)
                req.close()
            except:
                break

            sleep(1)


if __name__ == "__main__":
    state = ServerState()
    if len(sys.argv) > 1:
        state.load_config(sys.argv[1])

    ip = state.config.get("listen_ip", ("::1", 25565))

    if state.config.get("heartbeat_url", "").startswith("http"):
        heart = HeartBeater(state)
        heart_thread = threading.Thread(target=heart.start_beating)
        heart_thread.daemon = True
        heart_thread.start()

    with ThreadedServer(ip, RequestHandler, state) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        if (map := state.config.get("default_map")):
            state.load_map(
                    state.config.get("map_path", ".") + "/" + map + ".rtm")
        else:
            state.new_map(state.config.get("map_generator", "flatgrass"),
                    state.config.get("map_size", (128, 128, 128)))

        state.tick()

        try:
            while state.alive:
                state.sys_command(input().split())
        except (KeyboardInterrupt, EOFError):
            state.shutdown("Server shutting down")
        except:
            state.shutdown("Possible server crash")

        server.shutdown()

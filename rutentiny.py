#!/usr/bin/env python3
# rutentiny - Ada <sarahsooup@protonmail.com> CC0-1.0
# vim: et:ts=4:sw=4

try:
    match None:
        case _: pass
except:
    print("Rutentiny requires Python 3.10 or later")
    exit(1)

import gzip
import time
import typing
import threading
import socket
import socketserver
from enum import IntEnum
from gzip import compress, decompress
from struct import pack, unpack, calcsize
from typing import NamedTuple, Optional, Iterator

try:
    import opensimplex
    simplex_available = True
except:
    print("opensimplex unavailable, falling back to bad noise algorithm")


def pad(msg: str) -> bytes:
    return msg.ljust(64).encode("ascii")


def pad_data(data: bytes) -> bytes:
    trim = data[:1024]
    return trim + (b"\0" * (1024 - len(trim)))


def chunk_iter(data: bytes) -> Iterator[bytes]:
    return (data[pos:pos + 1024] for pos in range(0, len(data), 1024))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] " + msg)


# Most of the time these are used as int vectors anyway
class Vec3(NamedTuple):
    x: int
    y: int
    z: int


class Vec2(NamedTuple):
    x: int
    y: int


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
        self.air: int = 11
        self.gamemode: int = 1
        self.ticks: int = 0
        self.uses_cpe: bool = False
        self.cpe_exts: list[tuple[str, int]] = []
        self.oper: bool = False
        self.disconnected: bool = False

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

        if self.gamemode != 0: return

        old_health = self.health
        old_air = self.air
        x, y, z = self.pos
        head_block = self.server.level.get_block(x//32, y//32, z//32)
        body_block = self.server.level.get_block(x//32, (y-51)//32, z//32)
        feet_block = self.server.level.get_block(x//32, (y-52)//32, z//32)

        if head_block == BlockID.WATER:
            if (self.ticks % 20) == 0:
                if self.air > 0:
                    self.air -= 1
                else:
                    self.health -= 2
        elif (head_block == BlockID.LAVA
                or body_block == BlockID.LAVA):
            if (self.ticks % 10) == 0:
                self.health -= 4
        else:
            self.air = 11

        self.health = max(self.health, 0)
        self.air = max(self.air, 0)

        if old_health != self.health or old_air != self.air:
            air = 0 if self.air > 10 else self.air
            self.send(pack("BBBBx", 0xa1, self.health, 0, air))


    def packet(self):
        try:
            op, = unpack("B", self.socket.recv(1))
        except:
            self.disconnected = True
            return

        match op:
            case 0:
                if not self.server.level or not self.server.level.ready:
                    self.kick("Level is not ready yet")
                    return

                # TODO: figure out why this chokes occasionally
                try:
                    ver, name, key, pad = unpack("B64s64sB",
                            self.recv(130))
                    self.name = name.strip().decode("ascii")
                except:
                    log(f"client sent bad login packet")
                    self.kick("Invalid login packet")
                    return

                if (skey := self.server.config.get("key")):
                    if skey != key.strip().decode("ascii"):
                        self.kick("Invalid password or key")
                        return

                self.key = key.strip().decode("ascii")

                if pad == 0x42:
                    self.uses_cpe = True
                    self.server.cpe_handshake(self)

                self.server.send_id(self)
                self.server.level.send_to_client(self)
                self.spawn(Vec3(self.server.level.xs//2,
                    self.server.level.ys+1, self.server.level.zs//2),
                    Vec2(0, 0))
                self.teleport(Vec3(self.server.level.xs//2,
                    self.server.level.ys+1, self.server.level.zs//2),
                    Vec2(0, 0))
                self.server.add_client(self)

                if ("InventoryOrder", 1) in self.cpe_exts:
                    self.send(pack("BBB", 44, 0, BlockID.WATER_STILL))
                    self.send(pack("BBB", 44, 0, BlockID.LAVA_STILL))

            case 5:
                x, y, z, mode, block = unpack("!hhhBB", self.recv(8))

                if mode == 0:
                    self.server.set_block(self, Vec3(x, y, z), BlockID.NONE)
                else:
                    self.server.set_block(self, Vec3(x, y, z), block)

            case 8:
                x, y, z, pitch, yaw = unpack("!xhhhBB", self.recv(9))
                self.old_pos = self.pos
                self.old_angle = self.angle
                self.pos = Vec3(x, y, z)
                self.angle = Vec2(pitch, yaw)

            case 13:
                (msg,) = unpack("x64s", self.recv(65))
                msg = msg.strip().decode()
                self.server.message(self, msg)

            case _:
                log(f"{self.name}: Unknown opcode {hex(op)}")

    def set_gamemode(self, mode: str | int) -> None:
        if ("RutentoyGamemode", 1) not in self.cpe_exts:
            self.message("&cYour client doesn't support RutentoyGamemode.")
            return

        match mode:
            case "survival" | "0" | 0:
                self.send(pack("BBBBB", 0xa0, 0, 1, 20, 20))
                self.send(pack("BBBBx", 0xa1, 20, 0, 0))
                self.gamemode = 0

            case "creative" | "1" | 1:
                self.send(pack("BBBBB", 0xa0, 1, 1, 20, 20))
                self.send(pack("BBBBx", 0xa1, 20, 0, 0))
                self.gamemode = 1

            case "explore" | "2" | 2:
                self.send(pack("BBBBB", 0xa0, 1, 0, 20, 20))
                self.send(pack("BBBBx", 0xa1, 20, 0, 0))
                self.gamemode = 2

            case _:
                self.message("&cValid gamemodes are: survival, explore, creative")
                return

        self.message(f"&eYour gamemode is set to {mode}")

    def spawn(self, pos: Vec3, angle: Vec2) -> None:
        self.send(pack("!BB64shhhBB", 7, 255, pad(self.name),
            pos.x * 32, pos.y * 32 + 51, pos.z * 32, angle.x, angle.y))

    def teleport(self, pos: Vec3, angle: Vec2) -> None:
        self.pos = pos
        self.angle = angle
        self.send(pack("!BBhhhBB", 8, 255,
            pos.x * 32, pos.y * 32 + 51, pos.z * 32, angle.x, angle.y))

    def kick(self, reason: str) -> None:
        self.send(b"\x0e" + pad(reason))
        self.server.remove_client(self)
        self.disconnected = True

    def message(self, msg: str) -> None:
        self.send(pack("BB64s", 13, 0, pad(msg)))


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

        self.map = bytearray(self.xs * self.ys * self.zs)
        self.edge = ((BlockID.WATER, BlockID.BEDROCK), (self.ys // 2, -2))
        self.clouds = (self.ys + 2, 256)
        self.colors = [
                (0x63, 0x9b, 0xff),	# sky
                (0xcb, 0xdb, 0xfc), # clouds
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
            file.write(b"rtm\0")
            file.write(pack("!iiiq", self.xs, self.ys, self.zs, self.seed))
            file.write(pack("!BBii", self.edge[0][0], self.edge[0][1],
                self.edge[1][0], self.edge[1][1]))
            file.write(pack("!ii", self.clouds[0], self.clouds[1]))

            for i in range(0, 6):
                file.write(pack("!BBB", self.colors[i][0], self.colors[i][1],
                    self.colors[i][2]))

            file.write(self.map)

    def load(self, path: str) -> None:
        self.ready = False
        log(f"loading map {path}")
        with gzip.open(path, "rb") as file:
            magic = file.read(4)

            if magic != b"rtm\0":
                log(f"Invalid map format {repr(magic)}")
                raise ValueError(f"Invalid map format {repr(magic)}")

            self.xs, self.ys, self.zs, self.seed = unpack("!iiiq",
                    file.read(calcsize("!iiiq")))
            self.edge = (unpack("BB", file.read(2)),
                    (unpack("!ii", file.read(calcsize("!ii")))))
            self.clouds = unpack("!ii", file.read(calcsize("!ii")))

            for i in range(0, 6):
                self.colors.append(unpack("!BBB", file.read(3)))

            self.map = bytearray(file.read(self.xs * self.ys * self.zs))

        self.ready = True

    def _index(self, x: int, y: int, z: int) -> int:
        return (y * self.zs + z) * self.xs + x

    def set_block(self, x: int, y: int, z: int, block: BlockID) -> None:
        i = self._index(x, y, z)
        if i < 0 or i >= len(self.map):
            return

        self.map[i] = block

    def get_block(self, x: int, y: int, z: int) -> BlockID:
        i = self._index(x, y, z)
        if i < 0 or i >= len(self.map):
            return BlockID.NONE

        return typing.cast(BlockID, self.map[i])

    def noise(self, x: float, y: float) -> float:
        if simplex_available:
            return opensimplex.noise2(x, y)
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

    def generate_islands(self):
        from math import sqrt
        self.ready = False
        log("Generating Islands map")
        for z in range(0, self.zs):
            for x in range(0, self.xs):
                self.edge = ((BlockID.WATER, BlockID.SAND), (self.ys // 2, -2))

                height = self.noise(x / 96, z / 96) * 16
                height += self.noise(x / 24, z / 24) * 2
                height -= 6
                if height < 0: height *= .3
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

            if z % 16 == 0:
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

            if z % 16 == 0:
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

            if z % 16 == 0:
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

    def send_to_client(self, c: Client):
        c.socket.send(pack("B", 2))

        if ("EnvMapAspect", 1) in c.cpe_exts:
            # edge blocks
            c.socket.send(pack("!BBi", 41, 0, self.edge[0][1]))
            c.socket.send(pack("!BBi", 41, 9, self.edge[1][1]))
            c.socket.send(pack("!BBi", 41, 1, self.edge[0][0]))
            c.socket.send(pack("!BBi", 41, 2, self.edge[1][0]))

            # clouds
            c.socket.send(pack("!BBi", 41, 3, self.clouds[0]))
            c.socket.send(pack("!BBi", 41, 5, self.clouds[1]))

        for b in chunk_iter(compress(pack("!i", len(self.map)) + self.map)):
            c.socket.send(pack("!BH", 3, len(b)) + pad_data(b) + b"\0")

        if ("EnvColors", 1) in c.cpe_exts:
            for i in range(0, 5):
                col = self.colors[i]
                c.socket.send(pack("!BBhhh", 25, i, col[0], col[1], col[2]))

        c.socket.send(pack("!Bhhh", 4, self.xs, self.ys, self.zs))

        # allow any blocks because we dont have physics simulation
        if ("BlockPermissions", 1) in c.cpe_exts:
            c.send(pack("BBBB", 28, 7, 1, 1))
            c.send(pack("BBBB", 28, 8, 1, 1))
            c.send(pack("BBBB", 28, 9, 1, 1))
            c.send(pack("BBBB", 28, 10, 1, 1))
            c.send(pack("BBBB", 28, 11, 1, 1))


class ServerState:
    def __init__(self):
        self.level: Level = None
        self.clients: list[Client] = []
        self.config: dict = {}
        self.alive: bool = True

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

    def add_client(self, c: Client) -> None:
        if c not in self.clients:
            self.clients.append(c)
            self.system_message(f"{c.name} joined")

        for cl in self.clients:
            if c != cl:
                c.send(pack("!BB64shhhBB", 7, self.clients.index(cl),
                    pad(cl.name), cl.pos.x, cl.pos.y, cl.pos.z,
                    cl.angle.x, cl.angle.y))

                cl.send(pack("!BB64shhhBB", 7, self.clients.index(c),
                    pad(c.name), c.pos.x, c.pos.y, c.pos.z,
                    c.angle.x, c.angle.y))

    def remove_client(self, c: Client) -> None:
        if c not in self.clients:
            return

        self.system_message(f"{c.name} left")
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
        c.send(pack("!B64sH", 16, b"Rutentiny 0.1.0", 5))
        c.send(pack("!B64sI", 17, b"RutentoyGamemode", 1))
        c.send(pack("!B64sI", 17, b"CustomBlocks", 1))
        c.send(pack("!B64sI", 17, b"EnvMapAspect", 1))
        c.send(pack("!B64sI", 17, b"EmoteFix", 1))
        c.send(pack("!B64sI", 17, b"HackControl", 1))
        cname, extnum = unpack("!x64sH", c.recv(67))

        for i in range(0, extnum):
            extname, extver = unpack("!x64sI", c.recv(69))
            c.cpe_exts.append((extname.strip().decode(), extver))

        # TODO: send fallback blocks to clients that dont support this
        if ("CustomBlocks", 1) in c.cpe_exts:
            c.send(pack("BB", 19, 1))
            unpack("xB", c.recv(2))

    def load_map(self, path: str) -> None:
        for c in self.clients:
            c.send(pack("BB", 12, 255))

        self.level = Level(self)
        self.level.load(path)

        for c in self.clients:
            self.level.send_to_client(c)

        for c in self.clients:
            c.spawn(Vec3(self.level.xs//2, self.level.ys+2, self.level.zs//2),
                    Vec2(0, 0))
            self.add_client(c)

    def new_map(self, type: str, size: Vec3) -> None:
        for c in self.clients:
            c.send(pack("BB", 12, 255))

        self.level = Level(self, size)

        match type:
            case "stone_checkers":
                self.level.generate_stone_checkers()

            case "islands":
                self.level.generate_islands()

            case "flatgrass":
                self.level.generate_flatgrass()

            case "void":
                self.level.generate_void()

            case _:
                log(f"Unknown generator {state.config.get('map_generator')}")
                self.level.generate_flatgrass()

        for c in self.clients:
            self.level.send_to_client(c)

        for c in self.clients:
            c.spawn(Vec3(self.level.xs//2, self.level.ys+2, self.level.zs//2),
                    Vec2(0, 0))
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
                c.angle.x, c.angle.y))


    def set_block(self, c: Client, pos: Vec3, block: BlockID) -> None:
        match block:
            case BlockID.WATER_STILL:
                block = BlockID.WATER
            case BlockID.LAVA_STILL:
                block = BlockID.LAVA

        self.level.set_block(pos.x, pos.y, pos.z, block)

        for cl in self.clients:
            cl.send(pack("!BhhhB", 6, pos.x, pos.y, pos.z, block))

    def player_message(self, c: Client, msg: str) -> None:
        ms = f"{c.name}: {msg}"
        mb = pad(ms)

        log(ms)
        for c in self.clients:
            c.send(pack("Bx64s", 13, mb))

    def system_message(self, msg: str) -> None:
        log(msg)
        for c in self.clients:
            c.send(pack("BB64s", 13, 255, pad(msg)))

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
                self.system_message(" ".join(args[1:]))

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
                    self.system_message(f"Loading {path}")
                    self.load_map(
                            self.config.get("map_path", ".") + "/" + path)
                    self.system_message(f"Loaded {path}")
                except:
                    self.system_message(f"Failed to load {path}")

            case "save":
                if len(args) < 2 or len(args) > 2:
                    warn("/load takes one argument")
                    return

                path = f"{args[1]}.rtm"
                try:
                    self.level.save(
                            self.config.get("map_path", ".") + "/" + path)
                    self.system_message(f"Saved as {path}")
                except:
                    self.system_message(f"Failed to save {path}")

            case _:
                warn(f"Unknown command {args[0]}")

    def message(self, c: Client, msg: str) -> None:
        if msg[0] != "/":
            self.player_message(c, msg)
            return

        name = c.name

        if msg.startswith("/me "):
            self.system_message(f"{name} {msg[4:]}")
        elif msg.startswith("/gamemode "):
            c.set_gamemode(msg[10:].strip())
        elif msg.startswith("/load ") and c.oper:
            path = msg[6:].strip() + ".rtm"

            if "./" in path:
                c.message(f"&cInvalid path")
                return

            try:
                self.system_message(f"Loading {path}")
                self.load_map(self.config.get("map_path", ".") + "/" + path)
                self.system_message(f"Loaded {path}")
            except:
                c.message(f"&cFailed to load \"{path}\"")
        elif msg.startswith("/save ") and c.oper:
            path = msg[6:].strip() + ".rtm"

            if "./" in path:
                c.message(f"&cInvalid path")
                return

            try:
                self.level.save(self.config.get("map_path", ".") + "/" + path)
                self.system_message(f"Saved as {path}")
            except:
                self.system_message(f"Failed to save {path}")
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

            self.system_message(f"Generating new {args[0]} map")
            self.new_map(args[0],
                    Vec3(int(args[1]), int(args[2]), int(args[3])))
            self.system_message(f"Done generating {args[0]} map")
        elif msg.startswith("/tp "):
            args = msg[4:].split(" ")

            try:
                x, y, z = int(args[0]), int(args[1]), int(args[2])
                c.teleport(Vec3(x, y, z), c.angle)
            except:
                c.message("&cUsage: /tp x y z")
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
        self.state: ServerState = state
        self.allow_reuse_address = True
        super().__init__(address, request_handler)


if __name__ == "__main__":
    import sys

    state = ServerState()
    if len(sys.argv) > 1:
        state.load_config(sys.argv[1])

    ip = state.config.get("listen_ip", ("127.0.0.1", 25565))

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

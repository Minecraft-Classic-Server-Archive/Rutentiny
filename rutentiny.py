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
import re
from enum import IntEnum
from struct import pack, unpack, calcsize
from typing import NamedTuple, Optional, Iterator

try:
    import opensimplex
    simplex_available = True
except:
    print("opensimplex unavailable, falling back to bad noise algorithm")
    simplex_available = False

# FIXME: this could truncate UTF-8 sequences
def pad(msg: str, encoding: str = "ascii") -> bytes:
    return msg.ljust(64).encode(encoding, "replace")


# hack to get textwrap to leave cp437 control characters alone
msg_pad_wrapper = textwrap.TextWrapper(
        width=64,
        subsequent_indent="  ",
        expand_tabs=False,
        tabsize=1,
        replace_whitespace=False)
def pad_msg(msg: str, encoding: str = "ascii") -> list[bytes]:
    from textwrap import wrap
    buf = []
    for line in msg_pad_wrapper.wrap(msg):
        buf.append(pad(line, encoding))
    return buf


def pad_data(data: bytes) -> bytes:
    trim = data[:1024]
    return trim + (b"\0" * (1024 - len(trim)))


def chunk_iter(data: bytes) -> Iterator[bytes]:
    return (data[pos:pos + 1024] for pos in range(0, len(data), 1024))


color_remove_re = re.compile(r'&[0-9a-f]')
def log(msg: str) -> None:
    msg = color_remove_re.sub(r'', msg)
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

    def to_block(self) -> "Vec3":
        return Vec3(self.x // 32, (self.y - 51) // 32, self.z // 32)

    def to_fixed(self, ext: bool = False) -> "Vec3":
        if ext:
            limit = 2**31
        else:
            limit = 2**15

        return Vec3(
                int(clamp(self.x * 32, -limit, limit - 1)),
                int(clamp((self.y * 32) + 51, -limit, limit - 1)),
                int(clamp(self.z * 32, -limit, limit - 1)))

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

    # !! keep this up to date or find an automatic solution
    MAXIMUM = 67,

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
        from collections import deque

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
        self.dmg_queue: deque[int] = deque([])
        self.held_block: int = 0
        self.model: str = "humanoid"
        self.kills: int = 0
        self.deaths: int = 0
        #self.locale: str = "en-US"     # TODO: is this useful for anything?
        self.encoding: str = "ascii"

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
            self.deaths += 1

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
                self.message("&eYou died!", 100)
            elif self.gamemode == 4:
                self.score_update()

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
                    self.name = name.strip().decode("ascii", "replace")
                    self.key = key.strip().decode("ascii", "replace")

                    if ver != 7:
                        self.kick(f"Only protocol 7 is supported, got {ver}")
                        return
                except:
                    self.kick("Corrupted login packet")
                    return

                if self.name in self.server.config.get("banned", []):
                    self.kick("Banned")
                    return

                if len(self.server.clients) >= self.server.max_clients:
                    self.kick("Server is full")
                    return

                if self.server.config.get("heartbeat_url", None):
                    from hashlib import md5
                    expect_key = md5(self.server.key.encode() + name.rstrip(b" ")).hexdigest()
                    if self.key != expect_key:
                        self.kick("Invalid login key")
                        return
                elif self.server.key:
                    if self.server.key != self.key:
                        self.kick("Invalid passkey")
                        return

                if pad == 0x42:
                    self.uses_cpe = True
                    self.server.cpe_handshake(self)

                self.server.send_id(self)

                if welcome := self.server.config.get("welcome_msg"):
                    for msg in welcome:
                        self.message(msg)

                self.server.level.send_to_client(self)
                self.server.add_client(self)
                self.spawn(*self.server.level.get_spawnpoint())
                self.set_gamemode(self.server.level.gamemode)

                # we can dream
                if ("UTF-8", 1) in self.cpe_exts:
                    self.encoding = "utf_8"
                elif ("FullCP437", 1) in self.cpe_exts:
                    self.encoding = "cp437"

            # block set
            case 5:
                x, y, z, mode, block = unpack("!hhhBB", self.recv(8))

                # only survival and creative allow block placing
                if self.gamemode != 0 and self.gamemode != 1:
                    self.send(pack("!BhhhB", 6, x, y, z,
                        self.server.level.get_block(x, y, z)))
                else:
                    if mode == 0:
                        self.server.set_block(Vec3(x, y, z), BlockID.NONE)
                    else:
                        self.server.set_block(Vec3(x, y, z), block)

            # position update
            case 8:
                if ("ExtEntityPositions", 1) in self.cpe_exts:
                    strucdef = "!BiiiBB"
                else:
                    strucdef = "!BhhhBB"

                block, x, y, z, yaw, pitch = unpack(strucdef,
                        self.recv(calcsize(strucdef)))
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

                self.held_block = block

            # message
            case 13:
                import re
                ext, msg = unpack("B64s", self.recv(65))
                self.msg_buffer += msg

                if ext == 0:
                    msg = self.msg_buffer.strip().decode(self.encoding,
                            "replace")
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
                if self.gamemode == 4 and target.gamemode == 4:
                    if data[0] == 0 and data[1] == 0:
                        target.dmg_queue.append(200)
                        self.kills += 1
                        self.server.message(
                                f"&f{self.name} &7x &c{target.name}", 11)
                        self.score_update()
                elif self.gamemode == 0 and target.gamemode == 0:
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

    def score_update(self):
        if ("MessageTypes", 1) in self.cpe_exts:
            self.message(f"&eScore: \
&a{self.kills} &f| &c{self.deaths} &f| &7{self.kills - self.deaths}", 12)

    def health_update(self):
        if ("RutentoyGamemode", 1) in self.cpe_exts:
            self.send(pack("Bbbbx", 0xa1, self.health,
                self.armor, self.air))
        elif ("MessageTypes", 1) in self.cpe_exts:
            # HACK: CC appears to ignore all-space chat messages,
            # so send a single null byte which appears empty as well
            if self.health > -1 and self.gamemode != 1:
                self.message(f"&cHealth: {self.health}", 1)
            else:
                self.message("\x00", 1)

            if self.armor > -1:
                self.message(f"&7Armor: {self.armor}", 2)
            else:
                self.message("\x00", 2)

            if self.air > -1 and self.air < 21:
                self.message(f"&bAir: {self.air}", 3)
            else:
                self.message("\x00", 3)

    def set_gamemode(self, mode: str | int) -> None:
        ruten = ("RutentoyGamemode", 1) in self.cpe_exts

        self.health = 20
        self.armor = -1
        self.air = -1
        self.kills = 0
        self.deaths = 0

        hax = "-hax -push"
        allow_block = 0

        match mode:
            case "survival" | "0" | 0:
                if ruten:
                    self.send(pack("BBB", 0xa0, 0, 1))
                self.gamemode = 0
                allow_block = 1

            case "creative" | "1" | 1:
                if ruten:
                    self.send(pack("BBB", 0xa0, 1, 1))
                self.gamemode = 1
                hax = "+hax -push"
                allow_block = 1

            case "explore" | "2" | 2:
                if ruten:
                    self.send(pack("BBB", 0xa0, 1, 0))
                self.gamemode = 2

            case "instagib" | "4" | 4:
                if ruten:
                    self.send(pack("BBB", 0xa0, 0, 0))
                self.gamemode = 4
                hax = "-hax -push horspeed=2"

            case _:
                self.message("&cInvalid gamemode")
                return

        # clear the cpe MessageTypes stuff
        if ("MessageTypes", 1) in self.cpe_exts:
            self.message("\x00", 1)
            self.message("\x00", 2)
            self.message("\x00", 3)
            self.message("\x00", 11)
            self.message("\x00", 12)
            self.message("\x00", 13)

        self.health_update()

        # RutentoyGamemode will already provide this functionality but
        # upstream ClassiCube needs this to enforce block access
        if ("BlockPermissions", 1) in self.cpe_exts:
            for i in range(1, BlockID.MAXIMUM):
                # SetBlockPermission
                self.send(pack("BBBB", 28, i, allow_block, allow_block))

        if self.gamemode == 4:
            self.score_update()
            if ("HeldBlock", 1) in self.cpe_exts:
                # HoldThis
                self.send(pack("BBB", 20, BlockID.NONE, 1))
        else:
            # this has to be here or the held block will be permanently locked
            if ("HeldBlock", 1) in self.cpe_exts:
                # HoldThis
                self.send(pack("BBB", 20, BlockID.NONE, 0))

        # ClassiCube is really really picky and has SetBlockPermission/oper mode
        # conflicts that make it really annoying to enable/disable all blocks
        if ("InstantMOTD", 1) in self.cpe_exts:
            self.send(pack("BB64s64sB", 0, 7,
                pad(self.server.config.get("name", "Rutentiny")),
                pad(hax), 100 if allow_block else 0))

    # NOTE: expects a fixed-point Vec3, unlike teleport
    def spawn(self, pos: Vec3, angle: Vec2) -> None:
        if ("ExtEntityPositions", 1) in self.cpe_exts:
            strucdef = "!BB64siiiBB"
        else:
            strucdef = "!BB64shhhBB"

        self.send(pack(strucdef, 7, 255, pad(self.name),
            pos.x, pos.y, pos.z, angle.y, angle.x))

    def teleport(self, pos: Vec3, angle: Vec2) -> None:
        self.pos = pos
        self.angle = angle
        fpos = Vec3(pos.x + .5, pos.y, pos.z + .5).to_fixed()

        if ("ExtEntityPositions", 1) in self.cpe_exts:
            strucdef = "!BBiiiBB"
        else:
            strucdef = "!BBhhhBB"

        self.send(pack(strucdef, 8, 255,
            fpos.x, fpos.y, fpos.z,
            angle.y, angle.x))

    def kick(self, reason: str) -> None:
        self.send(b"\x0e" + pad(reason, self.encoding))
        self.server.remove_client(self)
        self.disconnected = True

    def message(self, msg: str, type: int = 0) -> None:
        # NOTE: wiki.vg doesn't document all of CC's capabilities,
        # maybe types 101 and 102 aren't standard CPE
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

        chunks = pad_msg(msg, self.encoding)
        for chunk in chunks:
            self.send(pack("BB64s", 13, type, chunk))


class NPCEntity(Client):
    def __init__(self, server: "ServerState", model: str):
        super().__init__(None, server)
        self.model = model
        self.name = ""

    def recv(self, size: int) -> bytes:
        return b""

    def send(self, data: bytes) -> None:
        pass

    def packet(self):
        pass

    def tick(self):
        if self.disconnected:
            return

    def spawn(self, pos: Vec3, angle: Vec2) -> None:
        self.pos = pos
        self.angle = angle
        self.server.set_playermodel(self, self.model)

    def teleport(self, pos: Vec3, angle: Vec2) -> None:
        self.pos = pos
        self.angle = angle

    def kick(self, reason: str | None) -> None:
        self.disconnected = True

    def message(self, msg: str, type: int = 0) -> None:
        pass


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
        self.last_modified: int = int(time.time())

        if size is None:
            self.xs, self.ys, self.zs = self.server.config.get("map_size",
                    (128, 128, 128))
        else:
            self.xs, self.ys, self.zs = size

        self.spawn_points: list[tuple[Vec3, Vec2]] = [
                (Vec3(self.xs/2, self.ys+1, self.zs/2).to_fixed(), Vec2(0, 0)),
                ]
        self.gamemode = "creative"
        self.name = "Unnamed"

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

    def update_modtime(self) -> None:
        self.last_modified = int(time.time())

    def save(self, path: str) -> None:
        log(f"saving map {path}")
        with gzip.open(path, "wb") as file:
            file.write(b"rtm\x01")
            file.write(pack("!iii", self.xs, self.ys, self.zs))
            file.write(self.map)

            # FIXME: sometimes this just breaks and corrupts the save
            try:
                file.write(pack("!q", self.seed))
            except Exception as e:
                log(f"saving: (self.seed = {self.seed}) {e}")
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

            file.write(pad(self.gamemode, "ascii"))
            file.write(pad(self.name, "ascii"))

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
                    self.seed, = unpack("!q", file.read(8))
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

                self.gamemode = file.read(64).strip().decode("ascii", "replace")
                self.name = (file.read(64) or b"").strip().decode("ascii", "replace")
            else:
                log(f"Invalid map format {repr(magic)}")
                raise ValueError(f"Invalid map format {repr(magic)}")

        self.ready = True

    def get_spawnpoint(self) -> tuple[Vec3, Vec2]:
        from random import choice

        if len(self.spawn_points) < 1:
            return (Vec3(self.xs//2, self.ys+1, self.zs//2).to_fixed(),
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
        self.update_modtime()

        if self.ready:
            for cl in self.server.clients:
                cl.send(pack("!BhhhB", 6, x, y, z, block))

    def get_block(self, x: int, y: int, z: int) -> BlockID:
        i = self._index(x, y, z)
        if i < 0 or i >= len(self.map):
            return BlockID.NONE

        return BlockID(self.map[i])

    def noise(self, x: float, y: float) -> float:
        if simplex_available:
            return opensimplex.noise2(x, y)
        else:
            from random import random, seed
            from math import trunc

            def lerp(a: float, b: float, t: float) -> float:
                return a + (b - a) * t

            def subhash(n: int) -> float:
                seed(n)
                return random()

            def fhash(n: float) -> float:
                t = (n - trunc(n)) ** 2
                a = subhash(int(n + self.seed))
                b = subhash(int(n + 1) + self.seed)
                return lerp(a, b, t)

            # for some reason it needs to be 3x to
            # be the same scale as the opensimplex?
            nx = fhash(x * 3)
            ny = fhash(-(y * 3) + 4335)

            return (nx + ny) - 1

    def tree(self, rx: int, ry: int, rz: int) -> None:
        # dont generate half outside the level
        if (rx, ry, rz) <= (0, 0, 0) or (rx, ry, rz) >= (self.xs - 1, self.ys - 1, self.zs - 1):
            return

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

        self.update_modtime()

    def boulder(self, rx: int, ry: int, rz: int) -> None:
        # dont generate half outside the level
        if (rx, ry, rz) <= (0, 0, 0) or (rx, ry, rz) >= (self.xs - 1, self.ys - 1, self.zs - 1):
            return

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

        self.update_modtime()

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
                        if randrange(0, 768) == 0:
                            self.boulder(x, height + randrange(-4, 0), z)

                    if n > -0.1 and n < 0.1:
                        if randrange(0, 32) == 0:
                            if randrange(0, 1) == 0:
                                self.set_block(x, height, z, BlockID.DANDELION)
                            else:
                                self.set_block(x, height, z, BlockID.ROSE)

            if z % 32 == 0:
                log(f"{int((z / self.zs) * 100)}% generated")
        log("Map generation done")
        self.ready = True
        self.update_modtime()

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
        self.update_modtime()

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
        self.update_modtime()

    def generate_moon(self):
        from random import randrange
        self.ready = False
        log("Generating Moon map")
        self.edge = ((BlockID.GRAVEL, BlockID.BEDROCK), (self.ys // 2, 0))
        self.clouds = (-16, 1)
        self.colors = [
                (0x00, 0x00, 0x00), # sky
                (0xdf, 0x71, 0x26), # clouds
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
        self.update_modtime()

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
        self.update_modtime()

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
        self.update_modtime()

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
        self.update_modtime()

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
                67, 67, 67,     # top, side, bottom textures
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
                37, 37, 37,     # top, side, bottom textures
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


class ServerState:
    def __init__(self):
        self.level: Level = None
        self.clients: list[Client] = []
        self.config: dict = {}
        self.alive: bool = True
        self.key: str = ""
        self.max_clients: int = 8
        self.next_autosave: int = 0

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
            if ("ExtEntityPositions", 1) in cl.cpe_exts:
                strucdef = "!BB64siiiBB"
            else:
                strucdef = "!BB64shhhBB"

            if c != cl:
                c.send(pack(strucdef, 7, self.clients.index(cl),
                    pad(cl.name), cl.pos.x, cl.pos.y, cl.pos.z,
                    cl.angle.y, cl.angle.x))

                cl.send(pack(strucdef, 7, self.clients.index(c),
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

    def add_npc(self, npc: NPCEntity, name: str,
            pos: Vec3, angle: Vec2) -> None:
        if npc in self.clients:
            return

        npc.pos = pos
        npc.angle = angle
        npc.name = name
        self.clients.append(npc)
        npci = self.clients.index(npc)

        for cl in self.clients:
            if ("ExtEntityPositions", 1) in cl.cpe_exts:
                strucdef = "!BB64siiiBB"
            else:
                strucdef = "!BB64shhhBB"

            if cl is not NPCEntity:
                cl.send(pack(strucdef, 7, npci,
                    pad(npc.name), npc.pos.x, npc.pos.y, npc.pos.z,
                    npc.angle.y, npc.angle.x))

        self.set_playermodel(npc, npc.model)

    def shutdown(self, reason: str) -> None:
        log("Server shutting down")

        if self.config.get("map_shutdown_autosave", False):
            path = f"autosave-{int(time.time())}.rtm"
            self.level.save(self.config.get("map_path", ".") + "/" + path)

        reasonpad = pad(reason)
        for c in self.clients:
            c.send(b"\x0e" + reasonpad)
            c.disconnected = True

        self.clients = []
        self.alive = False

    def send_id(self, c: Client) -> None:
        if c.name in self.config.get("opers", []):
            c.oper = True

        # always set the client type to operator to allow special blocks,
        # afaik the client doesn't do anything else with it
        c.send(pack("BB64s64sB", 0, 7,
            pad(self.config.get("name", "Rutentiny")),
            pad(self.config.get("motd", "My Cool Server")), 100))

    def cpe_handshake(self, c: Client) -> None:
        c.send(pack("!B64sH", 16, b"Rutentiny 0.1.0", 12))
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
        c.send(pack("!B64sI", 17, b"ExtEntityPositions", 1))
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

    def update_env_properties(self):
        for c in self.clients:
            if ("EnvMapAspect", 1) in c.cpe_exts:
                # edge blocks
                c.send(pack("!BBi", 41, 0, self.level.edge[0][1]))
                c.send(pack("!BBi", 41, 9, self.level.edge[1][1]))
                c.send(pack("!BBi", 41, 1, self.level.edge[0][0]))
                c.send(pack("!BBi", 41, 2, self.level.edge[1][0]))

                # clouds
                c.send(pack("!BBi", 41, 3, self.level.clouds[0]))
                c.send(pack("!BBi", 41, 5, self.level.clouds[1]))

            if ("EnvColors", 1) in c.cpe_exts:
                for i in range(0, 5):
                    col = self.level.colors[i]
                    c.send(pack("!BBhhh", 25, i, col[0], col[1], col[2]))

    def do_autosave(self) -> None:
        t = int(time.time())
        i = self.autosave_interval

        if self.level.last_modified + i >= t:
            path = f"autosave-{t}.rtm"
            self.level.save(self.config.get("map_path", ".") + "/" + path)

            self.message(f"&eAutosaving... (every {i / 60} minutes)")
        else:
            log(f"Would have autosaved, but no activity in the last {i} minutes")

        timer = threading.Timer(self.autosave_interval, self.do_autosave)
        timer.daemon = True
        timer.start()

    def tick(self) -> None:
        for c in self.clients:
            # if a player crashed or lost connection, remove them
            if c.disconnected:
                self.remove_client(c)

            if c.pos != c.old_pos or c.angle != c.old_angle:
                self.player_move(c)
            c.tick()

        timer = threading.Timer(1 / 20, self.tick)
        timer.daemon = True
        timer.start()

    def player_move(self, c: Client) -> None:
        i = self.clients.index(c)

        for cl in self.clients:
            if c == cl:
                continue

            if ("ExtEntityPositions", 1) in cl.cpe_exts:
                strucdef = "!BBiiiBB"
            else:
                strucdef = "!BBhhhBB"

            cl.send(pack(strucdef, 8, i, c.pos.x, c.pos.y, c.pos.z,
                c.angle.y, c.angle.x))

    def set_playermodel(self, c: Client, model: str, scale: int = 1000) -> None:
        i = self.clients.index(c)
        models = pad(model)

        c.model = model

        for cl in self.clients:
            if ("ChangeModel", 1) not in cl.cpe_exts:
                continue

            # ChangeModel
            if c == cl:
                cl.send(pack("!Bb64s", 29, -1, models))
            else:
                cl.send(pack("!Bb64s", 29, i, models))

            if ("EntityProperty", 1) not in cl.cpe_exts:
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

    def set_block(self, pos: Vec3, block: BlockID) -> None:
        self.level.set_block(pos.x, pos.y, pos.z, block)

    def message(self, msg: str, type: int = 0) -> None:
        if type == 0:
            log(msg)

        for c in self.clients:
            # re-encoding for each client is a necessary evil to support
            # multiple character encodings
            c.message(msg, type)

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
            return

        log(f"{c.name} used command '{msg}'")

        if msg.startswith("/gamemode ") \
                and (c.oper or self.level.gamemode == "creative"):
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
                c.teleport(Vec3(int(args[0]), int(args[1]), int(args[2])),
                        c.angle)
            except:
                c.message("&cUsage: /tp x y z")
        elif msg.startswith("/tree") and c.gamemode == 1:
            self.level.tree(*c.pos.to_block())
        elif msg.startswith("/model ") and (c.gamemode == 1 or c.oper):
            self.set_playermodel(c, msg[7:])
        elif msg.startswith("/lp ") and (c.gamemode == 1 or c.oper):
            args = msg[4:].split()

            if len(args) < 1:
                c.message("&eAvailable properties:")
                c.message("&e  name, gamemode, spawns,")
                c.message("&e  clouds, edge, sky-color, fog-color,")
                c.message("&e  cloud-color, diffuse-light, ambient-light")
                return

            match args[0]:
                case "spawns":
                    if len(args) < 2:
                        c.message("&espawns (list, remove, add)")
                        return

                    if args[1] == "list":
                        c.message("&eSpawn points:")
                        for i in range(len(self.level.spawn_points)):
                            point, _ = self.level.spawn_points[i]
                            c.message(f"&e{i:>4}: {point.to_block()}")
                    elif args[1] == "remove":
                        try:
                            i = int(args[2])
                            self.level.spawn_points.pop(i)
                            c.message(f"&eRemoved spawn point {i}")
                        except:
                            c.message("&cUsage: spawns remove index")
                    elif args[1] == "add":
                        self.level.spawn_points.append((c.pos, c.angle))
                        c.message(f"&eAdded new spawn point at {c.pos.to_block()}")
                    return

                case "clouds":
                    try:
                        h = int(args[1])
                        s = int(float(args[2]) * 256)
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cheight, speed")
                        return

                    self.level.clouds = (h, s)

                case "edge":
                    try:
                        top_id = int(args[1])
                        side_id = int(args[2])
                        height = int(args[3])
                        top_diff = int(args[4])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &ctop id, side id, height, side height")
                        return

                    self.level.edge = ((top_id, side_id), (height, top_diff))

                case "sky-color":
                    try:
                        r = int(args[1])
                        g = int(args[2])
                        b = int(args[3])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cred, green, blue")
                        return

                    self.level.colors[0] = (r, g, b)

                case "fog-color":
                    try:
                        r = int(args[1])
                        g = int(args[2])
                        b = int(args[3])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cred, green, blue")
                        return

                    self.level.colors[2] = (r, g, b)

                case "cloud-color":
                    try:
                        r = int(args[1])
                        g = int(args[2])
                        b = int(args[3])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cred, green, blue")
                        return

                    self.level.colors[1] = (r, g, b)

                case "diffuse-light":
                    try:
                        r = int(args[1])
                        g = int(args[2])
                        b = int(args[3])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cred, green, blue")
                        return

                    self.level.colors[4] = (r, g, b)

                case "ambient-light":
                    try:
                        r = int(args[1])
                        g = int(args[2])
                        b = int(args[3])
                    except:
                        c.message(f"&cRequired arguments:")
                        c.message(f"  &cred, green, blue")
                        return

                    self.level.colors[3] = (r, g, b)

                case "name":
                    if len(args) < 2:
                        c.message(f"Map name: &e{self.level.name}")
                        return

                    try:
                        newname = " ".join(args[1:])
                    except:
                        return

                    self.level.name = newname
                    c.message(f"Map name: &e{self.level.name}")

                case "gamemode":
                    if len(args) < 2:
                        c.message(f"Level gamemode: &e{self.level.gamemode}")
                        return

                    try:
                        newgm = args[1]
                    except:
                        return

                    self.level.gamemode = newgm
                    c.message(f"Level gamemode: &e{self.level.gamemode}")

                case _:
                    c.message(f"&cUnknown map property \"{args[0]}\"")
                    return

            self.update_env_properties()
        elif msg.startswith("/fill ") and c.oper:
            args = msg[6:].split()

            if len(args) < 7:
                c.message("&cUsage: /fill block-id x1 y1 z1 x2 y2 z2")
                return

            try:
                block = int(args[0])
                x1 = int(args[1])
                y1 = int(args[2])
                z1 = int(args[3])
                x2 = int(args[4])
                y2 = int(args[5])
                z2 = int(args[6])
            except Exception as e:
                c.message("&cUsage: /fill block-id x1 y1 z1 x2 y2 z2")
                return

            for y in range(y1, y2):
                for x in range(x1, x2):
                    for z in range(z1, z2):
                        self.set_block(Vec3(x, y, z), block)
        elif msg.startswith("/spawn ") and c.oper:
            c.message("&cNPCs are unfinished and will break your server!")
            model = msg[7:]
            model, _, name = model.partition(" ")
            if model:
                self.add_npc(NPCEntity(self, model), name or "",
                        c.pos, c.angle)
            else:
                c.message("&cUsage: /spawn type [name]")
        else:
            c.message("&cUnknown or disallowed command")


class RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.client: Client = Client(self.request, self.server.state)

        while not self.client.disconnected:
            try:
                self.client.packet()
            except Exception as e:
                self.client.kick(f"Error: {type(e)}")

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
        url += f"port={cfg('listen_ip', (None, 25565))[1]}&"
        url += f"max={cfg('max_clients', 8)}&"
        url += f"name={parse.quote(cfg('name', 'Rutentoy'))}&"
        url += f"public={parse.quote(str(bool(cfg('public', False))))}&"
        url += f"version=7&"
        url += f"salt={parse.quote(self.state.key)}&"
        url += f"software={parse.quote('Rutentiny 0.1.0')}&"
        url += f"web=False"

        # wait for the level to finish generating
        sleep(15)
        log(f"Heartbeating (1/min) to {cfg('heartbeat_url')}")

        last_heartbeat_response = None

        while True:
            if not state.level.ready:
                log("Was going to heartbeat, but level isn't ready")
                continue

            try:
                turl = f"{url}&users={len(self.state.clients)}"
                resp = request.urlopen(turl, timeout=5).read().decode()

                # don't spam the log with the same url the heartbeat server gives us
                if len(resp) > 0:
                    if resp != last_heartbeat_response:
                        last_heartbeat_response = resp
                        log(f"Heartbeat: {resp}")
            except Exception as e:
                log(f"Heartbeat error: {e}")

            sleep(60)


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

        if (ivl := state.config.get("autosave_interval", 0)) != 0:
            state.autosave_interval = ivl * 60

            # wait a bit so we dont immediately autosave on startup
            timer = threading.Timer(state.autosave_interval, state.do_autosave)
            timer.daemon = True
            timer.start()

        state.tick()

        try:
            while state.alive:
                state.sys_command(input().split())
        except (KeyboardInterrupt, EOFError):
            state.shutdown("Server shutting down")
        except:
            state.shutdown("Possible server crash")

        server.shutdown()

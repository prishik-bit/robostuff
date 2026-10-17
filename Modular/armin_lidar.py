"""
File purpose: TLDR - LDRobot D500 (LD19) lidar reader over UART

Description:
The D500 streams measurements continuously over UART at 230400 baud; there is
nothing to send to it. Every 47-byte packet holds 12 points spread between a
start and end angle:

    0x54 0x2C | speed u16 | start angle u16 | 12 x (distance u16, intensity u8)
              | end angle u16 | timestamp u16 | CRC8

All multi-byte fields are little-endian. Angles are in 0.01 degrees, distance is
in millimetres, speed is in degrees per second.

A background thread reads the port and keeps the latest distance for each whole
degree. Angles are reported in the robot convention used by the vision code:
0 = forward, clockwise positive. The LD19 itself measures clockwise viewed from
above; set LidarConfig.mounting_offset so the lidar's 0 lines up with forward.
"""

import struct
import threading
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import serial  # pyserial

from armin_config import LidarConfig

HEADER = 0x54
VER_LEN = 0x2C          # packet type 1 (upper 3 bits), 12 points (lower 5 bits)
POINTS_PER_PACKET = 12
PACKET_SIZE = 47


def _build_crc_table() -> List[int]:
    """CRC-8, polynomial 0x4D, no reflection, init 0, as used by LDRobot."""
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


CRC_TABLE = _build_crc_table()


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


def parse_packet(packet: bytes) -> Optional[Tuple[float, List[Tuple[float, int, int]]]]:
    """Decode one packet into ``(speed_deg_s, [(angle_deg, distance_mm, intensity), ...])``.

    Returns None if the header or CRC is wrong. Angles here are raw lidar angles.
    """
    if len(packet) != PACKET_SIZE or packet[0] != HEADER or packet[1] != VER_LEN:
        return None
    if crc8(packet[:-1]) != packet[-1]:
        return None

    speed, start_angle = struct.unpack_from('<HH', packet, 2)
    end_angle, _timestamp = struct.unpack_from('<HH', packet, 42)

    # Interpolate the 12 point angles; the end angle wraps past 360 on the packet that crosses 0.
    if end_angle < start_angle:
        end_angle += 36_000
    step = (end_angle - start_angle) / (POINTS_PER_PACKET - 1)

    points = []
    for i in range(POINTS_PER_PACKET):
        distance, intensity = struct.unpack_from('<HB', packet, 6 + i * 3)
        angle = ((start_angle + step * i) / 100.0) % 360.0
        points.append((angle, distance, intensity))
    return float(speed), points


def iter_packets(buffer: bytearray) -> Iterator[bytes]:
    """Pull every complete, valid packet out of ``buffer``, resyncing on corrupt data.

    Consumed bytes are removed from ``buffer`` in place; a trailing partial packet is kept.
    """
    while True:
        start = buffer.find(bytes((HEADER, VER_LEN)))
        if start < 0:
            # Keep a possible lone header byte at the end for the next read.
            del buffer[:-1]
            return
        if len(buffer) - start < PACKET_SIZE:
            del buffer[:start]
            return
        packet = bytes(buffer[start:start + PACKET_SIZE])
        if crc8(packet[:-1]) == packet[-1]:
            del buffer[:start + PACKET_SIZE]
            yield packet
        else:
            # False header match or corrupted packet: skip this header and search again.
            del buffer[:start + 1]


class LidarService:
    """Read the lidar on a background thread and expose the latest 360-degree scan."""

    def __init__(self, config: LidarConfig = None) -> None:
        self.config = config or LidarConfig()
        self.serial = None
        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()
        # One bin per whole degree in the robot frame. 0 distance = no valid reading.
        self._distances = np.zeros(360, dtype=np.float32)
        self._intensities = np.zeros(360, dtype=np.uint8)
        self._updated = np.zeros(360, dtype=np.float64)
        self.speed_deg_s = 0.0
        self.packets_ok = 0
        self.packets_bad = 0

    # ---- setup ----------------------------------------------------------

    def setup(self) -> None:
        """Open the UART and start the reader thread."""
        self.serial = serial.Serial(self.config.port, self.config.baudrate, timeout=0.1)
        self.serial.reset_input_buffer()
        self.is_running = True
        self._thread = threading.Thread(target=self._read_loop, name='lidar', daemon=True)
        self._thread.start()
        print(f"Lidar ready on {self.config.port}")

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.serial is not None:
            self.serial.close()

    # ---- reader thread --------------------------------------------------

    def _read_loop(self) -> None:
        buffer = bytearray()
        while self.is_running:
            chunk = self.serial.read(self.serial.in_waiting or PACKET_SIZE)
            if not chunk:
                continue
            buffer.extend(chunk)
            for packet in iter_packets(buffer):
                parsed = parse_packet(packet)
                if parsed is None:
                    self.packets_bad += 1
                    continue
                self.packets_ok += 1
                self._store(*parsed)

    def _store(self, speed: float, points: List[Tuple[float, int, int]]) -> None:
        now = time.time()
        with self._lock:
            self.speed_deg_s = speed
            for angle, distance, intensity in points:
                if not self._is_valid(distance, intensity):
                    continue
                index = int(round(angle + self.config.mounting_offset)) % 360
                self._distances[index] = distance
                self._intensities[index] = intensity
                self._updated[index] = now

    def _is_valid(self, distance: int, intensity: int) -> bool:
        return (
            self.config.min_distance_mm <= distance <= self.config.max_distance_mm
            and intensity >= self.config.min_intensity
        )

    # ---- queries (safe to call from the motor loop) ---------------------

    def get_scan(self) -> np.ndarray:
        """360 distances in mm indexed by robot bearing; stale or missing bins are NaN."""
        now = time.time()
        with self._lock:
            scan = self._distances.copy()
            stale = (now - self._updated) > self.config.stale_after
        scan[stale | (scan == 0)] = np.nan
        return scan

    def get_distance(self, bearing: float, window: int = 2) -> Optional[float]:
        """Median distance in mm around ``bearing`` (+/- ``window`` degrees), or None."""
        scan = self.get_scan()
        indices = [int(round(bearing + offset)) % 360 for offset in range(-window, window + 1)]
        values = scan[indices]
        values = values[~np.isnan(values)]
        return float(np.median(values)) if values.size else None

    def get_nearest(self) -> Optional[Tuple[float, float]]:
        """``(bearing, distance_mm)`` of the closest valid return, or None."""
        scan = self.get_scan()
        if np.all(np.isnan(scan)):
            return None
        index = int(np.nanargmin(scan))
        return float(index), float(scan[index])

    def get_wall_distances(self) -> dict:
        """Distances straight ahead, right, behind and left - handy for field positioning."""
        return {
            'front': self.get_distance(0),
            'right': self.get_distance(90),
            'back': self.get_distance(180),
            'left': self.get_distance(270),
        }


if __name__ == '__main__':
    lidar = LidarService()
    lidar.setup()
    try:
        while True:
            time.sleep(0.5)
            walls = lidar.get_wall_distances()
            print(
                f"{lidar.speed_deg_s / 360:4.1f} rev/s | ok {lidar.packets_ok} bad {lidar.packets_bad} | "
                + ' '.join(f"{name}={'---' if d is None else f'{d:.0f}mm'}" for name, d in walls.items())
            )
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()

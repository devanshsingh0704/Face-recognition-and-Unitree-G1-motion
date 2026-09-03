"""G1 locomotion for the face-target approach: walk forward and yaw to aim.

Every command is a LocoClient.SetVelocity carrying a short explicit duration,
never LocoClient.Move(continous_move=True) -- that one sends duration=864000,
ten days, and a killed process or a stalled wifi link would leave the robot
walking across the room. With CHUNK_S the robot expires the command by itself
within half a second of anything going quiet.

vy stays at zero. Sidestepping is the G1's least stable gait, and aiming with
yaw is what a person does anyway.
"""

from __future__ import annotations

import json
import sys
import threading
import time

SDK = "/home/unitree/unitree_sdk2_python"
CHUNK_S = 0.5

# Ceilings applied to whatever the caller asks for, so a mistyped gain upstream
# cannot turn into a sprint. Well inside what the G1 will do.
MAX_VX = 0.35
MAX_OMEGA = 0.6

# Read-only diagnostic hook. api 7001 returns the locomotion FSM id.
#
# Do NOT gate motion on a particular value. LocoClient.Start() happens to send
# SetFsmId(200), and on 2026-08-29 that led me to conclude 200 was the only
# walking state and 501 was not one -- which was wrong, and briefly shipped a
# check that refused a perfectly capable robot. The SDK's convenience wrappers
# are not a list of the firmware's valid states, and we do not have that list.
#
# What IS true and worth keeping: SetVelocity acks every command it parses, so
# a zero return says the message was understood, not that the robot moved.
# Confirm motion by its effect.
_API_GET_FSM_ID = 7001


def _ensure_sdk() -> None:
    if SDK not in sys.path:
        sys.path.insert(0, SDK)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


class G1Locomotion:
    """Lazy-connecting wrapper around LocoClient."""

    def __init__(self, iface: str = "eth0") -> None:
        self.iface = iface
        self._loco = None
        self._ready = False
        # connect() can be reached from the approach loop and from a
        # diagnostic at the same time, and ChannelFactoryInitialize must not
        # run twice.
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            if self._ready:
                return
            _ensure_sdk()
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

            ChannelFactoryInitialize(0, self.iface)
            loco = LocoClient()
            loco.SetTimeout(10.0)
            loco.Init()
            # Published only once it is fully built, so a racing caller never
            # sees a half-initialised client.
            self._loco = loco
            self._ready = True

    def fsm_id(self) -> int | None:
        """Current locomotion FSM id, or None if it cannot be read.

        Goes through LocoClient._Call because the client exposes setters for
        the FSM but no getter, and api 7001 is the only way to ask. Private,
        and therefore worth knowing it could break on an SDK update -- but the
        alternative is commanding a robot without knowing whether it will move.
        """
        if not self._ready or self._loco is None:
            return None
        try:
            code, data = self._loco._Call(_API_GET_FSM_ID, "")
        except Exception:
            return None
        if code:
            return None
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
            if isinstance(parsed, dict):
                return int(parsed.get("data"))
            return int(parsed)
        except (TypeError, ValueError):
            return None

    def stop(self) -> None:
        if not self._ready or self._loco is None:
            return
        self._loco.SetVelocity(0.0, 0.0, 0.0, CHUNK_S)

    def step(self, vx: float, omega: float = 0.0,
             duration: float | None = None) -> int:
        """Hold vx forward and omega yaw for *duration* seconds. Returns the SDK code.

        Positive omega turns left, matching the SDK's convention. A non-zero
        return usually means the robot is not standing and is worth surfacing
        rather than swallowing -- otherwise the dashboard reports an approach
        that the robot never began.
        """
        if not self._ready or self._loco is None:
            return 0
        dur = CHUNK_S if duration is None else float(duration)
        return self._loco.SetVelocity(
            _clamp(vx, MAX_VX), 0.0, _clamp(omega, MAX_OMEGA), dur
        )

    def step_forward(self, vx: float) -> int:
        return self.step(vx, 0.0)

    def damp(self) -> None:
        if not self._ready or self._loco is None:
            return
        self._loco.StopMove()

    def shutdown(self) -> None:
        if not self._ready:
            return
        try:
            self.stop()
            time.sleep(0.1)
        finally:
            self._ready = False
            self._loco = None

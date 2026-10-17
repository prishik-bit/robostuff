"""
File purpose: TLDR - Decides whether this robot is the goalie or the striker right now.

Description:
Both robots run a RoleManager and tell each other their role and whether they have the
ball over the radio heartbeat (field_model.pack_heartbeat role/has_ball). The swap:

  1. our goalie saves the ball and holds it on the dribbler for possession_confirm
     -> it claims STRIKER (it is the one with the ball, so it attacks)
  2. the old striker sees its mate claiming striker with the ball -> drops to GOALIE
  3. if both claim the same role for longer than conflict_timeout, a tie-break settles it:
     striker vs striker -> whoever has the ball keeps it, else the lower robot_id
     goalie  vs goalie  -> the lower robot_id becomes striker

Without the radio there is no swapping: both robots fall back to their default_role, so
one goal is always covered. A goalie with the ball then pushes it upfield and stays goalie.
"""

from typing import Optional

from armin_config import RoleConfig

GOALIE, STRIKER = 'goalie', 'striker'


class RoleManager:
    def __init__(self, config: RoleConfig = None, now: float = 0.0) -> None:
        self.config = config or RoleConfig()
        self.role = self.config.default_role
        self.changed_at = now
        self._ball_since = None
        self._conflict_since = None

    def _set(self, role: str, now: float, why: str) -> None:
        if role != self.role:
            print(f"[roles] {self.role} -> {role} ({why})")
            self.role = role
            self.changed_at = now
            self._conflict_since = None

    def update(self, now: float, have_ball: bool, mate_role: Optional[str],
               mate_has_ball: bool = False, mate_age: Optional[float] = None) -> str:
        """Call once per control tick. mate_role / mate_has_ball / mate_age come from the
        latest heartbeat (mate_age = seconds since it was received). Returns the role."""
        cfg = self.config
        if not have_ball:
            self._ball_since = None
        elif self._ball_since is None:
            self._ball_since = now

        radio_ok = mate_role is not None and mate_age is not None and mate_age < cfg.mate_stale
        if not radio_ok:
            self._set(cfg.default_role, now, 'no radio')
            return self.role

        # 1. goalie with the ball claims striker
        held = self._ball_since is not None and now - self._ball_since >= cfg.possession_confirm
        if self.role == GOALIE and held and now - self.changed_at >= cfg.swap_cooldown:
            self._set(STRIKER, now, 'goalie took possession')
            return self.role

        # 2/3. both robots in the same role
        if mate_role != self.role:
            self._conflict_since = None
            return self.role
        if self._conflict_since is None:
            self._conflict_since = now
        lower_id = cfg.robot_id < 3 - cfg.robot_id
        if self.role == STRIKER:
            if mate_has_ball and not have_ball:
                self._set(GOALIE, now, 'mate has the ball')            # yield at once
            elif (now - self._conflict_since >= cfg.conflict_timeout
                  and not (have_ball and not mate_has_ball) and not lower_id):
                self._set(GOALIE, now, 'tie-break: two strikers')
        elif now - self._conflict_since >= cfg.conflict_timeout and lower_id:
            self._set(STRIKER, now, 'tie-break: two goalies')
        return self.role

"""
Author: Aditya Gantimahapatruni
Date created: 18/9/2026

File purpose: TLDR - The file where all other modules are connected to each other.

Description:
This file is the main entry point for the robot application. 
It initializes the robot's configuration, state, vision, motors, and WebSocket controller.
It also sets up a console log tee to forward console output to the WebSocket clients.
"""

import asyncio
import math
import signal
import sys
import time
from pathlib import Path

import cv2
import websockets
from gpiozero import Button

from armin_config import RobotConfig, VisionConfig, TOFConfig
from armin_motors import MotorController
from armin_state import RobotState
from armin_vision import VisionService
from armin_websocket import WebSocketController
from armin_solenoid import Solenoid

class ConsoleLogTee:
    """Preserve console output while forwarding complete lines to the UI."""

    def __init__(self, console, on_line) -> None:
        self.console = console
        self.on_line = on_line
        self.pending = ''
    
    def write(self, text: str) -> int:
        """Every printed statement is sent to the console and also forwarded to the WebSocket clients."""
        self.console.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line:
                self.on_line(line)
        return len(text)

    def flush(self) -> None:
        self.console.flush()


class ArminApplication:
    """Compose robot services and coordinate camera, motor, and WebSocket work."""

    def __init__(self, config: RobotConfig = None) -> None:
        self.robot_config = config or RobotConfig()
        self.robot_state = RobotState()
        self.robot_vision = VisionService(self.robot_config.camera, self.robot_config.vision)
        self.robot_motors = MotorController(
            self.robot_config.motors,
            self.robot_config.vision,
            self.robot_state.control,
            self.robot_state,
            self.robot_config.control,
            self.robot_config.control.debug_print_hz,
        )
        self.robot_websocket = WebSocketController(
            self.robot_state,
            self.robot_config.vision,
            self.robot_vision,
            self.robot_motors,
        )
        self.robot_solenoid = Solenoid(
            self.robot_config.solenoid.GPIOpin, 
            self.robot_config.solenoid.active_time,
        )
        self.original_stdout = sys.stdout
        sys.stdout = ConsoleLogTee(sys.stdout, self.robot_websocket.record_log) # records smth for websockets
        self.mode_toggle_button = Button(
            self.robot_config.control.movement_switch_gpio
        )
        self.mode_toggle_button.when_pressed = lambda: self.toggle_mode("auto")
        self.mode_toggle_button.when_released = lambda: self.toggle_mode("manual")
        self._send_task = None
        self.tofs = TOFConfig()
        
        self.goal_toggle_switch = Button(
            self.robot_config.control.goal_switch_gpio,
            # bounce_time = 1
        )
        self.goal_toggle_switch.when_pressed = lambda: self.toggle_goal('yellow_goal')
        self.goal_toggle_switch.when_released = lambda: self.toggle_goal('blue_goal')
        self.last_goal_toggled_time : float = 0.0

    # def toggle_mode(self, mode: str | None = None) -> None:        
    #     if mode != None:
    #         if mode not in ['manual', 'auto']: raise ValueError('Please enter a valid robot control mode')
    #         self.robot_state.control.mode = mode
    #         return
        
    #     if self.robot_state.control.mode == 'manual':
    #         self.robot_state.control.mode = 'auto'
    #         self.robot_state.control.active_keys.clear()
    #     else:
    #         self.robot_state.control.mode = 'manual'
    #     return
    
    def toggle_goal(self, goal: str | None = None) -> None:
        if time.time() - self.last_goal_toggled_time < 2.0:
            return
        goals = self.robot_vision.config.goals
        target_goal = self.robot_vision.config.target_goal
        if goal is None:
            target_goal = goals[1] if target_goal == goals[0] else goals[0]
        else:
            if goal not in goals: raise ValueError("Please choose a valid goal to switch to")
            target_goal = goal
        print(f'Switched to {target_goal}')
        self.robot_vision.config.target_goal = target_goal
        return

    async def run(self) -> None:
        """Initialize hardware, serve the browser, and run until shutdown."""
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        self.robot_websocket.set_loop(asyncio.get_running_loop())
        self.robot_motors.setup()
        picam = self.robot_vision.setup_camera()
        self.toggle_mode('manual')
        # if self.goal_toggle_switch.is_active:
        #     self.toggle_goal('yellow_goal')
        # else:
        #     self.toggle_goal('blue_goal')
        server = await websockets.serve(
            self.robot_websocket.handle,
            '0.0.0.0',
            self.robot_config.network.websocket_port,
            compression=None
        )
        print(
            'WebSocket control + video on port '
            f"{self.robot_config.network.websocket_port} — open control.html to drive."
        )
        print('Ctrl+C to stop.')
        try:
            await asyncio.gather(
                self.stream_camera(picam),
                self._manual_loop(),
            )
        finally:
            server.close()
            await server.wait_closed()
            self.robot_motors.stop()
            sys.stdout = self.original_stdout

    async def stream_camera(self, picam) -> None:
        PREVIEW_EVERY = 3   # 60 fps camera -> 20 fps preview
        t_report = time.perf_counter()
        loops = sent = skipped = 0
        proc = enc_t = 0.0
        kb = 0.0

        while self.robot_state.is_running:
            t0 = time.perf_counter()
            frame, ball_offset, goal_offset = await asyncio.to_thread(
                self.robot_vision.process_frame, picam,
            )
            t1 = time.perf_counter()
            now = time.time()

            # Only stamp last_seen on a real detection so staleness checks stay honest.
            self.robot_state.ball.offset = ball_offset
            if ball_offset is not None:
                self.robot_state.ball.last_seen = now
            self.robot_state.goal.offset = goal_offset
            if goal_offset is not None:
                self.robot_state.goal.last_seen = now
            self._print_requested_vector()

            loops += 1
            proc += t1 - t0

            # Preview: only every Nth frame, only with a client, only if the last send finished.
            if self.robot_state.clients and loops % PREVIEW_EVERY == 0:
                if self._send_task is None or self._send_task.done():
                    ok, enc = await asyncio.to_thread(self._encode_preview, frame)
                    enc_t += time.perf_counter() - t1
                    if ok:
                        payload = enc.tobytes()
                        self._send_task = asyncio.create_task(
                            self.robot_websocket.broadcast(payload)
                        )
                        sent += 1
                        kb += len(payload) / 1000
                else:
                    skipped += 1

            t3 = time.perf_counter()
            if t3 - t_report >= 1.0:
                # print(f"[perf] {loops} fps | process {proc/loops*1000:.1f} ms | "
                #     f"encode {enc_t/max(sent,1)*1000:.1f} ms | sent {sent} skipped {skipped} | "
                #     f"{kb/max(sent,1):.0f} KB/frame | {kb*8/1000:.1f} Mbit/s")
                t_report = t3
                loops = sent = skipped = 0
                proc = enc_t = kb = 0.0

            await asyncio.sleep(0)
    def take_photo(self, frame):        
        if not self.robot_state.control.take_photo: 
            return
        self.robot_state.control.take_photo = False
        PHOTOS_DIR = Path("Photos")

        n = 0
        
        for item in PHOTOS_DIR.iterdir():
            if item == None: break
            if item.name[:5] == 'calib':
                n += 1
        
        cv2.imwrite(f"Photos/calib_{n + 1}.png", frame)
        return
    
    def _print_requested_vector(self) -> None:
        if not self.robot_state.control.print_vector_requested:
            return
        self.robot_state.control.print_vector_requested = False
        offset = self.robot_state.ball.offset
        if offset is None:
            print('[v] No ball currently detected — no vector to print.')
            return
        dx, dy = offset
        distance = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dx, -dy)) % 360
        print(
            f'[v] Ball vector: dx={dx}px, dy={dy}px, '
            f'distance={distance:.1f}px, bearing={bearing:.1f}deg'
        )

    @staticmethod
    def _encode_preview(frame):
        small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])

    @staticmethod
    def _bearing_from_offset(offset, camera_rotation_offset: float) -> float:
        """Convert an image offset into the robot-frame bearing convention."""
        offset_x, offset_y = offset
        return (
            math.degrees(math.atan2(offset_x, -offset_y))
            + camera_rotation_offset
        ) % 360

    async def _manual_loop(self) -> None:
        """Apply manual commands or ball-following decisions at a fixed interval."""
        while self.robot_state.is_running:
            now = time.time()
            control = self.robot_state.control
            self.robot_motors.spin_dribbler(True)
            if control.mode == 'manual':
                timeout = self.robot_config.control.key_lost_timeout
                stale = (
                    not control.active_keys
                    or now - control.keys_last_seen > timeout
                )
                if stale: # Stale if never seen this frame or the last detection is too old.
                    self.robot_motors.debug('[manual] no keys held or stale -> stop()')
                    self.robot_motors.stop()
                else:
                    self.robot_motors.apply_manual_keys(control.active_keys)
            else:
                if self.robot_state.is_goalie: 
                    self.apply_goalie(now)
                else:
                    self.apply_striker(now)
            await asyncio.sleep(self.robot_config.control.motor_loop_delay)

    def apply_striker(self, now: float) -> None:
        """Turn fresh ball/goal observations into an offensive play."""
        state = self.robot_state
        vision = self.robot_config.vision
        motors = self.robot_motors
        ball, goal = state.ball, state.goal
        timeout = vision.ball_lost_timeout
        solenoid = self.robot_solenoid
        tofchain = self.tofs
        
        # ! Outstanding bug - this is code below is now redundant because I have bypassed 
        # ! it figure out search logic later.
        # Find the ball and the goal
        if ball.offset is None and now - ball.last_seen > timeout:
            ball_vector = None
        else:
            ball_vector = ball.offset

        if goal.offset is None and now - goal.last_seen > timeout:
            goal_vector = None
        else:
            goal_vector = goal.offset

        ball_bearing, ball_distance = None, None
        goal_bearing, goal_distance = None, None
        
        if ball_vector:
            ball_distance = math.hypot(*ball.offset)
            ball_bearing = self._bearing_from_offset(
                ball.offset,
                vision.camera_rotation_offset,
            )
            vision.last_ball_vector = ball_vector
            vision.last_ball_distance = ball_distance
        
        if goal_vector:
            goal_distance = math.hypot(*goal.offset)
            goal_bearing = self._bearing_from_offset(
                goal.offset,
                vision.camera_rotation_offset,
            )
            vision.last_goal_vector = goal_vector
            vision.last_goal_distance = goal_distance
        
        if vision.last_goal_vector is not None:
            if goal_vector is None or goal_distance is None:
                goal_vector = vision.last_goal_vector
                goal_bearing = self._bearing_from_offset(goal_vector, vision.camera_rotation_offset)
                goal_distance = vision.last_goal_distance
            
        if vision.last_ball_vector is not None:    
            if ball_vector is None or ball_distance is None:
                ball_vector = vision.last_ball_vector
                ball_bearing = self._bearing_from_offset(ball_vector, vision.camera_rotation_offset)
                ball_distance = vision.last_ball_distance
        
        if goal_vector is None and ball_vector is not None:
            motors.drive_to_the_ball(True, ball_bearing, ball_distance)
        
        if ball_vector is None:
            motors.spin(motors.config.max_speed * 0.3)
               
        # Normalizes goal bearing before calculating easing

        if goal_bearing != None:
            rotation_ease = (((goal_bearing + 180) % 360) - 180) / 180
        else:
            rotation_ease = 0
        rotation_speed = rotation_ease * (motors.config.max_speed * 2)
        
        rotation_ease = math.floor(rotation_ease)
        rotation_speed = math.floor(rotation_speed)
        
        # Possession is latched: a ball held in the dribbler can sit outside the
        # bot mask and vanish, so it only clears when the ball is seen escaping
        # beyond the capture (orbit) radius.
        
        for i in tofchain.TOF_CHAIN_1:
            pass
        
        if state.has_possession and (ball_distance > vision.ball_dribble_radius):
            state.has_possession = False
            state.has_orbited = False

        if state.has_possession: # Maintain posession
            print(goal_distance)
            if goal_vector:
                if goal_distance < vision.goal_stop_distance:
                    motors.spin_dribbler(False)
                    time.sleep(0.4)
                    solenoid.kick()
                    motors.stop()
                motors.spin_dribbler(True)
                motors.drive_to_goal(goal_bearing, rotation_speed)
            else:
                motors.spin_dribbler(True)
                motors.spin(motors.config.max_speed * 0.5)
            return
        
        if state.has_orbited and not state.has_possession:
            # target_bearing = goal_bearing - ball_bearing
            # if motors.spin_to_bearing(target_bearing, 3):
            motors.spin_dribbler(True)
            motors.rotate_and_move(ball_bearing, motors.config.max_speed, rotation_speed)
            if ball_distance <= vision.ball_dribble_radius:
                state.has_possession = True
        
        elif ball_distance > vision.orbit_radius:
            if ball_distance - vision.orbit_radius > 10: # Orbiting a little early
                motors.orbit_to_behind_ball()            
            motors.rotate_and_move(ball_bearing, motors.config.max_speed, rotation_speed)
            state.has_orbited = False
        
        elif ball_distance > vision.ball_dribble_radius or ball_bearing != goal_bearing - motors.config.goal_align_tolerance_degrees:
            # Line the ball up with the goal so driving at the ball pushes it goalward.
            target_bearing = goal_bearing
            if motors.orbit_to_behind_ball(target_bearing, rotation_speed):
                state.has_orbited = True
                return
            
        else:
            state.has_possession = True
            motors.spin_dribbler(True)

    def _on_sigint(self) -> None:
        print('\nShutting down...')
        self.shutdown()   # sets is_running=False and stops motors

    def shutdown(self) -> None:
        """Stop future loops and remove motor output immediately."""
        self.robot_state.is_running = False
        self.robot_motors.stop()


def main() -> None:
    application = ArminApplication()
    try:
        asyncio.run(application.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
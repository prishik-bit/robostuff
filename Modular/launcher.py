import signal
import subprocess
import sys
from gpiozero import Button

PYTHON = "/home/aditya/???/env/bin/python"   # output of `which python` inside (env)
WORKDIR = "/home/aditya/TheTitans/Modular"
LAUNCH_PIN = 26   # free GPIO: not 2/3 (I2C), not 26, not your goal-switch pin

proc = None

def start():
    global proc
    if proc is not None and proc.poll() is None:
        return                      # already running, ignore bounce
    proc = subprocess.Popen([PYTHON, "-u", "armin_main.py"], cwd=WORKDIR)
    print("robot started")

def stop():
    global proc
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)  # your handler zeroes the motors on SIGINT
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()                  # last resort: motors may keep their last speed!
    print("robot stopped")

def on_term(*_):
    stop()
    sys.exit(0)

signal.signal(signal.SIGTERM, on_term)   # systemd stop / shutdown

switch = Button(LAUNCH_PIN)              # wire the switch between this pin and GND
switch.when_pressed = start
switch.when_released = stop
if switch.is_pressed:                    # switch already ON at boot
    start()

signal.pause()
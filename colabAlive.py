import time
from pynput.mouse import Controller ,Button

MouseClick = Controller()

while True:

    MouseClick.click(Button.left, 1)
    time.sleep(5)
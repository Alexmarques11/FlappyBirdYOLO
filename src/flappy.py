import asyncio
import sys
import cv2
import pygame
from ultralytics import YOLO
import time
import threading
import numpy as np
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.cap = cv2.VideoCapture(0)
        self.phone_class_index = 67

        self.model = YOLO("yolov8n.pt")
        print("Known classes ({})".format(len(self.model.names)))
        for i in range(len(self.model.names)):
            print("{} : {}".format(i, self.model.names[i]))

        self.show_camara = True
    async def start(self):

        threading.Thread(target=self.segmentation_thread, daemon=True).start()

        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)

            await self.splash()
            await self.play()
            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def segmentation_thread(self):
        cv2.namedWindow("Image")
        last_frame_timestamp = time.time()

        while True:
            begin_time_stamp = time.time()

            try:
                framerate = 1 / (begin_time_stamp - last_frame_timestamp)
            except ZeroDivisionError:
                framerate = 0

            last_frame_timestamp = begin_time_stamp

            if not self.cap.isOpened():
                self.cap.open(0)
            _, image = self.cap.read()
            image = image[:, ::-1, :]
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

            results = self.model(image, verbose=False)

            image_objects = image.copy()

            objects = results[0]
            for object in objects:
                box = object.boxes.data[0]
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                confidence = box[4]
                class_id = int(box[5])

                if class_id == self.phone_class_index:

                    self.player.update_position((pt1[1] + pt2[1]) / 2)

                    cv2.rectangle(img=image_objects, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)
                    text = "{}:{:.2f}".format(objects.names[class_id], confidence)
                    cv2.putText(img=image_objects,
                                text=text,
                                org=np.array(np.round((float(box[0]), float(box[1] - 1))), dtype=int),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 255, 0),
                                thickness=1)

            image_objects = cv2.cvtColor(src=image_objects, code=cv2.COLOR_RGB2BGR)
            cv2.imshow(winname="Image", mat=image_objects)

            c = cv2.waitKey(delay=1)
            if c == 27:
                break

    def check_quit_event(self, event):
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    def __del__(self):
        self.cap.release()
#!/usr/bin/env python3

"""
This is a custom script developed by FRANK based on duckietown 
joystick script in order to allow user drive duckietown with joystick
and obtain log for further training.
"""

import argparse
import json
import sys
import cv2

import gym
import numpy as np
import pyglet
import math
from _loggers import Logger
import logging

from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

from distortion import Distortion

#! Camera Distorters
distorter = Distortion()

#! Logger setup:
logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.WARNING)

#! Parser sector:
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='zigzag_dists')
parser.add_argument('--draw-curve', default=True, action='store_true',
                    help='draw the lane following curve')
parser.add_argument('--draw-bbox', default=False, action='store_true',
                    help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', default=False, action='store_true',
                    help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int,
                    help='number of frames to skip')
args = parser.parse_args()


#! Start Env
if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        max_steps=math.inf,

        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=0,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,

    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

#! Recorder Setup:
# global variables for demo recording
actions = []
observation = []
datagen = Logger(env, log_file='training_data.log')


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        datagen.on_episode_done()
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
        env.render()
    elif symbol == key.ESCAPE or symbol == key.Q:
        env.close()
        sys.exit(0)


@env.unwrapped.window.event
def on_joybutton_press(joystick, button):
    """
    Event Handler for Controller Button Inputs
    Relevant Button Definitions:
    3 - Y - Resets Env.
    """

    # Y Button
    if button == 3:
        print('RESET')
        datagen.on_episode_done()
        env.reset()
        env.render()


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global actions, observation

    #! Joystick deadband
    if round(joystick.x, 2) == 0 and round(joystick.ry, 2) == 0:
        return

    #! Nominal Joystick Interpretation
    x = round(joystick.y, 2)*0.3
    z = round(joystick.rx, 2)

    #! DRS enable for straight line
    if joystick.buttons[5]:
        x *= 5

    #! Break peddle:


    action = np.array([-x, -z])

    #! GO! and get next
    obs, reward, done, info = env.step(action)

    #! Distort image for storage
    obs = distorter.distort(obs)


    print('Current [Linear, Angular]: ', action, ' speed. Score: ',reward)

    #! ADD IMAGE-PREPROCESSING HERE!!!!!
    observation = cv2.resize(obs, (80, 60))
    # NOTICE: OpenCV changes the order of the channels !!!
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

    cv2.imshow('Whats logged',observation)
    cv2.waitKey(1)

    datagen.log(observation, action, reward, done, info)

    env.render()

#! Enter main event loop
pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
#! Get Joystick
# Registers joysticks and recording controls
joysticks = pyglet.input.get_joysticks()
assert joysticks, 'No joystick device is connected'
joystick = joysticks[0]
joystick.open()
joystick.push_handlers(on_joybutton_press)
pyglet.app.run()

#! Log and exit
datagen.close()
env.close()

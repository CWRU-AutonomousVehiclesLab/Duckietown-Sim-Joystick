import os
import pickle
import numpy as np
import cv2
class SteeringToWheelVelWrapper:
    """ Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, wheel_dist=0.102):
        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        # Distance between wheels
        self.wheel_dist = wheel_dist

    def convert(self, vel, angle):

        # Distance between the wheels
        baseline = self.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        vels = np.array([u_l, u_r])

        return vels
convertion_wrapper = SteeringToWheelVelWrapper()


class Reader:

    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        end = False
        observations = []
        actions = []

        while not end:
            try:
                log = pickle.load(self._log_file)
                for entry in log:
                    step = entry['step']
                    action = step[1]
                    x = action[0]
                    z = action[1]
                    pwm_left,pwm_right =convertion_wrapper.convert(x, z)
                    print('Current Motor Speed: ',pwm_left,'&&',pwm_right)
                    canvas = cv2.resize(step[0], (640,480))
                    #! Speed bar indicator
                    cv2.rectangle(canvas, (20, 240), (50, int(240+220*x)),
                                  (76, 84, 255), cv2.FILLED)
                    cv2.rectangle(canvas, (320, 430), (int(320+300*z), 460),
                                  (76, 84, 255), cv2.FILLED)

                    cv2.imshow('Playback', canvas)
                    cv2.waitKey(100)
                # TODO: Add selection
            except EOFError:

                end = True

        return observations, actions

    def close(self):
        self._log_file.close()


reader = Reader('raw_log.log')
reader.read()

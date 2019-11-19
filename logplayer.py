import os
import pickle
import numpy as np
import cv2
import csv
from pwmcalculator import SteeringToWheelVelWrapper
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
                    with open('distribution.csv','a') as newFile:
                        newFileWriter = csv.writer(newFile)
                        newFileWriter.writerow([x,z,pwm_left,pwm_right])
                        
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

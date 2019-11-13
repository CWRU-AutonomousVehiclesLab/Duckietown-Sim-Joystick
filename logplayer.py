import os
import pickle
import numpy as np
import cv2
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
                    observations.append(step[0])
                    actions.append(step[1])
            
                    #! Playback
                    val =np.array_str(step[1])
                
                    observation = cv2.resize(step[0], (80, 60))
                    # NOTICE: OpenCV changes the order of the channels !!!
                    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)

                    cv2.imshow(val,observation)

                    cv2.waitKey(10)

            except EOFError:
            
                end = True

        
        
        return observations, actions

    def close(self):
        self._log_file.close()

reader = Reader('training_data.log')
reader.read()
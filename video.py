import cv2
import numpy
import sys

class sequence(object):
    def __init__(self):
        self.pos_ini = 0
        self.pos_fin = 0
        self.step    = 1
        self.current = 0

    def increment(self):
        self.current = self.current + self.step  

    def decrement(self):
        self.current = self.current - self.step  
              
################################################################################
################################################################################
class video_sequence(sequence):
    """ video sequence class 
       for opencv video  
    """
    def __init__(self, video_file, ini = 0, fin = sys.maxsize):
        super(video_sequence, self).__init__()

        self.cap        = cv2.VideoCapture(video_file)
        self.height     = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width      = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.pos_ini    = ini
        self.current    = ini
        
        last_frame      = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        #validating the inputs
        if( fin > int(last_frame) ):
            self.pos_fin    = last_frame
        else:
            self.pos_fin    = fin

        #setting the first position
        if( ini > 0 ):
            self.setCurrent(self.current)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #set a new frame position
    def setCurrent(self, pos):
        self.current = pos - ( pos % self.step )
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

################################################################################
################################################################################
class video_sequence_by1(video_sequence):
    """ video sequence class step = 1
    """
    def __init__(self, video_file, ini = 0, fin = sys.maxsize):
        super().__init__(video_file, ini, fin)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #return the current position frame
    def getCurrent(self):
        if self.current > self.pos_fin: 
            return (False, [])
        ans = self.cap.read()
        self.increment()
        return ans

################################################################################
################################################################################
class video_sequence_byn(video_sequence):
    """ video sequence class step = 1
    """
    def __init__(self, video_file, step, ini = 0, fin = sys.maxsize):
        super().__init__(video_file, ini, fin)
        self.step = step
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #return the current position frame
    def getCurrent(self):
        self.increment()
        if self.current > self.pos_fin: 
            return (False, [])
        self.setCurrent(self.current)
        return self.cap.read()
 


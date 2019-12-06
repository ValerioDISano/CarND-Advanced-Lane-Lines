class PerspectiveTransformer(object):

    def __init__(self, nx, ny, offset):
        
        self.offset = None
        self.img_shape = None

    def warpPerspective(self, img):

        self._setDestPoints(img)

    

    def _setDestPoints(self, img):
        
        if (self.img_size is None) or (img.shape != self.img_shape):
            self.dst = np.float32([])
            
        return

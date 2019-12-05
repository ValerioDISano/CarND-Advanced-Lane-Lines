import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

class DataLoader(object):

    def __init__(self, folder_path):
        self.file_list = [os.path.join(folder_path,x) for x in os.listdir(folder_path)]
        self.loaded_data = None
        self.current_idx = int(-1)
        self.n_file = len(self.file_list)

    def getFileList(self):
        return file_list

class ImagesLoader(DataLoader):

    def __init__(self, folder_path):
        super(ImagesLoader, self).__init__(folder_path)

    def getNextImg(self):
        self.current_idx += 1
        self.current_idx %= self.n_file

        self.loaded_data = mpimg.imread(\
                            self.file_list[self.current_idx]\
                            )
        return self.loaded_data 

    def showImg(self, idx):
        
        try:
            plt.figure()
            plt.imshow(self.getNextImg())
            plt.show()
        except:
            print("Index out of boundaries")
            sys.exit(1)

    def showAll(self):
        print("Number of images to print: {}".format(self.n_file))
        [self.showImg(x) for x in range(self.n_file)]
        return
        
    def dataIterable(self):
        for _ in range(self.n_file):
            yield self.getNextImg()

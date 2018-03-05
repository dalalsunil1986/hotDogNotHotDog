import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class PlotImages(object):
    def __init__(self):
        pass

    def imagesToNumpy(self, path_to_folder, limit=None):
        """Get images in afolder and turns them in a list of numpy array."""
        numpyArray = []

        if os.path.isfile(path_to_folder):
            print('it is a file')
            numpyArray.append(cv2.imread(path_to_folder))
        else:
            for root, dirs, files in os.walk(path_to_folder, topdown=False):
                if files:
                    if limit is None or limit > len(files):
                        limit = len(files)

                    idx = 0
                    while idx < limit:
                        imagePath = os.path.join(root, files[idx])
                        numpyArray.append(cv2.imread(imagePath))
                        idx += 1
        return numpyArray

    def show_images(self, images, rows = 3, titles = None):
        """Display a list of images in a single figure with matplotlib.
        
        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.
        
        rows (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(rows))).
        
        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        print('length of images:', n_images)
        
        if titles is None:
            titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()

        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

# def main():
#     plotThis = PlotImages() 
#     imagesToPlot = plotThis.imagesToNumpy('videoFrames/video1', 10)
#     plotThis.show_images(imagesToPlot)

# if __name__ == '__main__':
#     main()
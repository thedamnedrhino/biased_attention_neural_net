import pickle
import matplotlib.pyplot as plt


IMAGES = None
LABELS = None

class Image:
    def __init__(self, pixels, label, channels=3, rows=32, cols=32):
        self.pixels = pixels
        self.label = label
        self.shaped_img = None
        self.channels = channels
        self.rows = rows
        self.cols = cols
    
    def show(self, p=True):
        plt.imshow(self.shaped())
        plt.show()
        if p:
            print("LABEL: ++ {} ++\n".format(self.label))
        return plt
        
    def shape(self, merged=True):
        img = self.pixels
        assert len(img) == 1024 * 3
        channels = [img[i*1024:(i+1)*1024] for i in range(3)]
        channels = [ [channel[row*32:(row+1)*32] for row in range(32)] for channel in channels ]
        assert len(channels) == 3
        assert len(channels[0]) == 32
        assert len(channels[2][2]) == 32
        if merged:
            channels = [[ [channels[0][i][j], channels[1][i][j], channels[2][i][j]] for j in range(32)] for i in range(32)]
        return channels
                
       
    def shaped(self):
        if self.shaped_img is None:
            self.shaped_img = self.shape()
        return self.shaped_img
    
def get_data(datadir='./datasets', dataset='train'):
    """
    param dataset: 'train', 'valid'
    """
    filename = datadir + '/' + dataset + 'set.pickle'
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        data = dict['data']
        labels = dict['label']
    global IMAGES
    images = [Image(data[i], labels[i]) for i in range(len(data))]
    IMAGES = images if IMAGES is None else IMAGES
    return images
                
def init():
    get_data()

if __name__ == '__main__':
    init()
    images = sorted(IMAGES, key=lambda x: x.label)
    igs = images
    def s(n):
        x = igs[n].show()
    
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = np.random.randn(512, 512, 3)
    plt.imshow(img)
    plt.plot()

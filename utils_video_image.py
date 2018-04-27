import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def cv2ShowWait(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

################################################################################
################################################################################
def pil2matRGB(pil_i, show=False):
    open_cv_image = np.array(pil_i) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    if show == True :
        cv2ShowWait(open_cv_image)
    return open_cv_image

################################################################################
################################################################################
def mat2pil(cv_i, show=False):
    cv2_im = cv2.cvtColor(cv_i,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    if show == True :
        image.show()
    return image

################################################################################
################################################################################
def plot_chart(imgs, x, y, fig_sze = 1, ax_vis = True, gray = False,
               title = '',
               titles = []):
    
    #fig_sze -> (20,4)
    fig = plt.figure(fig_sze)
    fig.suptitle(title, fontsize=24, fontweight='bold')
    
    pos = 1
    col = 1
    row = 1

    if len(imgs) != len(titles):
        titles = ['' for  x in range(len(imgs)) ]

    for img in imgs:
        ax = plt.subplot(x, y, pos)
        if gray: plt.gray()
        plt.imshow(img)
        pos += 1
        ax.get_xaxis().set_visible(ax_vis)
        ax.get_yaxis().set_visible(ax_vis)
        ax.set_title(titles[pos-2])

    plt.show()

################################################################################
################################################################################
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
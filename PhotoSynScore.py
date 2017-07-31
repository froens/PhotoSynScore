import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import gc
import httplib
import cv2

def readImageFromServer(IMG_URL):
    conn = httplib.HTTPConnection(IMG_URL)
    conn.request("GET", "/")
    response = conn.getresponse()
    return response.read()

def alignImagesECC(img_vis, img_ir):
    img_vis = np.array(img_vis)
    img_ir = np.array(img_ir)

    #im1_gray = cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
    #im2_gray = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = img_vis.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(img_vis, img_ir, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(img_ir, warp_matrix, (sz[1], sz[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(img_ir, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    return im2_aligned
    #cv2.imshow("Image 1", img_vis)
    #cv2.imshow("Image 2", img_ir)
    #cv2.imshow("Aligned Image 2", im2_aligned)

# function for generating NDVI imagery from NGB or NBG input files
def photosynscore(VIS_IMG_URL, IR_IMG_URL):

    vis_img_temp_file = 'VIS.data'
    ir_img_temp_file = 'IR.data'

    img_data_vis = readImageFromServer(VIS_IMG_URL)
    img_vis = Image.frombuffer('RGB', (1024, 768), img_data_vis, "raw")
    f_vis = open(vis_img_temp_file, "w+")
    f_vis.write(img_data_vis)
    f_vis.close()

    img_data_ir = readImageFromServer(IR_IMG_URL)
    img_ir = Image.frombuffer('RGB', (1024, 768), img_data_ir, "raw")
    f_ir = open(ir_img_temp_file, "w+")
    f_ir.write(img_data_ir)
    f_ir.close()

    imgR_vis, _, _ = img_vis.split() #get channels
    imgR_ir, _, _ = img_ir.split()  #get channels
    del img_ir
    del img_vis

    imgR_ir_aligned = alignImagesECC(imgR_vis, imgR_ir)

    #showNDVI(imgR_vis, imgR_ir)
    showNDVI(imgR_vis, Image.fromarray(imgR_ir_aligned))
    #showNDVI(imgR_vis, imgR_ir)
    i = 1
    #plt.imshow(imgR_vis)
    #plt.imshow(imgR_ir)


def showNDVI(imgR_vis, imgR_ir):

    img_w, img_h = imgR_ir.size

    #compute the NDVI
    arrR_ir = np.asarray(imgR_ir).astype('float64')
    arrR_vis = np.asarray(imgR_vis).astype('float64')

    num   = (arrR_ir - arrR_vis)
    denom = (arrR_ir + arrR_vis)
    del arrR_ir
    del arrR_vis

    with np.errstate(divide='ignore', invalid='ignore'):
        arr_ndvi = np.true_divide(num,denom)
        arr_ndvi[arr_ndvi == np.inf] = 0
        arr_ndvi = np.nan_to_num(arr_ndvi)

        # Needs to be floating point
        colormap = plt.cm.spectral #plt.cm.gist_gray
        dpi=600.0
        fig_w=img_w/dpi/8
        fig_h=img_h/dpi/8
        fig=plt.figure(figsize=(fig_w,fig_h),dpi=dpi)
        fig.set_frameon(False)

        ax_rect = [0.0, #left
                   0.0, #bottom
                   1.0, #width
                   1.0] #height

        ax = fig.add_axes(ax_rect)
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticklabels([])
        ax.set_axis_off()
        ax.axes.get_yaxis().set_visible(False)
        ax.patch.set_alpha(0.0)
        print arr_ndvi.min()
        print arr_ndvi.max()
        axes_img = ax.imshow(arr_ndvi, cmap=colormap, vmin=-1, vmax=0.9, aspect='equal', interpolation="nearest")
        del axes_img
        #fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.0, )
        del fig

    # threshold = arr_ndvi[ numpy.where(arr_ndvi>=vmin) ]
    # del arr_ndvi
    # normalized = numpy.multiply(numpy.add(threshold,1.0),1/2.0)
    # del threshold
    # print(numpy.median(normalized)*100.0)
    gc.collect()

###### testing the code #######
if __name__ == "__main__":

    if len(sys.argv)<2:
        print("python photosynscore.py VIS_SERVER_URL IR_SERVER_URL [output.png]")
        sys.exit(1)
    output=None
    if len(sys.argv)==3:
        output=sys.argv[2]
    photosynscore(sys.argv[1], sys.argv[2])
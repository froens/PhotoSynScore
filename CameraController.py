#!/usr/bin/python
from threading import Lock
from flask import Flask
from flask_restful import Resource, Api
from flask import send_file
from shutil import copyfile
import http

# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import numpy as numpy
from PIL import Image
import gc
import http.client
import io
import urllib
import imghdr

import gdal
from gdal import Open
from ndvi import ndvi

# import picamera

app = Flask(__name__)
api = Api(app)
camLock = Lock()
tmp_file_name = "cam_temp.data"
mode_vis_str = "VIS"
mode_ir_str = "IR"
self_url = "http://127.0.0.1:5000/IR"
cam_str = "cam"
ndvi_str = "ndvi"


class Cammock:

    __src = ""
    __img_type = ""

    def __init__(self, srcfilename):
        self.__src = srcfilename

    def capture(self, filename, img_type):
        self.__img_type = img_type
        copyfile(self.__src, filename)

    def close(self):
        a = ""

class PiCameraController(Resource):

    __mode = ""

    def __init__(self, mode):
        self.__mode = mode

    def get(self):
        attachFileName = self.__mode + '.data'
        with camLock:
            # camera = picamera.PiCamera()
            camera = Cammock(attachFileName)
            camera.resolution = (1024, 768)
        try:
            camera.capture(tmp_file_name, 'rgb')

            return send_file(tmp_file_name,mimetype='image/x-rgb', attachment_filename=attachFileName , as_attachment=True)
        finally:
            camera.close()

class NDVIController(Resource):

    __irURL = ""
    __visURL = ""

    def __init__(self, visURL, irURL):
        self.__visURL = visURL  # + "/" + cam_str
        self.__irURL = irURL  # + "/" + cam_str

    def get(self):
        photosynscore(self.__visURL, self.__irURL)

def readImageFromServer(IMG_URL):
    conn = http.client.HTTPConnection(IMG_URL)
    conn.request("GET", "/")
    response = conn.getresponse()
    return response.read()

# function for generating NDVI imagery from NGB or NBG input files
def photosynscore(VIS_IMG_URL, IR_IMG_URL, vmin = -1.0, vmax = 1.0, output= 'plt.png'):

    vis_img_temp_file = 'vis_tmp.data'
    ir_img_temp_file = 'ir_tmp.data'
    type = imghdr.what(ir_img_temp_file)

    with urllib.request.urlopen(VIS_IMG_URL) as url:
        vis_img_data = url.read()

    with urllib.request.urlopen(IR_IMG_URL) as url:
        ir_img_data = url.read()

    # f = open(vis_img_temp_file, 'wb')
    # f.write(vis_img_data)
    #
    # f = open(ir_img_temp_file, 'wb')
    # f.write(ir_img_data)

    # if isinstance(vis_img_temp_file,str):  # treat as a filename
    byteSteam = io.BytesIO(vis_img_data)
    byteSteam.seek(0)
    img_vis = Image.frombuffer('RGB',(1024,768),vis_img_data)

    # if isinstance(ir_img_temp_file,str):  # treat as a filename
    byteSteam = io.BytesIO(ir_img_data)
    byteSteam.seek(0)
    img_ir = Image.frombuffer('RGB',(1024,768),ir_img_data)


    # create the matplotlib figure
    img_w,img_h=img_vis.size

    # imgV, _, _ = img_vis.split() # get channels
    imgI, _, imgV = img_ir.split()  # get channels
    # del img_vis, img_ir

    # compute the NDVI
    arrV = numpy.asarray(imgV).astype('float64')
    arrI = numpy.asarray(imgI).astype('float64')

    plt.imshow(imgV)
    plt.imshow(imgI)

    num   = (arrI - arrV)
    denom = (arrI + arrV)
    del arrV
    del arrI

    with numpy.errstate(divide='ignore', invalid='ignore'):
        arr_ndvi = numpy.true_divide(num,denom)
        arr_ndvi[arr_ndvi == numpy.inf] = 0
        arr_ndvi = numpy.nan_to_num(arr_ndvi)
    if output!=None:
        # Needs to be floating point
        colormap = plt.cm.spectral #plt.cm.gist_gray
        dpi=600.0
        fig_w=img_w
        fig_h=img_h
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
        axes_img = ax.imshow(arr_ndvi,
                         cmap=colormap,
                         vmin = vmin,
                         vmax = vmax,
                         aspect = 'equal',
                         interpolation="nearest"
                        )
        del axes_img
        fig.savefig(output,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0.0,
                )
        del fig

    threshold = arr_ndvi[ numpy.where(arr_ndvi>=vmin) ]
    del arr_ndvi
    normalized = numpy.multiply(numpy.add(threshold,1.0),1/2.0)
    del threshold
    print(numpy.median(normalized)*100.0)
    gc.collect()

def main():

    mode = ""
    if len(sys.argv) <= 1:
        print("Missing argument Mode")
        return

    mode = sys.argv[1]

    if ((mode != mode_ir_str) & (mode != mode_vis_str)):
        print("Mode is not valid. Modes available: IR, VIS")
        return

    api.add_resource(PiCameraController, '/VIS', endpoint="viscam", resource_class_kwargs={'mode': "VIS"})
    api.add_resource(PiCameraController, '/IR', endpoint="ircam", resource_class_kwargs={'mode': "IR"})

    slaveURL = ""
    if (len(sys.argv)) > 2:
        slaveURL = sys.argv[2]
        # try:
        #     conn = http.client.HTTPConnection(slaveURL)
        #     conn.request("GET", "/")
        # except:
        #     print("URL is not working")
        #     return

    if (slaveURL != ""):
        if(mode == mode_ir_str):
            api.add_resource(NDVIController, '/ndvi', resource_class_kwargs={'visURL': slaveURL, 'irURL': self_url})
        if(mode == mode_vis_str):
            api.add_resource(NDVIController, '/ndvi', resource_class_kwargs={'visURL': self_url, 'irURL': slaveURL})



    app.run(host='127.0.0.1', debug=True, threaded=True)


if __name__ == '__main__':
    main()

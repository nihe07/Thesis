from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse

import numpy as np
import scipy.spatial.distance as dis
import scipy
import sqlite3
from sqlite3 import Error

import PIL as pil
from PIL import Image 
import scipy.cluster
from io import BytesIO
import requests

import ast
import cv2

from .models import *
from .forms import ImageForm

import io
from google.cloud import vision

import os, json, boto3, base64

import time

import multiprocessing as mp
from multiprocessing import Process, Queue, Pool
from functools import partial

# new dependencies
from google_images_search import GoogleImagesSearch

import threading

from threading import Thread


#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## GLOBAL VARIABLES
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
imurls = []
imcols = []
METdone = False


#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## USER IMAGE UPLOAD
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})



#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## MAIN RETURN FUNCTIONS
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def return_palinim(request):
  if request.method == 'GET':
    data = []

    imsrc = request.GET['imgsrc']
    pal = request.GET['palette']

    loc = 'media' + imsrc.split('media', 1)[1] 
    img = pil.Image.open(loc)
    img = img.convert('RGB')

    img = img.resize((200, int((img.height * 200) / img.width) ))

    pal = np.array(json.loads(pal))
    palinim = pal_in_image(pal, img)

    data.append(palinim)
  return JsonResponse(data, safe=False)


def return_hue(request):
  if request.method == 'GET':
    data = []

    imsrc = request.GET['imgsrc']
    val = int(request.GET['val'])
    loc = 'media' + imsrc.split('media', 1)[1] 
    img = pil.Image.open(loc)
    img = img.convert('RGB')
    img = img.resize((200, int((img.height * 200) / img.width) ))
    
    hue_im = hueim(img, val)
    data.append(hue_im)

  return JsonResponse(data, safe=False)

def return_pix(request):
  if request.method == 'GET':
    data = []

    imsrc = request.GET['imgsrc']
    val = int(request.GET['val'])
    loc = 'media' + imsrc.split('media', 1)[1] 
    img = pil.Image.open(loc)
    img = img.convert('RGB')
    img = img.resize((200, int((img.height * 200) / img.width) ))
    
    pix_im = pixelate(img, val)
    data.append(pix_im)

  return JsonResponse(data, safe=False)


def returnWordsearch(request):
  if request.method == 'GET':
    data = []
    word = request.GET['word']
    imurls, imcols = MET_word_search(word)
    data.append(imurls) 
    data.append(imcols)


  return JsonResponse(data, safe=False)


def returnMET(request):
  global imurls
  global imcols
  global METdone
  global c1_imurls
  global c1_imcols
  global c2_imurls
  global c2_imcols
  global c3_imurls
  global c3_imcols
  global c4_imurls
  global c4_imcols

  if (METdone == True): METdone = False
  if request.method == 'GET': 
    imurls = []
    imcols = []
    c1_imurls = []
    c1_imcols = []
    c2_imurls = []
    c2_imcols = []
    c3_imurls = []
    c3_imcols = []
    c4_imurls = []
    c4_imcols = []

    data = []
    imsrc = request.GET['imsrc']
    loc = 'media' + imsrc.split('media', 1)[1] 
    image = pil.Image.open(loc)
    image = image.convert('RGB')
    palette = find_palette(image, 5)

## COMMENT THIS OUT FOR NOW
    Thread(target = color_search, args=(palette,)).start()

    data.append('return MET was called')

  return JsonResponse(data, safe=False)


def returnMETvals(request):
  global imurls
  global imcols  
  global METdone
  global c1_imurls
  global c1_imcols
  global c2_imurls
  global c2_imcols
  global c3_imurls
  global c3_imcols
  global c4_imurls
  global c4_imcols

  if request.method == 'GET':  
    data = [None] * 15

    data[0] = METdone
    data[1] = imurls
    data[2] = imcols
    data[3] = c1_imurls
    data[4] = c1_imcols
    data[5] = c2_imurls
    data[6] = c2_imcols
    data[7] = c3_imurls
    data[8] = c3_imcols
    data[9] = c4_imurls
    data[10] = c4_imcols

    if (METdone == True): METdone == False

  return JsonResponse(data, safe=False)

def returnAll(request):
  # print("returnAll was called")
  if request.method == 'GET':
    tic = time.time()
    data = [None] * 43
    palette =[]
    imsrc = request.GET['imsrc']
    # response = requests.get(imsrc)
    # image= pil.Image.open(BytesIO(response.content))
    loc = 'media' + imsrc.split('media', 1)[1] 

    image = pil.Image.open(loc)
    image = image.convert('RGB')

    ## send to google vision label detection
    labels = find_labels(image)

    ## resize image
    image = image.resize((200, int((image.height * 200) / image.width)))

    ## MAYBE FIND A BETTER SEARCH FUNCTION FOR THIS...
    npgoogleimpal = google_im_search_palette(labels)

    # fakepal = np.array([[164.01729191090269, 162.9841735052755, 160.91295427901525], [97.34078212290503, 89.03526536312849, 80.96857541899442], [58.22502557108762, 56.81316058643028, 54.88032730992158], [209.75497597803707, 165.12628689087165, 119.14001372683596], [138.17767653758543, 125.37129840546697, 111.76233864844343]])
    googleimpal = npgoogleimpal.tolist()


    # PALETTE CALLS
    palette = find_palette(image, 5)
    lowsat = sat_palette(palette, .3)
    highsat = sat_palette(palette, 1.8)
    valu = val_palette(palette, 1.3)
    compl1 = comp_palette(palette, 36)
    compl2 = comp_palette(palette, 72)
    compl3 = comp_palette(palette, 108)
    compl4 = comp_palette(palette, 144)
    compl5 = comp_palette(palette, 180)
    invpal = inv_palette(palette)
    bwpal = bw_palette(palette)


    # IMAGE MANIPULATION CALLS
    imtic = time.time()

    # PALETTE IMAGES
    piiA = pil2datauri(image)
    piiB, piiC, piiD, piiE, piiF, piiG, piiH, piiI, piiJ, piiK = pal_ims(image)
     

    switch1 = switchRGB1(image)
    switch2 = switchRGB2(image)
    switch3 = switchRGB3(image)

    hue1, hue2, hue3, palinim1, palinim2, palinim3 = manip_ims(image, palette)

    binim = binarize(image, 128)
    binim2 = binarize2(image)
    edge = cannyedge(image)

    pix1 = pixelate(image, 10)
    pix2 = pixelate(image, 20)
    pix3 = pixelate(image, 30)
    
    invertim = piiK
    sepia1 = sepia(image)
    bw1 = bw(image)


    piiL = pal_hues_in_image(npgoogleimpal, image)

    imtoc = time.time() # TIMER 1 END
    print('image manip timing: ', (imtoc-imtic))

    data[0] = palette.tolist() #0
    data[1] = lowsat.tolist() #1
    data[2] = highsat.tolist() #2
    data[3] = valu.tolist() #3
    data[4] = compl1.tolist() #4
    data[5] = compl2.tolist() #5
    data[6] = compl3.tolist() #6
    data[7] = compl4.tolist() #7
    data[8] = compl5.tolist() #8
    data[9] = invpal.tolist() #9
    data[10] = bwpal.tolist() #10

    data[11] = piiA
    data[12] = piiB
    data[13] = piiC
    data[14] = piiD
    data[15] = piiE
    data[16] = piiF
    data[17] = piiG
    data[18] = piiH
    data[19] = piiI
    data[20] = piiJ 
    data[21] = piiK 
    data[22] = piiL

    data[23] = switch1 
    data[24] = switch2 
    data[25] = switch3 
    data[26] = hue1 
    data[27] = hue2 
    data[28] = hue3 
    data[29] = palinim1 
    data[30] = palinim2 
    data[31] = palinim3 
    data[32] = binim
    data[33] = binim2
    data[34] = edge 
    data[35] = pix1
    data[36] = pix2
    data[37] = pix3
    data[38] = invertim 
    data[39] = sepia1 
    data[40] = bw1
    data[41] = labels 
    data[42] = googleimpal 

    # toc = time.time()
    # print("full timing: ", toc-tic)

  return JsonResponse(data, safe=False)



#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## PIL IMAGE OPERATIONS
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

#converts PIL image to datauri
def pil2datauri(img):
    data = BytesIO()
    img.save(data, "PNG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## IMAGE MANIPULATIONS
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def binarize(im, thresh):
  # im = image.resize((200, int((image.height * 200) / image.width) ))
  grey = im.convert("L")
  imarr = np.asarray(grey)
  arr = np.copy(imarr)
  arr[arr < thresh] = 0
  arr[arr >= thresh] = 255
  result = pil.Image.fromarray(arr)

  s = pil2datauri(result)

  return s

def binarize2(im):
  # im = image.resize((200, int((image.height * 200) / image.width) ))
  grey = im.convert("L")
  imarr = np.asarray(grey)
  arr = np.copy(imarr)

  inds = np.where(arr < 128)

  arr[inds] = 255 - arr[inds]

  result = pil.Image.fromarray(arr)
  s = pil2datauri(result)

  return s

def pixelate(im, size):
  # im = image.resize((200, int((image.height * 200) / image.width) ))
  w = im.width
  h = im.height

  imarr = np.asarray(im)
  arr = np.copy(imarr)

  wr = np.arange(0, w, size)
  hr = np.arange(0, h, size)

  for x in hr:
    for y in wr:
      pix = arr[x][y]
      ir = np.arange(x, x + size)
      jr = np.arange(y, y + size)
      for i in ir:
        for j in jr:
          if i >= h: continue
          if j >= w: continue
          arr[i][j] = pix

  result = pil.Image.fromarray(arr)

  s = pil2datauri(result)

  return s

def cannyedge(image):
    cvimg = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(cvimg,150,250)
    edges = 255 - edges
    result = pil.Image.fromarray(edges)

    s = pil2datauri(result)

    return s

def switchRGB1(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  w = im.width
  h = im.height
  imarr = np.asarray(im)
  arr = np.copy(imarr)

  shape = arr.shape
  arr = arr.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value

  # swaps r & b
  arr[:, [2, 0]] = arr[:, [0, 2]] 
  arr.shape = shape
  arr = arr.astype(np.uint8)
  result = pil.Image.fromarray(arr)
  s = pil2datauri(result)

  return s

def switchRGB2(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  w = im.width
  h = im.height
  imarr = np.asarray(im)
  arr = np.copy(imarr)

  shape = arr.shape
  arr = arr.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value

  # swaps g & b
  arr[:, [2, 1]] = arr[:, [1, 2]] 
  arr.shape = shape
  arr = arr.astype(np.uint8)
  result = pil.Image.fromarray(arr)

  s = pil2datauri(result)

  return s

def switchRGB3(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  w = im.width
  h = im.height
  imarr = np.asarray(im)
  arr = np.copy(imarr)

  shape = arr.shape
  arr = arr.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value

  # swaps r & g
  arr[:, [1, 0]] = arr[:, [0, 1]] 
  arr.shape = shape
  arr = arr.astype(np.uint8)
  result = pil.Image.fromarray(arr)

  s = pil2datauri(result)

  return s

def hueim(im, hue):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  for i in range(im.height):
    for j in range(im.width):
      h,s,v = rgb_to_hsv(imarr[i][j][:3])
      rgb = hsv_to_rgb([hue,s,v])
      imarr[i][j][:3] = rgb

  imarr = imarr.astype(np.uint8)
  result = pil.Image.fromarray(imarr)

  s = pil2datauri(result)

  return s


def comp(im, val):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  for i in range(im.height):
    for j in range(im.width):
      h,s,v = rgb_to_hsv(imarr[i][j][:3])

      h = h + val

      rgb = hsv_to_rgb([h,s,v])
      imarr[i][j][:3] = rgb
  imarr = imarr.astype(np.uint8)
  result = pil.Image.fromarray(imarr)

  s = pil2datauri(result)

  return s

def val(im, val):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  for i in range(im.height):
    for j in range(im.width):
      h,s,v = rgb_to_hsv(imarr[i][j][:3])
      v = v * val
      if (v > 255): v = 255 ## makes sure v does not exceed limit

      rgb = hsv_to_rgb([h,s,v])
      imarr[i][j][:3] = rgb
  imarr = imarr.astype(np.uint8)
  result = pil.Image.fromarray(imarr)

  s = pil2datauri(result)

  return s

def sat(im, val):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  for i in range(im.height):
    for j in range(im.width):
      h,s,v = rgb_to_hsv(imarr[i][j][:3])

      s = s * val
      if (s > 255): s = 255

      rgb = hsv_to_rgb([h,s,v])
      imarr[i][j][:3] = rgb
  imarr = imarr.astype(np.uint8)
  result = pil.Image.fromarray(imarr)

  s = pil2datauri(result)

  return s

def bw(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  grey = im.convert("L")
  s = pil2datauri(grey)

  return s

def invert(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  for i in range(im.height):
    for j in range(im.width):
      imarr[i][j][:3] = [255,255,255] - imarr[i][j][:3]
  imarr = imarr.astype(np.uint8)
  result = pil.Image.fromarray(imarr)

  s = pil2datauri(result)

  return s

def pal_in_image(pal, im):
  # im = image.resize((200, int((image.height * 200) / image.width) ))
  imarr = np.array(im)
  newimarr = np.zeros((im.height, im.width, 3))

  for i in range(im.height):
    for j in range(im.width):
        pix = imarr[i][j][:3]
        pixs = np.full(pal.shape, pix)

        dist1 = np.linalg.norm((pal - pixs), axis = 1)
        minind = np.argmin(dist1)
        minpix = pal[minind]

        newimarr[i][j] = minpix

  newimarr = newimarr.astype(np.uint8)
  result = pil.Image.fromarray(newimarr)

  s = pil2datauri(result)

  return s

def pal_hues_in_image(pal, im):
  # im = image.resize((200, int((image.height * 200) / image.width) ))
  imarr = np.array(im)
  newimarr = np.zeros((im.height, im.width, 3))

  for i in range(im.height):
    for j in range(im.width):
        pix = imarr[i][j][:3]
        pixs = np.full(pal.shape, pix)
       
        ## find the closest palette value
        dist1 = np.linalg.norm((pal - pixs), axis = 1)
        minind = np.argmin(dist1)
        minpix = pal[minind]

        h,s,v = rgb_to_hsv(minpix)
        h1,s1,v1 = rgb_to_hsv(pix)

        newpix = hsv_to_rgb([h,s1,v1])
        newimarr[i][j] = newpix

  newimarr = newimarr.astype(np.uint8)
  result = pil.Image.fromarray(newimarr)

  s = pil2datauri(result)

  return s


def sepia(image):
  imarr = np.asarray(image)
  sha = imarr.shape

  finalim = np.zeros(shape=(sha[0]*sha[1], 3))
  imarr.shape=(sha[0]*sha[1], 3)

  rarr = imarr[:,0]
  garr = imarr[:,1]
  barr = imarr[:,2]

  tr = 0.393 * rarr + 0.769 * garr + 0.189 * barr
  tg = 0.349 * rarr + 0.686 * garr + 0.168 * barr
  tb = 0.272 * rarr + 0.534 * garr + 0.131 * barr

  tr[tr > 255] = 255
  tg[tg > 255] = 255
  tb[tb > 255] = 255

  finalim[:,0] = tr
  finalim[:,1] = tg
  finalim[:,2] = tb

  finalim.shape = sha

  arr = finalim.astype(np.uint8)
  result = pil.Image.fromarray(arr)

  s = pil2datauri(result)

  return s


def pal_ims(im):
  # im = image.resize((200, int((image.height * 200) / image.width)))
  imarr = np.array(im)

  cv1 = 36
  cv2 = 72
  cv3 = 108
  cv4 = 144
  cv5 = 180

  sv1 = .3
  sv2 = 1.8

  vv1 = 1.3

  comp1 = np.zeros(shape=imarr.shape)
  comp2 = np.zeros(shape=imarr.shape)
  comp3 = np.zeros(shape=imarr.shape)
  comp4 = np.zeros(shape=imarr.shape)
  comp5 = np.zeros(shape=imarr.shape)
  sat1 = np.zeros(shape=imarr.shape)
  sat2 = np.zeros(shape=imarr.shape)
  val1 = np.zeros(shape=imarr.shape)
  bw = np.zeros(shape=imarr.shape)
  inv = np.zeros(shape=imarr.shape)


  for i in range(im.height):
    for j in range(im.width):
      h,s,v = rgb_to_hsv(imarr[i][j][:3])

      h1 = h + cv1
      h2 = h + cv2
      h3 = h + cv3
      h4 = h + cv4
      h5 = h + cv5

      rgb1 = hsv_to_rgb([h1,s,v])
      rgb2 = hsv_to_rgb([h2,s,v])
      rgb3 = hsv_to_rgb([h3,s,v])
      rgb4 = hsv_to_rgb([h4,s,v])
      rgb5 = hsv_to_rgb([h5,s,v])
      comp1[i][j][:3] = rgb1
      comp2[i][j][:3] = rgb2
      comp3[i][j][:3] = rgb3
      comp4[i][j][:3] = rgb4
      comp5[i][j][:3] = rgb5

      s1 = s * sv1
      s2 = s * sv2
      if (s1 > 255): s1 = 255
      if (s2 > 255): s2 = 255

      rgb6 = hsv_to_rgb([h,s1,v])
      rgb7 = hsv_to_rgb([h,s2,v])   
      sat1[i][j][:3] = rgb6
      sat2[i][j][:3] = rgb7

      v1 = v * vv1
      if (v1 > 255): v1 = 255 ## makes sure v does not exceed limit
      
      rgb8 = hsv_to_rgb([h,s,v1])
      val1[i][j][:3] = rgb8

      rgb9 = hsv_to_rgb([h,0,v])
      bw[i][j][:3] = rgb9

      inv[i][j][:3] = [255,255,255] - imarr[i][j][:3]


  comp1 = comp1.astype(np.uint8)
  comp2 = comp2.astype(np.uint8)
  comp3 = comp3.astype(np.uint8)
  comp4 = comp4.astype(np.uint8)
  comp5 = comp5.astype(np.uint8)
  sat1 = sat1.astype(np.uint8)
  sat2 = sat2.astype(np.uint8)
  val1 = val1.astype(np.uint8)
  bw = bw.astype(np.uint8)
  inv = inv.astype(np.uint8)


  sat1 = pil.Image.fromarray(sat1)
  sat2 = pil.Image.fromarray(sat2)
  val1 = pil.Image.fromarray(val1)
  comp1 = pil.Image.fromarray(comp1)
  comp2 = pil.Image.fromarray(comp2)
  comp3 = pil.Image.fromarray(comp3)
  comp4 = pil.Image.fromarray(comp4)
  comp5 = pil.Image.fromarray(comp5)
  bw = pil.Image.fromarray(bw)
  inv = pil.Image.fromarray(inv)


  s1 = pil2datauri(sat1)
  s2 = pil2datauri(sat2)
  s3 = pil2datauri(val1)

  s4 = pil2datauri(comp1)
  s5 = pil2datauri(comp2)
  s6 = pil2datauri(comp3)
  s7 = pil2datauri(comp4)
  s8 = pil2datauri(comp5)

  s9 = pil2datauri(bw)
  s10 = pil2datauri(inv)


  return s1, s2, s3, s4, s5, s6, s7, s8, s9, s10

def manip_ims(im, pal):
  imarr = np.array(im)

  hue1 = np.zeros(shape=imarr.shape)
  hue2 = np.zeros(shape=imarr.shape)
  hue3 = np.zeros(shape=imarr.shape)
  palinim1 = np.zeros(shape=imarr.shape)
  palinim2 = np.zeros(shape=imarr.shape)
  palinim3 = np.zeros(shape=imarr.shape)

  hv1 = 50
  hv2 = 270
  hv3 = 300
  pal1 = pal
  pal2 = np.array([[117, 150, 101],[100, 101, 92],[227, 224, 216],[160, 101, 73],[173, 162, 134]])
  pal3 = np.array([[59, 83, 112],[29, 104, 173],[177, 102, 102],[155, 201, 174],[142, 117, 186]])

  for i in range(im.height):
    for j in range(im.width):
      pix = imarr[i][j][:3]
      pixs = np.full(pal1.shape, pix)

      dist1 = np.linalg.norm((pal1 - pixs), axis = 1)
      dist2 = np.linalg.norm((pal2 - pixs), axis = 1)
      dist3 = np.linalg.norm((pal3 - pixs), axis = 1)

      minind1 = np.argmin(dist1)
      minind2 = np.argmin(dist2)
      minind3 = np.argmin(dist3)

      minpix1 = pal1[minind1]
      minpix2 = pal2[minind2]
      minpix3 = pal3[minind3]

      palinim1[i][j] = minpix1
      palinim2[i][j] = minpix2
      palinim3[i][j] = minpix3

      h,s,v = rgb_to_hsv(pix)
      rgb1 = hsv_to_rgb([hv1,s,v])
      rgb2 = hsv_to_rgb([hv2,s,v])
      rgb3 = hsv_to_rgb([hv3,s,v])
      hue1[i][j][:3] = rgb1
      hue2[i][j][:3] = rgb2
      hue3[i][j][:3] = rgb3


  hue1 = hue1.astype(np.uint8)
  hue2 = hue2.astype(np.uint8)
  hue3 = hue3.astype(np.uint8)
  palinim1 = palinim1.astype(np.uint8)
  palinim2 = palinim2.astype(np.uint8)
  palinim3 = palinim3.astype(np.uint8)


  hue1 = pil.Image.fromarray(hue1)
  hue2 = pil.Image.fromarray(hue2)
  hue3 = pil.Image.fromarray(hue3)
  palinim1 = pil.Image.fromarray(palinim1)
  palinim2 = pil.Image.fromarray(palinim2)
  palinim3 = pil.Image.fromarray(palinim3)

  s1 = pil2datauri(hue1)
  s2 = pil2datauri(hue2)
  s3 = pil2datauri(hue3)

  s4 = pil2datauri(palinim1)
  s5 = pil2datauri(palinim2)
  s6 = pil2datauri(palinim3)

  return s1, s2, s3, s4, s5, s6
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## FIND PALETTE // FIND DOMINANT COLOR
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def find_palette(image, num_clust):
# resize to reduce time
  im = image.resize((100, int((image.height * 100) / image.width)))
  ar = np.asarray(im)
  shape = ar.shape
  ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value
  ar = ar[:,:3] # accounts for PNG files, with 4th alpha balue

  codes, dist = scipy.cluster.vq.kmeans(ar, num_clust) # performs k-means on image
  vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
  counts, bins = np.histogram(vecs, len(codes))    # count occurrences
  idx = (-counts).argsort()[:5]
  colors = np.zeros((5,3))
  allcolors = codes[idx]

  # edge cases, if k-means returns fewer than 5 clusters
  n = counts.size
  if (n >= 5):
  	colors = allcolors 
  else:
  	for i in range(0, n):
  	 	colors[i] = allcolors[i]

  	if (n > 2):
  		for j in range(n, 5):
  			colors[j] = allcolors[j-n]
  	else:
  		for j in range(n,5):
  			colors[j] = colors[0]
  return colors 



def find_dom(image, num_clust):
  im = image.resize((100, int((image.height * 100) / image.width) )) # resize to reduce time

  ar = np.asarray(im)
  shape = ar.shape

  ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float) # reshape so it is an array of all rgb value
  ar = ar[:,:3] # accounts for PNG files, with 4th alpha balue

  codes, dist = scipy.cluster.vq.kmeans(ar, num_clust) # performs k-means on image
  vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
  counts, bins = np.histogram(vecs, len(codes))    # count occurrences

  n = len(counts)
  idx = (-counts).argsort()[:n]
  allcolors = codes[idx]

  dom = allcolors[0]

  return dom

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## NEW PALETTES
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def comp_palette(colors, val):
    pal = np.zeros(shape=colors.shape)
    hsv = np.apply_along_axis(rgb_to_hsv, 1, colors)
    h = hsv[:,0]
    s = hsv[:,1]
    v = hsv[:,2]

    # alter h 
    h = h + val

    # replace h
    hsv[:,0] = h

    pal = np.apply_along_axis(hsv_to_rgb, 1, hsv)
    return pal


def sat_palette(colors, val):
    pal = np.zeros(shape=colors.shape)
    hsv = np.apply_along_axis(rgb_to_hsv, 1, colors)
    h = hsv[:,0]
    s = hsv[:,1]
    v = hsv[:,2]

    # alter s
    s = s * val
    s[s>255] = 255

    # replace h
    hsv[:,1] = s

    pal = np.apply_along_axis(hsv_to_rgb, 1, hsv)
    return pal

def val_palette(colors, val):
    pal = np.zeros(shape=colors.shape)
    hsv = np.apply_along_axis(rgb_to_hsv, 1, colors)
    h = hsv[:,0]
    s = hsv[:,1]
    v = hsv[:,2]

    # alter v 
    v = v * val
    v[v>255] = 255 ## makes sure v does not exceed limit

    # replace v
    hsv[:,2] = v

    pal = np.apply_along_axis(hsv_to_rgb, 1, hsv)
    return pal

def inv_palette(colors):
  pal = np.zeros(shape=colors.shape)

  for i in range(len(colors)):
    pal[i] = [255,255,255] - colors[i]

  return pal 


def bw_palette(colors):
    pal = np.zeros(shape=colors.shape)
    hsv = np.apply_along_axis(rgb_to_hsv, 1, colors)

    # alter s
    s = 0

    # replace h 
    hsv[:,1] = s

    pal = np.apply_along_axis(hsv_to_rgb, 1, hsv)
    return pal 

def hue_palette(colors, hue):
    pal = np.zeros(shape=colors.shape)
    hsv = np.apply_along_axis(rgb_to_hsv, 1, colors)

    # alter h and s
    h = hue

    # replace h and s
    hsv[:,0] = h
    pal = np.apply_along_axis(hsv_to_rgb, 1, hsv)

    return pal 

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## COLOR CONVERSION
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def rgb_to_hsv(rgb):
    r = float(rgb[0])
    g = float(rgb[1])
    b = float(rgb[2])
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else (d/high)*255

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]

        h /= 6
        h *= 360

    return h, s, v

def hsv_to_rgb(hsv):
    h = hsv[0] / 360
    s = hsv[1] / 255
    v = hsv[2]

    i = np.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]

    return r, g, b

def rgb2xyz(rgb):
  rgb = rgb / 255.0

  rgb = rgb ** 2.2

  r = rgb[0]
  g = rgb[1]
  b = rgb[2]

  xyz = [0.41239080 * r + 0.35758434 * g + 0.18048079 * b, 
         0.21263901 * r + 0.71516868 * g + 0.07219232 * b, 
         0.01933082 * r + 0.11919478 * g + 0.95053215 * b]

  return xyz

def xyz2lab(xyz):
  RefX = 0.95047
  RefY = 1.00
  RefZ = 1.08883
  var_X = xyz[0] / RefX
  var_Y = xyz[1] / RefY
  var_Z = xyz[2] / RefZ

  if (var_X > 0.008856):
    var_X = var_X **( 1/3 )
  else:
    var_X = ( 7.787 * var_X ) + ( 4 / 29 )
   
  if (var_Y > 0.008856):
    var_Y = var_Y **( 1/3 )
  else:
    var_Y = ( 7.787 * var_Y ) + ( 4 / 29)
   
  if (var_Z > 0.008856):
    var_Z = var_Z **( 1/3 )
  else:
    var_Z = ( 7.787 * var_Z ) + ( 4 / 29 )

  L = (116 * var_Y ) - 16
  a = 500 * (var_X - var_Y)
  b = 200 * (var_Y - var_Z)

  return (L, a, b)



#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## COLOR COMPARISION / DELTA E 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def deltE(lab1, lab2):
  dL = lab1[0] - lab2[0]

  c1 = (lab1[1]**2 + lab1[2]**2) ** .5
  c2 = (lab2[1]**2 + lab2[2]**2) ** .5
  dC = c1 - c2
  
  da = lab1[1] - lab2[1]
  db = lab1[2] - lab2[2]

  dHsq = da**2 + db**2 - dC**2
 
  sL = 1 
  sC = 1 + 0.045 * c1
  sH = 1 + 0.015 * c1

  var1 = (dL/sL)**2
  var2 = (dC/sC)**2
  var3 = dHsq/(sH**2)

  delt = (var1 + var2 + var3) ** .5

  return delt

def findDeltE(searchRGB, imageRGB):
  xyz1 = rgb2xyz(searchRGB)
  xyz2 = rgb2xyz(imageRGB)
  lab1 = xyz2lab(xyz1)
  lab2 = xyz2lab(xyz2)
  delt = deltE(lab1, lab2)

  return delt


def find_avg(search_pal, pal):
  # sort_pal = pal
  # weighted = 0

  # for i in range(len(search_pal)):
  #   rgb = search_pal[i]
  #   rgb2 = pal[i]
  #   d = findDeltE(rgb, rgb2)
  #   weighted = weighted + (d * .2)


  # SORT & CODE 
  sort_pal = [0,0,0,0,0]
  weighted = 0

  for i in range(len(search_pal)):
    rgb = search_pal[i]
    min_ind = 0
    min_rgb = []
    min_d = 10000

    for j in range(len(pal)):
      rgb2 = pal[j]
      # d = np.linalg.norm(rgb - rgb2)
      d = findDeltE(rgb, rgb2)
      if (d < min_d):
        min_d = d
        min_rgb = rgb2
        min_ind = j
        
    sort_pal[i] = min_rgb 

    if i == 0: weight = .2
    if i == 1: weight = .2
    if i == 2: weight = .2
    if i == 3: weight = .2
    if i == 4: weight = .2

    weighted = weighted + (min_d * weight)
    pallist = pal.tolist()
    pallist.pop(min_ind)
    pal = np.array(pallist)

  return sort_pal, weighted



#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## MET DATABASE OPERATIONS
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

## MULTIPROCESSING VERSION
def cpu_bound(number, searchpal, METcolors):
  avgs = [None] * 59519
  sort_pals = [None] * 59519

  for i in range(number, number + 59519): ##833 for subset, 59595 for full
    pal = METcolors[i]
    sort_pals[i - number], avgs[i - number] = find_avg(searchpal, pal) ## call find average here
  return avgs, sort_pals


def find_sums(numbers, searchpal, METcolors):
  with mp.Pool() as pool:
    poolvars = partial(cpu_bound, searchpal=searchpal, METcolors=METcolors) ## here, 'hello' will be the palette variable
    avgs, sort_pals = zip(*pool.map(poolvars, numbers))   

    npavgs = np.array(avgs)
    npavgs = np.ndarray.flatten(npavgs)

    npsort_pals = np.array(sort_pals)
    npsort_pals.shape = (238076, 5, 3)
    # npsort_pals.shape = (4998, 5, 3)

    return npavgs, npsort_pals

def conv(x):
  return np.array(json.loads(x))


def color_search(search_pal):
    global imurls
    global imcols
    global METdone
    global c1_imurls
    global c1_imcols
    global c2_imurls
    global c2_imcols
    global c3_imurls
    global c3_imcols
    global c4_imurls
    global c4_imcols


    print("COLOR SEARCH WAS CALLED")

    conn = sqlite3.connect('METdatabase.db')
    c = conn.cursor()
    query = "SELECT ImageURL, Colors, Class FROM database "
    c.execute(query)
    metobjs = c.fetchall()


    # tic = time.time() # TIMER 1 START
    heynp = np.array(metobjs)
    METimurls = heynp[:,0]
    colors = heynp[:,1]
    METclass = heynp[:,2]  
    METcolors = np.array([conv(xi) for xi in colors])  
    # toc = time.time() # TIMER 1 END
    # print('convert to float: ', (toc-tic))

    numbers = [59519*x for x in range(4)]
    # numbers = [833*x for x in range(6)]


    start_time = time.time() # TIMER 2 START
    avgs, sort_pals = find_sums(numbers, search_pal, METcolors)
    duration = time.time() - start_time # TIMER 2 END
    print(f"Duration {duration} seconds")

    # tic1 = time.time() # TIMER 3 START
    args = np.argsort(avgs)


    ## MAKE Class DISTINCTIONS HERE

    imurls = METimurls[args]
    imcols = sort_pals[args]
    classes = METclass[args]

    # imurls = imurls[0:10]
    # imcols = imcols[0:10]
    # classes = classes[0:10]

    # RETURN ALL
    # CLASS1 PAINTINGS/DRAWINGS
    c1_ind = np.where(classes == "C1")
    c1_imurls = imurls[c1_ind][0:30]
    c1_imcols = imcols[c1_ind][0:30]


    # CLASS2 TEXTILES/CLOTHING
    c2_ind = np.where(classes == "C2")
    c2_imurls = imurls[c2_ind][0:30]
    c2_imcols = imcols[c2_ind][0:30]


    # CLASS3 METAL/GLASS
    c3_ind = np.where(classes == "C3")
    c3_imurls = imurls[c3_ind][0:30]
    c3_imcols = imcols[c3_ind][0:30]


    # CLASS4 STONE/SCULPTURE/WOOD
    c4_ind = np.where(classes == "C4")
    c4_imurls = imurls[c4_ind][0:30]
    c4_imcols = imcols[c4_ind][0:30]


    imurls = imurls[0:30]
    imcols = imcols[0:30]


    imurls = imurls.tolist()
    imcols = imcols.tolist()

    c1_imurls = c1_imurls.tolist()
    c2_imurls = c2_imurls.tolist()
    c3_imurls = c3_imurls.tolist()
    c4_imurls = c4_imurls.tolist()

    c1_imcols = c1_imcols.tolist()
    c2_imcols = c2_imcols.tolist()
    c3_imcols = c3_imcols.tolist()
    c4_imcols = c4_imcols.tolist()

    conn.close()
    METdone = True

    return imurls, imcols, c1_imurls, c2_imurls, c3_imurls, c4_imurls, c1_imcols, c2_imcols, c3_imcols, c4_imcols
    # return imurls, imcols

## ORIGINAL CODE
# def color_search(search_pal):
#     global xyztime
#     global labtime
#     global dtime
#     print("color search was called")
    
#     conn = sqlite3.connect('METObjects.db')
#     c = conn.cursor()
#     query = "SELECT ImageURL, Colors, Classification FROM data "
#     # search = "Paintings"
#     # c.execute(query, (search,))
#     c.execute(query)
#     metobjs = c.fetchall()
#     # tot = len(metobjs)
#     tot = 1000
    
#     imurls = [None] * tot
#     imcols = [None] * tot
#     classifications = [None] * tot
#     avgs = [None] * tot


#     tic = time.clock()

#     # sort = np.array([[111,111,111],[222,222,222],[155,155,155],[13,13,13],[99,99,99]])
#     # avg = 3

#     avgtime = 0
   
#     for i in range(tot):
#       metobj = metobjs[i]
#       imurls[i] = metobj[0]
#       classifications[i] = metobj[2]

#       stcols = metobj[1]

#       npcols = np.array(json.loads(stcols)) ## fastest
#       # npcols = eval(stcols)
#       # npcols = np.array(eval(stcols)) ## faster than ast.literal_eval
#       # npcols = np.array(ast.literal_eval(stcols), dtype=float

     
#       avgstart = time.clock()
#       sort, avg = find_avg(search_pal, npcols)
#       avgend = time.clock()

#       runtime = avgend - avgstart
#       avgtime = avgtime + runtime
      
#       imcols[i] = sort
#       avgs[i] = avg


#     toc = time.clock()
#     print("loop timer ", (toc - tic)) 

#     avgtime = avgtime/tot
#     print("search func avg time ", avgtime)
    
#     avgs = np.array(avgs)
#     imurls = np.array(imurls)
#     imcols = np.array(imcols)
#     classifications = np.array(classifications)

#     args = np.argsort(avgs)

#     imurls = imurls[args]
#     imcols = imcols[args]
#     classifications = classifications[args]

#     # print("xyz avg time ", (xyztime/tot))
#     # print("lab avg time ", (labtime/tot))
#     # print("delta avg time ", (dtime/tot))


#     # # paintings & prints
#     # paintind = np.where((classifications == 'Paintings') | (classifications == 'Prints'))
#     # imurls = imurls[paintind]
#     # imcols = imcols[paintind]
#     # classificaitons = classifications[paintind]


#     imurls = imurls[0:10]
#     imcols = imcols[0:10]
#     classifications = classifications[0:10]

#     imurls = imurls.tolist()
#     imcols = imcols.tolist()
#     classifications = classifications.tolist()


#     conn.close()

#     return imurls, imcols, classifications

def MET_word_search(word):
    conn = sqlite3.connect('METdatabase.db')
    c = conn.cursor()

    search = "%" + word + "%"
    query = " SELECT ImageURL, Colors FROM database WHERE Tags LIKE ? or ObjectName LIKE ?"

    c.execute(query, (search, search,))

    imurls = []
    imcols = []

    metobjs = c.fetchall()
    for metobj in metobjs:
      imurls.append(metobj[0])

      st = metobj[1]
      npar = np.array(ast.literal_eval(st), dtype=float)
      cols = npar.tolist()

      imcols.append(cols)

    conn.close()

    return imurls, imcols


#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## GOOGLE CLOUD VISION API
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def find_labels(im):
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"
  listlabels = []

  # [START vision_python_migration_client]
  client = vision.ImageAnnotatorClient()
  # [END vision_python_migration_client]

  # The name of the image file to annotate
  # file_name = os.path.abspath(loc)

  # Loads the image into memory
  # with io.open(file_name, 'rb') as image_file:
  #   content = image_file.read()

  buffer = io.BytesIO()
  im.save(buffer, "JPEG")
  content = buffer.getvalue()

  image = vision.Image(content=content)

  # Performs label detection on the image file
  response = client.label_detection(image=image)
  labels = response.label_annotations

  for label in labels:
      listlabels.append(" " + label.description)
  # [END vision_quickstart]

  listlabels = listlabels[0:5]

  return listlabels


#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
## GOOGLE IMAGE PALETTE
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def google_im_search_palette(queries):
  gis = GoogleImagesSearch('AIzaSyA5H4YVZnReJRv7GZAx2m3wg3q3Q2OvOOw', '32e819b64b0302e9c')
  domcols = np.zeros((5,3))

  ## HACK - CONSIDER CHANGING FOR VARYING LABEL LIST LENGTHS
  for i in range(5):
  ## THIS IS SUPER HACKY RIGHT NOW, DON'T KNOW WHY SOME IMAGES CAN'T BE READ
      _search_params = {
        'q': queries[i],
        'num': 1,
      }

      my_bytes_io = BytesIO()

      # this will only search for images:
      gis.search(search_params =_search_params)

      for image in gis.results():
          my_bytes_io.seek(0)
          raw_image_data = image.get_raw_data()
          image.copy_to(my_bytes_io, raw_image_data)
          my_bytes_io.seek(0)

          try:
            temp_img = pil.Image.open(my_bytes_io)
            dom = find_dom(temp_img, 5)
            domcols[i] = dom
            break

          except: 
            print("exception")
            pass

  return domcols




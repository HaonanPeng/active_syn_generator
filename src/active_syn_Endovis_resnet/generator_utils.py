# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:55:00 2020

@author: 75678
"""
import numpy as np
import math
import cv2
import ast
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# import random_shape_generator as rsg


# [function] distance_2points
# [Discription]: compute distance between two points
# [parameters]: point1 & point2 - [x,y] coordinate of points
# [return]: scalar distance
def distance_2points(point1, point2):
    distance = np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return distance


# [function] distance_2points
# [Discription]: compute the derivative between two points
# [parameters]: point1 & point2 - [x,y] coordinate of points
# [return]: scalar derivative
def derivative_2points(point1, point2):
    if point2[0]-point1[0] == 0:
        print('[derivative_2points]: infinite slope, set to a very large number')
    derivative = (point2[1]-point1[1])/((point2[0]-point1[0])+1e-10)
    return derivative

# [function] line_solver
# [Discription]: given 2 points, find the line 'y = ax + b'
# [parameters]: point1 & point2 - [x,y] coordinate of points
# [return]: scalar derivative
def line_solver(point1, point2):
    a = derivative_2points(point1, point2)
    b = point1[1] - a * point1[0]
    return a, b

def rand_from_range(lim1, lim2):
    randn_number = lim1 + (lim2 - lim1) * np.random.rand()
    return randn_number

def round_up_to_odd(f):
    if f >=0 and f <=1:
        f = 1
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def show_img(name, img):
    # img = img.astype(np.uint8)
    # cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    img = cv2.resize(img, (600,600))
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def show_multi_imgs(name, imgs, save_path = None):
    # img = img.astype(np.uint8)
    count = 0
    for img in imgs:
        cv2.imshow(name + str(count),img)
        if save_path != None:
            cv2.imwrite(save_path + name + str(count) + '.jpg', img)
        count = count + 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def show_tool_map(name, tool_map):
    img = np.array((tool_map.astype(np.uint8)*255)).clip(0,255)
    img = cv2.resize(img, (600,600))
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

def show_contour_distance_map(name, contour_distance_map, tool_contour):
    cd_map_pos = np.copy(contour_distance_map)
    cd_map_pos[cd_map_pos<0] = 0
    # cd_map_pos = np.sqrt(cd_map_pos)
    
    cd_map_nag = contour_distance_map * (-1)
    cd_map_nag[cd_map_nag<0] = 0
    cd_map_nag = np.sqrt(cd_map_nag)
    
    cd_map_pos = np.int_(np.divide(cd_map_pos, np.max(cd_map_pos)) * 155)
    cd_map_nag = np.int_(np.divide(cd_map_nag, np.max(cd_map_nag)) * 155)
    
    img = np.zeros([3001,3001,3],np.uint8)
    
    img[:,:,0] =  cd_map_nag + 100 
    img[:,:,1] =  0
    img[:,:,2] =  cd_map_pos + 100
    
    img[tool_contour[:,0]+1000,tool_contour[:,1]+1000,:] = 255
    
    img2 = np.zeros([3001,3001,3],np.uint8) 
    img2[:,:,0] =  np.bool_(cd_map_pos)*255
    img2[:,:,1] =  np.bool_(cd_map_pos)*255
    img2[:,:,2] =  np.bool_(cd_map_pos)*255
    img2[tool_contour[:,0]+1000,tool_contour[:,1]+1000,0] = 0
    img2[tool_contour[:,0]+1000,tool_contour[:,1]+1000,1] = 0
    img2[tool_contour[:,0]+1000,tool_contour[:,1]+1000,2] = 255
    
    img = cv2.resize(img, (1000, 1000)) 
    img2 = cv2.resize(img2, (1000, 1000)) 
    
    cv2.imshow('name',img)
    cv2.imshow('name' + '_bw',img2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return None

# draw points on a list of points (notice that if there is only one point, the input should still be like [[x,y]])
def draw_points(img, points, color = (100,254,0)):
    img_w_points = img
    print('draw_points:')
    print(points)
    for point in points:
        img_w_points = cv2.circle(img_w_points, center = (int(point[1]), int(point[0])), radius=4, color = color)
    
    return img_w_points

# draw arrowed vectors based on the start points and the vectors, they are all nx2 arrays
def draw_vectors(img, points, vectors, color = (100,200,0)):
    for point, vector in zip(points, vectors):
        point1 = (int(point[1]), int(point[0]))
        point2 = (int(point[1] + vector[1]), int(point[0] + vector[0]))
        img = cv2.arrowedLine(img, point1, point2 , color, thickness = 2)
    
    return img

# normalize vectors, the input should be a list-like vectors, notice that even with only one vector, the input should still be [vector]
def normalize_vectors(vectors):
    vectors_norm = []
    for vector in vectors:
        vector_norm = vector / np.sqrt(vector[0]**2 + vector[1]**2)
        vectors_norm.append(vector_norm)
    return vectors_norm

def from_np_array(array_string):
    array_string = (','.join(array_string.replace('[ ', '[').split())).replace('[,','[')
    print(array_string)
    print(type(array_string))
    array_string1 = np.matrix(array_string)
    # array_string1 = ast.literal_eval(array_string)
    print('----------------------------')
    print(array_string1)
    print(type(array_string1))
    np_array = np.fromstring(array_string, sep=' ')
    # np_array = np.array(ast.literal_eval(array_string))
    print(np_array)
    return np_array

def from_np_array_object(array_string):
    array_string = ' '.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string), dtype = object)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D((int(0.5*image.shape[0]), int(0.5*image.shape[1])), angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def img_erosion(img, val_type, ero_size):
    erosion_size = val_type
    erosion_type = 0
    if val_type == 0:
        erosion_type = cv2.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv2.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    erosion_dst = cv2.erode(img, element)
    return erosion_dst
        
def img_dilatation(img, val_type, dila_size):
    dilatation_size = dila_size
    dilatation_type = 0
    if val_type == 0:
        dilatation_type = cv2.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv2.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(img, element)
    return dilatation_dst

def extract_tool(image, label, dilation = 60):
    image = np.copy(image)
    tool_label = label
    tool_mask = np.copy(tool_label)
    tool_mask_small = img_dilatation(tool_mask, 0, int(dilation/2))
    tool_mask_large = img_dilatation(tool_mask, 0, dilation)
    # tool_label = tool_label.clip(0, 1)
    tool_mask_small[tool_mask_small>0] = 1
    tool_mask_large[tool_mask_large>0] = 1
    
    
    tool_img = image[:]
    for chn in range(0,3):
        tool_img[:,:,chn] = image[:,:,chn] * tool_mask_large
    tool_extract_map = tool_mask_small
    return tool_img, tool_label, tool_extract_map

def mask_mul(img, mask):
    img_new = np.zeros(img.shape)
    for chn in range(0,3):
        img_new[:,:,chn] = img[:,:,chn] * mask
        
    return img_new

# return a rotated img_2 so that it can be used to inpaint img_1
def find_inpaint_rotation(img_1, label_1, img_2, label_2):
    
    if np.sum(label_1*label_2) < 5:
        return img_2
    # elif np.sum(label_1*cv2.rotate(np.copy(label_2), cv2.ROTATE_90_CLOCKWISE)) < 5:
    #     return cv2.rotate(img_2, cv2.ROTATE_90_CLOCKWISE)
    elif np.sum(label_1*cv2.rotate(np.copy(label_2), cv2.ROTATE_180)) < 5:
        return cv2.rotate(img_2, cv2.ROTATE_180)
    # elif np.sum(label_1*cv2.rotate(np.copy(label_2), cv2.ROTATE_90_COUNTERCLOCKWISE)) < 5:
    #     return cv2.rotate(img_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return np.zeros(img_1.shape)

def visual_mask(mask, color):
    shape = mask.shape
    mask_v = np.zeros((shape[0], shape[1], 3), np.float)
    for chn in range(0,3):
        mask_v[:,:,chn] = mask * color[chn]
    mask_v = mask_v.clip(0,255)
    mask_v = mask_v.astype(np.uint8)
        
    return mask_v

def make_transparent_bkgd(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    img_new = np.dstack([img, alpha])
    return img_new

def center_add_image(small_img, large_img):   
    # load resized image as grayscale
    img = small_img
    img_shape = img.shape
    h = img_shape[0]
    w = img_shape[1]
    
    # load background image as grayscale
    back = large_img
    back_shape = back.shape
    hh = back_shape[0]
    ww = back_shape[1]
    
    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round(abs(hh-h)/2)
    xoff = round(abs(ww-w)/2)
    
    # use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    if img_shape[0] > back_shape[0] and img_shape[1] < back_shape[1]:
        result[:, xoff:xoff+w] = img[yoff:yoff+hh,:]
    elif img_shape[0] < back_shape[0] and img_shape[1] > back_shape[1]:
        result[yoff:yoff+hh, :] = img[:, xoff:xoff+w]
    elif img_shape[0] < back_shape[0] and img_shape[1] < back_shape[1]:
        result[yoff:yoff+h, xoff:xoff+w] = img[:]
    else:
        result[:] = img[yoff:yoff+hh, xoff:xoff+w]
    return result


    

def random_shape(control_points = 5, rad = 0.2, edgy = 0.05 ):
    
    c = np.array([1,1])
    a = rsg.get_random_points(n=control_points, scale=1) + c
    c, _ = rsg.get_bezier_curve(a,rad=rad, edgy=edgy)
    c = (c - 1) * 999
    c = c.astype(np.int32)
    
    map_shape = np.zeros([1001,1001])
    
    cv2.drawContours(map_shape,[c], 0, 1, -1)
    return map_shape


    
    
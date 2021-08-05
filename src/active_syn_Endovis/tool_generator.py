# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:42:54 2020

@author: 75678
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

import generator_utils as gutil



class tool_generator():
    
    # [class members]
    img_tool_texture = None # the background of the generated tool, usually a metal texture image, it will be resized to (1001,1001,3)
    tool_contour = None # a list of points representing the contour of the tool [[x1,y1],[x2,y2],....]
    tool_map = None # a boolean matrix that the tool points has value 1 while all other points are 0 
    tool_type = None # an int number indecating the type of the tool
    lines = None # a list of sub-lists of points, each sub-list is points represents a line or a curve
    contour_distance_map = None # a 4001x4001 matrix (gray image), where the tool contour is at the center and the value in the matrix is the distance of that point to the contour
    tool_img = None # the generated tool image [1001x1001x3] BGR color image with not rendering
    geo_feature_points = None # [unfinished] A list of points indecating the geometric features of the tool, which will be used as the ground truth for later neural network training
    geo_feature_vectors = None # [unfinished] A list of vectors indecating the geometric features of the tool, which will be used as the ground truth for later neural network training
    render_feature_points = None # A list of points provide information for rendering the combined tool and background. This varies from different type of tools
    render_feature_vectors = None # A list of vectors provide information for rendering the combined tool and background. This varies from different type of tools

    
    # [function] output_result
    # [Discription]: output the results of this tool generator
    # [parameters]: None
    # [return]: see class members
    def output_result(self):
        return self.tool_img, self.tool_type, self.tool_contour, self.tool_map, self.lines, self.contour_distance_map, self.geo_features
        
    
    # [function] load_img_tool_texture
    # [Discription]: load the texture image of the tool and resize it to [1001,1001,3]
    # [parameters]: file_path - string of the image file path
    # [return]: None
    def load_img_tool_texture(self, file_path):
         img = cv2.imread(file_path)
         
         if img is None:
             print('[Tool Generator] Error: Tool texture image load failed, image does not exist')
             sys.exit()
             
         if np.abs((img.shape[0]-img.shape[1])/img.shape[0]) > 0.2:
             print('[Tool Generator] Error: Too large width & hight ratio. Tool texture image should be close to square')
             sys.exit()
             
         self.img_tool_texture = cv2.resize(img, (1001,1001))
         return None
    
    
    # [function] contour_define
    # [Discription]: using feature points to define the contour of the tool, details in documentation
    # [parameters]: featurePoints - a list of feature points; tool_type - associate with feature points 
    # [result]: a list of point indecating the contour of the tool in a [1001,1001] matrix
    # [return]: None
    def contour_define(self, featurePoints, tool_type = 1):
        # [tool type 1]: for this type, 5 points are used to define the tool. Finally 3 line and 1 curve are formed
        self.tool_type = tool_type
        if tool_type == 1: 
            point1 = featurePoints[0] # assign points
            point2 = featurePoints[1]
            point3 = featurePoints[2]
            point4 = featurePoints[3]
            point5 = featurePoints[4]
            
            a1, b1 = gutil.line_solver(point1, point2) # solve the lines and the curve
            [a2, b2, c2] = np.polyfit([point2[0], point3[0], point4[0]], [point2[1], point3[1], point4[1]], deg = 2)
            a3, b3 = gutil.line_solver(point4, point5)
            a4, b4 = gutil.line_solver(point1, point5)
            
            
            line1 = [] # construct the lines and curves, point1 is contained in line1, point 2 is contained in line2(curve), and so on
            # line1.append(point1) included in the last step
            for x1 in range(point1[0]+1, point2[0]):
                y1 = int(a1 * x1 + b1)
                line1.append([x1, y1])
            
                
            line2 = [] # the curve
            line2.append(point2)
            for x2 in range(point2[0]+1, point4[0]):
                y2 = int(a2 * x2**2 + b2 * x2 + c2)
                line2.append([x2, y2])
            
                
            line3 = [] # 
            line3.append(point4)
            for x3 in range(point4[0]+1, point5[0]):
                y3 = int(a3 * x3 + b3)
                line3.append([x3, y3])
            
                
            line4 = [] # 
            line4.append(point5)
            for x4 in range(point5[0], point1[0]+1, -1):
                y4 = int(a4 * x4 + b4)
                line4.append([x4, y4])
            
            self.lines = [np.array(line1), np.array(line2), np.array(line3), np.array(line4)]
            
            self.tool_contour = np.array([point1])
            for line in self.lines:
                self.tool_contour = np.append(self.tool_contour, line, axis = 0)
             
            point_mid15 = [int(0.5*(point1[0]+point5[0])), int(0.5*(point1[1]+point5[1]))]    
            self.geo_feature_points = featurePoints # [unfinished] shoule be replaced by the needed ground truth information
            self.geo_feature_vectors = [[point1[0] - point2[0], point1[1] - point2[1]], 
                                        [point5[0] - point4[0], point5[1] - point4[1]],
                                        [point_mid15[0] - point3[0], point_mid15[1] - point3[1]]]
            
            self.render_feature_points = featurePoints
            self.render_feature_vectors = [[point1[0] - point2[0], point1[1] - point2[1]], 
                                           [point5[0] - point4[0], point5[1] - point4[1]],
                                           [point_mid15[0] - point3[0], point_mid15[1] - point3[1]]]
                
            # print(str(self.tool_contour.shape))
                
        else:
            print('[tool generator]: unkown tool type, please check the document')
        
        return None
    
    
    # [function] contour_distance_map_gen
    # [Discription]: using point polygon test to generate a 3001x3001 matrix (gray image), where the tool contour is at the center and the value in the matrix is the distance of that point to the contour
    # [parameters]: None
    # [result]: self.contour_distance_map - a 3001x3001 matrix (gray image), where the tool contour is at the center and the value in the matrix is the distance of that point to the contour
    # [return]: contour_distance_map
    def contour_distance_map_gen(self):
        self.contour_distance_map = np.zeros([3001,3001])
        contour = self.tool_contour + 1000
        for row in range (0,3001):
            for col in range (0,3001):
                self.contour_distance_map[row, col] = cv2.pointPolygonTest(contour, (row, col), measureDist = True)
        
        return self.contour_distance_map
    
    # [function] tool_gen
    # [Discription]: using the generated tool contour and the texture to generate a fack tool [1001,1001,3] BGR image (note that in opencv, color channel is BGR instead of RGB) 
    # [parameters]: None
    # [result]: self.tool_img - a 1001x1001x3 matrix (color image),
    # [return]: tool_img
    def tool_gen(self):
        self.tool_img = np.zeros((1001,1001,3), np.uint8)
        
        self.tool_map = np.bool_(self.contour_distance_map[1000:2001, 1000:2001].clip(0,100))
        
        for channel in range(0,3):
            self.tool_img[:, :, channel] = self.img_tool_texture[:, :, channel] * self.tool_map
            
        return None
     

        
     
    def show_img_tool_texture(self):
        cv2.imshow('img_tool_texture',self.img_tool_texture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    
    def show_tool_img(self):
        cv2.imshow('tool_img',self.tool_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    
    def show_tool_contour(self):
        img = np.zeros([1001,1001,3], np.uint8)
        
        img[self.tool_contour[:,0],self.tool_contour[:,1],:] = 255
        
        
        cv2.imshow('img_tool_contour',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
    
    def show_contour_distance_map(self):
        cd_map_pos = np.copy(self.contour_distance_map)
        cd_map_pos[cd_map_pos<0] = 0
        # cd_map_pos = np.sqrt(cd_map_pos)
        
        cd_map_nag = self.contour_distance_map * (-1)
        cd_map_nag[cd_map_nag<0] = 0
        cd_map_nag = np.sqrt(cd_map_nag)
        
        cd_map_pos = np.int_(np.divide(cd_map_pos, np.max(cd_map_pos)) * 155)
        cd_map_nag = np.int_(np.divide(cd_map_nag, np.max(cd_map_nag)) * 155)
        
        img = np.zeros([3001,3001,3],np.uint8)
        
        img[:,:,0] =  cd_map_nag + 100 
        img[:,:,1] =  0
        img[:,:,2] =  cd_map_pos + 100
        
        img[self.tool_contour[:,0]+1000,self.tool_contour[:,1]+1000,:] = 255
        
        img2 = np.zeros([3001,3001,3],np.uint8) 
        img2[:,:,0] =  np.bool_(cd_map_pos)*255
        img2[:,:,1] =  np.bool_(cd_map_pos)*255
        img2[:,:,2] =  np.bool_(cd_map_pos)*255
        img2[self.tool_contour[:,0]+1000,self.tool_contour[:,1]+1000,0] = 0
        img2[self.tool_contour[:,0]+1000,self.tool_contour[:,1]+1000,1] = 0
        img2[self.tool_contour[:,0]+1000,self.tool_contour[:,1]+1000,2] = 255
        
        img = cv2.resize(img, (1000, 1000)) 
        img2 = cv2.resize(img2, (1000, 1000)) 
        
        cv2.imshow('contour_distance',img)
        cv2.imshow('contour_distance_bw',img2)
        
        cv2.imwrite('temp_img/contour_distance.jpg', img)
        cv2.imwrite('temp_img/contour_distance_bw.jpg', img2)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
        
        
    
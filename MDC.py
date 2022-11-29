"""
Created on Sat. June 4 2022
@author: Wang Zhicheng
"""

import cv2
import numpy as np

# 构造MDC类

class MDC():
    def __init__(self,img):
        h,w,c = img.shape
        self.img = img.astype('float64') # 将图片存入
        self.img_num = cv2.integral(np.ones((h,w)))
        # 计算积分图，加速计算 积分图的i,j，为从[0,i),[0,j),不包括(i,j)
        self.integral_map, self.integral2_map = cv2.integral2(img) # 计算平方积分图
    # 计算方向对比度
    # DCir = sqrt(sum(sum(Ii,ch-Ij,ch)**2))

    def MDC_Contrast(self):
        h,w,c = self.img.shape
        Contrast_map = np.zeros((4, h, w))
        # 左上角的结果
        l_u_2 = self.integral2_map
        l_u = self.integral_map
        l_u_num = self.img_num
        Contrast_map[0] = np.sum(l_u_2[:h,:w],axis=2) - 2 * np.sum(l_u[:h,:w]*self.img,axis=2) + l_u_num[:h,:w]*np.sum(np.power(self.img,2),axis=2)
        # 右上角的结果
        # 先求区域求和的结果
        r_u_2 = self.integral2_map[:,-1,:].reshape(-1,1,3) - self.integral2_map
        r_u = self.integral_map[:,-1,:].reshape(-1,1,3) - self.integral_map
        r_u_num = self.img_num[:,-1].reshape(-1,1) - self.img_num
        Contrast_map[1] = np.sum(r_u_2[:h,1:],axis=2) - 2 * np.sum(r_u[:h,1:]*self.img,axis=2) + r_u_num[:h,1:]*np.sum(np.power(self.img,2),axis=2)
        # 左下角的结果
        l_d_2 = self.integral2_map[-1, :, :].reshape(1, -1, 3) - self.integral2_map
        l_d = self.integral_map[-1, :, :].reshape(1, -1, 3) - self.integral_map
        l_d_num = self.img_num[-1, :].reshape(1, -1) - self.img_num
        Contrast_map[2] = np.sum(l_d_2[1:, :w], axis=2) - 2 * np.sum(l_d[1:, :w] * self.img, axis=2) + l_d_num[1:,:w] * np.sum(np.power(self.img, 2), axis=2)
        # 右下角结果
        r_d_2 = self.integral2_map[-1, -1, :].reshape(1, 1, 3) - self.integral2_map[-1, :, :].reshape(1, -1, 3) - self.integral2_map[:,-1,:].reshape(-1,1,3) + self.integral2_map
        r_d = self.integral_map[-1, -1, :].reshape(1, 1, 3) - self.integral_map[-1, :, :].reshape(1, -1, 3) - self.integral_map[:,-1,:].reshape(-1,1,3)+ self.integral_map
        r_d_num = self.img_num[-1, -1].reshape(1, 1) - self.img_num[-1, :].reshape(1, -1) - self.img_num[:,-1].reshape(-1,1)+ self.img_num
        Contrast_map[3] = np.sum(r_d_2[1:, 1:], axis=2) - 2 * np.sum(r_d[1:, 1:] * self.img, axis=2) + r_d_num[1:,1:] * np.sum(np.power(self.img, 2), axis=2)
        cmap =np.sqrt(np.min(Contrast_map,axis=0))
        cmap = cv2.normalize(cmap, None, 0, 255, cv2.NORM_MINMAX)
        self.cmap = cmap
        return cmap

    def smooth_cmap(self,w=1):
        c = 16 # 设定切分的通道数
        cmap_L = self.cmap # 读取原始显著性图
        h, w = cmap_L.shape # 得到显著性图的大小
        cmap_L = cmap_L / c  # 从256减少为16通道
        cmap_L = cmap_L.astype('int')
        # 找到16个不同的亮度，并且计算其各自的连通度
        # 得到一个边界图，得到一个
        cmap_BC = np.zeros(c) # 每个区域的平均连通性得分
        cmap_L_a = np.zeros_like(cmap_L) # 用来保存图片格式的连通性得分
        for i in range(c):
            cmap_L_find = np.zeros_like(cmap_L)
            cmap_L_find[np.where(cmap_L==i)] = 1 # 将i区域设定为1
            pnum = np.sum(np.sum(cmap_L_find)) # 找到i区域的像素个数
            cmap_L_find_copy = cmap_L_find.copy()
            cmap_L_find_copy[1:h-1,1:w-1] = 0
            bnum = np.sum(np.sum(cmap_L_find_copy)) # 找到边界个数
            BC = bnum/np.sqrt(pnum)
            cmap_find = cmap_L_find * self.cmap # 找到区域对应的原始cmap，并且其他位置都是0
            cmap_L_a[np.where(cmap_L==i)] = np.sum(np.sum(cmap_find * np.exp(-w*BC))) / pnum
        self.cmap = cmap_L_a*0.5 + self.cmap * 0.5 # 计算平滑后的显著性图
        self.cmap = cv2.normalize(self.cmap, None, 0, 255, cv2.NORM_MINMAX)
        self.cmap = self.cmap.astype('uint8') # 得到平滑后显著性图
        return self.cmap

    def enhance_cmap(self,theta=0.8,a=0.5,b=0.5):
        t,r = cv2.threshold(self.cmap,0,255,cv2.THRESH_OTSU)
        M = np.ones_like(self.cmap)*127
        M[np.where(self.cmap>(1+theta)*t)] = 255
        M[np.where(self.cmap<(1-theta)*t)] = 0
        # 寻找使用ostu划分后的显著图
        self.cmap = self.cmap.astype('float')/255
        self.cmap[np.where(M==255)] = 1-a*(1-self.cmap[np.where(M==255)])
        self.cmap[np.where(M == 0)] = b*self.cmap[np.where(M==0)]
        # 获得显著性增强后图
        self.cmap = (self.cmap*255).astype('uint8')
        return self.cmap


if __name__=="__main__":
    img = cv2.imread('test.jpg')
    M = MDC(img)
    cmap_original = M.MDC_Contrast()
    cmap_smooth = M.smooth_cmap()
    cmap_enhance = M.enhance_cmap()
    cv2.imwrite('test_MDC.png',cmap_enhance)
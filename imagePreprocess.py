import cv2
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import shutil

# global variables
imgs_in = "E:/FIB_SEM_Pic/data/%d.tif"
imgs_label_in = "E:/FIB_SEM_Pic/label/%d.tif"

res_imgs_in = "E:/FIB_SEM_Pic/test/data/%d.tif"
res_imgs_label_in = "E:/FIB_SEM_Pic/test/label/%d.tif"
#method_in = "E:/FIB_SEM_Pic/test/fcn8-diceloss/fcn8-diceloss-result/%d_predict.tif"          #作为fcn
#method_in = "E:/FIB_SEM_Pic/test/fcn-softmax-diceloss/result/%d_predict.tif"
#method_in = "E:/FIB_SEM_Pic/test/pspnet50-softmax + catogray_crossentropy_loss/result/%d_predict.tif" #作为pspnet
method_in = "E:/FIB_SEM_Pic/test/baseline_Subconv/result/%d_predict.tif" #作为unet
#method_in = "E:/FIB_SEM_Pic/test/unet-softmax-CELoss-result/result/%d_predict.tif" # 作为porposed
#method_in = "E:/FIB_SEM_Pic/test/new_SCDENet_noBN_CELOSS/result/%d_predict.tif"   # 作为segnet
#method_in = "E:/FIB_SEM_Pic/test/SCDENet_noBN_CELoss/result/%d_predict.tif"   # 作为segnet

res_imgs_out = "E:/FIB_SEM_Pic/test"

imgs_out = "E:/FIB_SEM_Pic/rnd_data/%d.tif"
imgs_label_out = "E:/FIB_SEM_Pic/rnd_label/%d.tif"

rename_dir = "E:/FIB_SEM_Pic/val/label"

imgs_in2 = "E:/FIB_SEM_Pic/zmtsy_2/3/%d.tif"
imgs_out2 = "E:/WRH_PAPER/DL/pic"

snake_origin_in = "E:/snake_test/origin/%d.tif"
snake_label_in = "E:/snake_test/label/%d.tif"
#snake_res_in = "E:/snake_test/out-res/%d.tif"
#snake_res_in = "E:/snake_test/yuzhi-add/%d.BMP"
snake_res_in = "E:/snake_test/DL-res/%d_predict.tif"
snake_out = "E:/snake_test"


PIC_NUM=100
# PIC_NUM=290
START_INDEX = 500

def get_series_histplot(imgs_in):
    '''
    得到序列图的直方图
    '''
    for i in range(1,PIC_NUM+1):
        image = cv2.imread(imgs_in%(i))
        #cv2.imshow("Image",image)
        #plt.imshow(image)
        hist = cv2.calcHist([image],[0],None,[256],[0,256])
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.ylim([0,15000])
        plt.savefig(imgs_out%(i))
        plt.show()

def get_single_histplot(image):
    #cv2.imshow("Image",image)
    #plt.imshow(image)
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    #plt.ylim([0,262144])
    plt.ylim([0,20000])
    plt.show()


def mouse_click(event, x, y, flags, para):
    '''
    鼠标点击图片,查看灰度
    '''
    gray = cv2.cvtColor(para, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(para, cv2.COLOR_BGR2HSV)
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        print('PIX:', x, y)
        #print("BGR:", para[y, x])
        print("GRAY:", gray[y, x])
        #print("HSV:", hsv[y, x])
        print("\n")

def view_pixel(img):
    cv2.namedWindow("img")
    cv2.setMouseCallback("img",mouse_click,img)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()

def get_canny(img):
    '''
    Canny算子检测边缘与图像叠加
    '''
    canny_img=cv2.Canny(img,88,220)
    #return canny_img
    h=img.shape[0]
    w=img.shape[1]
    for row in range(h):
        for col in range(w):
            if canny_img[row][col] == 255 :
                img[row][col][0]=0
                img[row][col][1]=0
                img[row][col][2]=255

    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    return img

def view_canny(path):
    for i in range(START_INDEX,PIC_NUM+START_INDEX):
        tmpath = path%(i)
        img = cv2.imread(tmpath)
        #img = kmean_seg(img)
        img = get_canny(img)
        #cv2.imwrite(imgs_kmeans_out%(i),img)
        cv2.imshow('canny',img)
        cv2.waitKey(0)


def GetGrandientImg(image,thre=0):
    '''
    得到梯度图像
    '''
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 中值滤波
    h = image.shape[0]
    w = image.shape[1]
    iSharp = np.zeros(image.shape, np.uint8)
    for i in range(h - 1):
        for j in range(w - 1):
            x = abs(int(image[i, j + 1]) - int(image[i, j]))
            y = abs(int(image[i + 1, j]) - int(image[i, j]))
            v = (y + x) * 3
            if (v >= 255):
                v = 255
            if (v < thre):
                v = 0
            iSharp[i, j] = int(v)
    #return iSharp
    cv2.imshow('img',iSharp)
    cv2.waitKey(0)


def mouse_click2(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        para[0]=x
        para[1]=y
        print('选取像素点坐标: ',x,y)

def view_pixel_change(path):
    '''
    查看某一点像素值的变化情况,生成plot
    '''
    para=[0,0]
    x = np.linspace(1,PIC_NUM,PIC_NUM)
    y = np.zeros((PIC_NUM,),dtype=int)
    img = cv2.imread(path%(1))
    cv2.namedWindow("img")
    cv2.setMouseCallback("img",mouse_click2,para)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()
    for i in range(1,PIC_NUM):
        img = cv2.imread(path%(i))
        #y[i-1]=img[pos_y][pos_x][0]
        y[i-1]=img[para[1]][para[0]][0]

    plt.figure()
    plt.title("Pixel Change")
    plt.xlabel("Index")
    plt.ylabel("Pixels")
    plt.ylim([0, 256])
    plt.plot(x,y)
    plt.show()

    for i in range(1,PIC_NUM):
        img = cv2.imread(path%(i))
        img = cv2.circle(img,(para[0],para[1]),3,(0,0,255))
        cv2.imshow('img',img)
        cv2.waitKey()
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_XZ(path):
    '''
    得到点击像素处的XZ面
    并显示白线
    并显示该像素点的灰度变化
    '''
    #显示点
    raw_data=[]
    para=[0,0]
    for i in range(START_INDEX,PIC_NUM+START_INDEX):
        img = cv2.imread(path%(i))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        raw_data.append(img)

    show_img = cv2.imread(path%(START_INDEX))
    mask_img = np.zeros_like(show_img)
    cv2.namedWindow("img")
    cv2.setMouseCallback("img",mouse_click2,para)
    cv2.imshow('img',show_img)
    cv2.waitKey(0)
    cv2.destroyWindow('img')
    cv2.circle(mask_img,(para[0],para[1]),3,(0,0,255))
    cv2.imshow('point_img',cv2.add(mask_img,show_img))

    #得到XZ面

    xz_w=show_img.shape[0]
    xz_h=PIC_NUM
    xz_img=np.zeros((xz_h,xz_w),np.uint8)
    for row in range(xz_h):
        for col in range(xz_w):
            xz_img[row][col] = raw_data[row][col][para[0]]

    for i in range(PIC_NUM):
        xz_img[i][para[1]]=255

    #显示该点的灰度变化情况
    x = np.linspace(START_INDEX,PIC_NUM + START_INDEX,PIC_NUM)
    y = np.zeros((PIC_NUM,),dtype=int)
    for i in range(START_INDEX,PIC_NUM + START_INDEX):
        img_pix = cv2.imread(path%(i))
        #y[i-1]=img[pos_y][pos_x][0]
        y[i-START_INDEX]=img_pix[para[1]][para[0]][0]

    plt.figure()
    plt.title("Pixel Change")
    plt.xlabel("Index")
    plt.ylabel("Pixels")
    plt.ylim([0, 256])
    plt.plot(x,y)
    plt.show()


    cv2.imshow('X-Z',xz_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_XZ(path_in,path_out):
    '''
    存储每一列的XZ面
    '''
    raw_data=[]
    para=[0,0]
    for i in range(1,PIC_NUM+1):
        img = cv2.imread(path_in%(i))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        raw_data.append(img)

    show_img = cv2.imread(path_in%(1))
    cv2.imshow('img',show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pic_sum=show_img.shape[1]
    xz_w=show_img.shape[0]
    xz_h=PIC_NUM
    xz_img=np.zeros((xz_h,xz_w),np.uint8)
    for i in range(pic_sum):
        for row in range(xz_h):
            for col in range(xz_w):
                xz_img[row][col] = raw_data[row][col][i]
        cv2.imwrite(path_out%(i),xz_img)


def kmean_seg(img):
    '''
    K均值聚类
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 设置分为几类
    k = 6
    # 二维转为一维数据,转为float32,处理速度快
    img1d = img.reshape((img.shape[0]*img.shape[1],1))
    img1d = np.float32(img1d)

    # 设置停止迭代模式
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,10.0)

    # 设置种子起始位置
    flags=cv2.KMEANS_RANDOM_CENTERS

    # 执行k均值
    compactness, labels, centers = cv2.kmeans(img1d, k, None, criteria, 10, flags)

    # labels 存值为0,1,2...k-1
    labels = np.uint8(labels)
    img_seg=labels.reshape((img.shape[0],img.shape[1]))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img_seg[row][col]=np.uint8(centers[img_seg[row][col]])
    #print('compactness: ',compactness)
    #print('centers: ',centers)

    cv2.imshow('img_seg',img_seg)
    cv2.waitKey(0)

    return img_seg


def mouse_click3(event, x, y, flags, para):
    #if event == cv2.EVENT_LBUTTONDOWN:  # 左键鼠标点击
    #if flags == cv2.EVENT_FLAG_LBUTTON:  # 鼠标press
    if flags == cv2.EVENT_FLAG_ALTKEY or flags == cv2.EVENT_FLAG_LBUTTON:  # 按住alt+鼠标点击
        para[0]=0
        para[1]=x
        para[2]=y
        print('选取前景像素点坐标: ',x,y)
    #elif event == cv2.EVENT_RBUTTONDOWN:  # 右键鼠标点击
    #elif flags == cv2.EVENT_FLAG_RBUTTON:  # 按住shift+鼠标点击
    elif flags == cv2.EVENT_FLAG_SHIFTKEY or flags == cv2.EVENT_FLAG_RBUTTON:  # 按住shift+鼠标点击
        para[0]=1
        para[1]=x
        para[2]=y
        print('选取背景像素点坐标: ',x,y)
    elif flags == cv2.EVENT_FLAG_CTRLKEY:
        para[0]=2
        para[1]=x
        para[2]=y
        print('擦除像素点坐标: ',x,y)

def lk_optical_flow(path_in):
    '''
    运用稀疏光流跟踪标记点:
    鼠标点击设置标记点,按下空格保存该标记点,esc选择完毕退出
    按下空格观察标记点变化情况
    '''
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=50,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.8))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    first_frame=cv2.imread(path_in%(1))
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    first_frame_mask=np.zeros_like(first_frame)
    #p0 = np.append(p0,[[[0.0,0.0]]],axis=0)
    #print(p0)


    # 自定义选择关键点:取法一
    '''
    p0=[]
    para=[0,0]
    cv2.namedWindow("img")
    key = 1
    cv2.setMouseCallback("img",mouse_click3,para)
    while key != 27:  # esc
        cv2.circle(first_frame_mask,(para[0],para[1]),3,(0,0,255),-1)
        point_img=cv2.add(first_frame,first_frame_mask)
        cv2.imshow('img',point_img)
        key = cv2.waitKey(30)
        if key == 32:  # space
            p0.append([[para[0],para[1]]])
            print(p0)
            key = 1

    p0 = np.array(p0,dtype=np.float32)
    cv2.destroyAllWindows()
    '''

    # 更新: 选择的点的邻域圆也设置为关键点:取法2
    p0=[]
    para=[0,0]
    cv2.namedWindow("img")
    key = 1
    cv2.setMouseCallback("img",mouse_click3,para)
    while key != 27:  # esc
        cv2.circle(first_frame_mask,(para[0],para[1]),2,(0,0,255),-1)
        point_img=cv2.add(first_frame,first_frame_mask)
        cv2.imshow('img',point_img)
        key = cv2.waitKey(30)

    cv2.circle(first_frame_mask,(para[0],para[1]),10,(0,0,0),-1)

    h = first_frame_mask.shape[0]
    w = first_frame_mask.shape[1]

    for row in range(h):
        for col in range(w):
            if first_frame_mask[row][col][2] == 255:
                p0.append([[col,row]])

    p0 = np.array(p0,dtype=np.float32)
    cv2.destroyAllWindows()


    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    for i in range(2,PIC_NUM+1):
        frame = cv2.imread(path_in%(i))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # 是否画出轨迹
            mask = cv2.line(mask, (a, b), (c, d), (50,20,50), 2)

            frame = cv2.circle(frame, (a, b), 3, (0,255,0), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        cv2.waitKey(0)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        #是否要保存图像
        #cv2.imwrite("E:/WRH_PAPER/optical_watershed/%d.png"%(i),img)

    cv2.destroyAllWindows()


def dense_optflow(path):
    '''
    稠密光流此处无用,因为变化过小
    '''
    frame1 = cv2.imread(path%(1))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    for i in range(2,PIC_NUM+1):
        frame2 = cv2.imread(path%(i))
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
        cv2.waitKey(0)
        prvs = next
    cv2.destroyAllWindows()


'''
分水岭分割
'''
def watershed_seg(imgs_in,imgs_out_path):
    '''
    交互式分水岭分割(适用于序列图)
    1. 左键为目标标记,右键为背景标记,按住ctrl清除鼠标位置标记.
    2. 按下esc进行一下张(下一步),在结果图片窗口上按下'r'重新对这一张进行修改
    3. 前一张的标记点会保留到下一张,减少工作量.
    :param imgs_in: 输入图片路径
    :param imgs_out_path: 输出图片路径
    :return: null
    '''

    '''
    # 均值漂移 meanshift
    for i in range(1,PIC_NUM+1):
        img = cv2.imread(path%(i))
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.pyrMeanShiftFiltering(img,10,20)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    '''
    # 分水岭demo
    # gray\binary image
    img = cv2.imread(path%(1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)

    # morphology operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(mb, kernel, iterations=3)
    cv2.imshow("strcut operation", sure_bg)

    # distance transform
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    dist_output = cv2.normalize(dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow("distanceTransform", dist_output * 70)

    ret, surface = cv2.threshold(dist, dist.max() * 0.6, 255, cv2.THRESH_BINARY)
    cv2.imshow("seekseeds", surface)

    surface_fg = np.uint8(surface)
    unknown = cv2.subtract(sure_bg, surface_fg)
    cv2.imshow('unkown',unknown)
    ret, markers = cv2.connectedComponents(surface_fg)
    print(ret)

    # watershed transfrom
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers=markers)
    img[markers == -1] = [0, 0, 255]
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    start_index = 0
    end_index = 290
    para = [0,-1,-1]
    first_img = cv2.imread(imgs_in%(0))
    mask_img = np.zeros_like(first_img)

    i = start_index
    while i != end_index:
        img = cv2.imread(imgs_in%(i))
        img_bk = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #mask_img = np.zeros_like(img)
        markers = np.zeros_like(gray,dtype=np.int32)
        cv2.namedWindow('img')
        cv2.imshow('img',img)
        cv2.setMouseCallback('img',mouse_click3,para)
        key = 0

        red_label = (0,0,255)
        green_label = (0,255,0)
        clear_label = (0,0,0)

        # 手动选择标记点
        while key != 27:  # esc
            if para[0] == 0:
                cv2.circle(mask_img,(para[1],para[2]),1,red_label,-1)
            elif para[0] == 1:
                cv2.circle(mask_img,(para[1],para[2]),2,green_label,-1)
            elif para[0] == 2:
                cv2.circle(mask_img,(para[1],para[2]),10,clear_label,-1)
            point_img=cv2.add(mask_img,img)
            cv2.imshow('img',point_img)
            key = cv2.waitKey(30)

        cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

        h = mask_img.shape[0]
        w = mask_img.shape[1]

        for row in range(h):
            for col in range(w):
                if mask_img[row][col][2] == 255:
                    markers[row][col]=1
                elif mask_img[row][col][1] == 255:
                    markers[row][col]=2

        markers = cv2.watershed(img,markers=markers)
        img_res = np.copy(img)

        # 显示边界
        img[markers==-1] = [0,0,255]
        cv2.imshow('res_bonder_%d'%(i),img)

        # 显示分割后的图像
        #img_res[markers==1] = [0,0,255]
        img_res[markers==1] = [255,255,255]
        img_res[markers!=1] = [0,0,0]
        #cv2.imshow('res_seg_%d'%(i),img_res)

        # 显示原图
        #cv2.imshow('origin_%d'%(i),img_bk)
        r_key = cv2.waitKey(0)
        if r_key == 114:  # r_key=='r'
            cv2.destroyAllWindows()
            continue
        else:
            cv2.destroyAllWindows()

        img_gray_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)

        #cv2.imwrite(os.path.join(imgs_out_path,"%d.tif"%i),img_gray_res)
        #cv2.imwrite(os.path.join(imgs_out_path,"%d_edge.tif"%i),img)
        i = i+1


'''
将分水岭和光流结合
'''

def watershed_optical_seg(path):

    '''
    分水岭和光流结合,先手动选取标记点,之后再分水岭分割
    得到分割后的图像的孔隙轮廓(或整个孔隙区域)作为新的标记点进行光流跟踪
    无法在过程中手动修改标记点
    :param path: 图片输入路径
    :return: void
    '''
    tmpath = path%(0)
    #first_img = cv2.imread(path%("0500"))
    first_img = cv2.imread(tmpath)
    gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(first_img)
    markers = np.zeros_like(gray,dtype=np.int32)
    para=[-1,-1]
    cv2.namedWindow('img')
    cv2.imshow('img',first_img)
    #cv2.setMouseCallback('img',mouse_click3,para)
    cv2.setMouseCallback('img',mouse_click2,para)
    key = 0
    pos=[]

    label = (0,0,255)

    # 手动选择标记点
    while key != 27:  # esc
        cv2.circle(mask_img,(para[0],para[1]),7,label,-1)
        point_img=cv2.add(mask_img,first_img)
        cv2.imshow('img',point_img)
        key = cv2.waitKey(30)
        if key == 32:  # space
            pos.append([para[0],para[1]])
            print(pos)
            label = (0,255,0)
            key = 1

    cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

    h = mask_img.shape[0]
    w = mask_img.shape[1]

    for row in range(h):
        for col in range(w):
            if mask_img[row][col][2] == 255:
                markers[row][col]=1
            elif mask_img[row][col][1] == 255:
                markers[row][col]=2

    markers = cv2.watershed(first_img,markers=markers)

    img_res = np.copy(first_img)
    img_bound = np.copy(first_img)
    # 显示边界
    img_bound[markers==-1] = [0,0,255]
    cv2.imshow('res_bounder',img_bound)

    # 显示分割后的图像
    img_res[markers==1] = [0,0,255]
    cv2.imshow('res_seg',img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''
    开始光流追踪序列图
    '''
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=50,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.8))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    first_frame = np.copy(first_img)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    first_frame_mask=np.zeros_like(first_frame)
    #p0 = np.append(p0,[[[0.0,0.0]]],axis=0)
    #print(p0)

    p0=[]
    for row in range(img_res.shape[0]):
        for col in range(img_res.shape[1]):

            #跟踪轮廓点还是整个区域点
            #temp = img_bound[row][col]
            temp = img_res[row][col]
            if temp[0]==0 and temp[1]==0 and temp[2]==255:
                #去除图像最外面的方形边界
                if (row==0 or row==img_res.shape[0]-1) or (col==0 or col==img_res.shape[1]-1):
                    continue
                else:
                    p0.append([[col,row]])
    p0 = np.array(p0,dtype=np.float32)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    for i in range(2,PIC_NUM+1):
        frame = cv2.imread(path%(i))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # 是否画出轨迹
            mask = cv2.line(mask, (a, b), (c, d), (50,20,50), 2)

            #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            frame = cv2.circle(frame, (a, b), 1, (0,255,0), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('optical_frame', img)
        cv2.waitKey(0)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        #是否保存图片
        #cv2.imwrite("E:/WRH_PAPER/optical_gac/%d.png"%(i),img)
    cv2.destroyAllWindows()

def show_max_min(path):
    img = cv2.imread(path%(1))
    img_temp = np.copy(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    markers = np.zeros_like(gray,dtype=np.int32)

    min_pix = np.min(gray)
    max_pix = np.max(gray)

    h = img.shape[0]
    w = img.shape[1]
    for row in range(h):
        for col in range(w):
            if gray[row][col] >= 155 or gray[row][col] <= 25:
                img[row][col] = [0,0,255]
                markers[row][col] = 1
            #elif gray[row][col] == max_pix:
            #    img[row][col] = [0,255,0]
            elif gray[row][col]>=98 and gray[row][col] <= 99 :
                img[row][col] = [255,0,0]
                markers[row][col] = 2

    markers = cv2.watershed(img_temp,markers=markers)
    #显示边界
    img_temp[markers==-1] = [0,0,255]

    cv2.imshow("origin",gray)
    cv2.imshow("minmax",img)
    cv2.imshow("res",img_temp)
    cv2.waitKey(0)


def optical_watershed_iter(path):
    '''
    首先手动设置标记点,再跟踪这些标记点,再将新的标记点作为分水岭分割
    无法在过程中手动修改标记点
    :param path:
    :return:
    '''
    first_img = cv2.imread(path%(1))
    gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(first_img)
    markers = np.zeros_like(gray,dtype=np.int32)
    para=[-1,-1]
    cv2.namedWindow('img')
    cv2.imshow('img',first_img)
    #cv2.setMouseCallback('img',mouse_click3,para)
    cv2.setMouseCallback('img',mouse_click2,para)
    key = 0
    pos=[]

    red_point=[]
    green_point=[]

    label = (0,0,255)

    max_point_dist = 15.0

    # 手动选择标记点
    while key != 27:  # esc
        cv2.circle(mask_img,(para[0],para[1]),7,label,-1)
        point_img=cv2.add(mask_img,first_img)
        cv2.imshow('img',point_img)
        key = cv2.waitKey(30)
        if key == 32:  # space
            pos.append([para[0],para[1]])
            print(pos)
            label = (0,255,0)
            key = 1

    cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

    h = mask_img.shape[0]
    w = mask_img.shape[1]

    for row in range(h):
        for col in range(w):
            if mask_img[row][col][2] == 255:
                markers[row][col]=1
                red_point.append([[col,row]])
            elif mask_img[row][col][1] == 255:
                markers[row][col]=2
                green_point.append([[col,row]])

    markers = cv2.watershed(first_img,markers=markers)

    img_res = np.copy(first_img)
    img_bound = np.copy(first_img)
    # 显示边界
    img_bound[markers==-1] = [0,0,255]
    cv2.imshow('res_bounder',img_bound)

    # 显示分割后的图像
    img_res[markers==1] = [0,0,255]
    cv2.imshow('res_seg',img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''
    开始光流追踪序列图
    '''
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=50,
                          qualityLevel=0.2,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1))
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    first_frame = np.copy(first_img)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    first_frame_mask=np.zeros_like(first_frame)

    p0_red_point = np.array(red_point,dtype=np.float32)
    p0_green_point = np.array(green_point,dtype=np.float32)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    for i in range(2,PIC_NUM+1):
        frame = cv2.imread(path%(i))
        frame_temp = np.copy(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1_red_point, st_red, err_red = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_red_point, None, **lk_params)
        p1_green_point, st_green, err_green = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_green_point, None, **lk_params)

        # Select good points
        red_good_new = p1_red_point[st_red == 1]
        red_good_old = p0_red_point[st_red == 1]

        green_good_new = p1_green_point[st_green == 1]
        green_good_old = p0_green_point[st_green == 1]


        #运用分水岭分割显示图像
        markers_temp = np.zeros_like(frame_gray,dtype=np.int32)

        red_delete_index = []
        green_delete_index = []

        # draw the red points tracks
        for j, (new, old) in enumerate(zip(red_good_new,red_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                red_delete_index.append(j)
            else:
                # 是否画出轨迹
                mask = cv2.line(mask, (a, b), (c, d), (50,20,50), 2)

                #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                frame = cv2.circle(frame, (a, b), 1, (0,0,255), -1)

                if b<512 and a<512:
                    markers_temp[int(b)][int(a)] = 1

        #draw the green points tracks
        for j, (new, old) in enumerate(zip(green_good_new,green_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                green_delete_index.append(j)
            else:
                # 是否画出轨迹
                mask = cv2.line(mask, (a, b), (c, d), (20,50,20), 2)

                #frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                frame = cv2.circle(frame, (a, b), 1, (0,255,0), -1)

                if b < 512 and a < 512:
                    markers_temp[int(b)][int(a)] = 2

        markers_temp = cv2.watershed(frame_temp,markers=markers_temp)

        #显示分水岭分割后的边界
        frame_temp[markers_temp==-1] =[0,0,255]
        cv2.imshow('optical_bound',frame_temp)


        img = cv2.add(frame, mask)
        cv2.imshow('optical_frame', img)

        #去除移动距离过远的点
        red_good_new = np.delete(red_good_new,red_delete_index,axis=0)
        green_good_new = np.delete(green_good_new,green_delete_index,axis=0)

        old_gray = frame_gray.copy()
        p0_red_point = red_good_new.reshape(-1, 1, 2)
        p0_green_point = green_good_new.reshape(-1,1,2)

        # 是否保存图片
        result_img = np.copy(img)
        result_img[markers_temp==-1] = [0,0,255]
        cv2.imshow('add',result_img)
        cv2.waitKey(0)
        #cv2.imwrite("E:/WRH_PAPER/result/%d.png"%(i),result_img)
    cv2.destroyAllWindows()

def seedgrow(path):
    '''
    种子生长算法
    :param path:
    :return: void
    '''
    connects = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
    para=[-1,-1]
    img=cv2.imread(path%(20))
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #点是否被标记
    img_mark=np.zeros_like(img_gray)
    #已经生长的点
    seedlist=[]
    h=img.shape[0]
    w=img.shape[1]

    cv2.imshow('img',img)
    cv2.setMouseCallback('img',mouse_click3,para)
    cv2.waitKey(0)


    seedlist.append([para[1],para[0]])
    #标记类别1
    class_A = 1
    #阈值
    T = 5

    #开始生长
    while (len(seedlist) > 0) :
        tmp_seed = seedlist[0]
        seedlist.pop(0)
        img_mark[tmp_seed[0]][tmp_seed[1]]=class_A
        #遍历8邻域
        for i in range(8):
            tmp_pos=[tmp_seed[0]+connects[i][0],tmp_seed[1]+connects[i][1]]
            if(tmp_pos[0]<0 or tmp_pos[1]<0 or
                    tmp_pos[0] >=h or tmp_pos[1] >= w):
                continue
            dist = abs(int(img_gray[tmp_seed[0]][tmp_seed[1]])-
                       int(img_gray[tmp_pos[0]][tmp_pos[1]]))
            if (dist < T and img_mark[tmp_pos[0]][tmp_pos[1]] == 0):
                img_mark[tmp_pos[0]][tmp_pos[1]] = 1
                seedlist.append(tmp_pos)

    #生长结束,显示结果
    for j in range(h):
        for k in range(w):
            if(img_mark[j][k] == 1):
                img[j][k]= [0,0,255]
    cv2.imshow('img_res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def optical_watershed_rnd_iter(path):
    '''
    首先手动选择第一帧的标记点,用光流跟踪
    之后再分水岭分割得到的区域中随机选择点作为下一帧的标记点
    无法在过程中手动修改标记点
    :param path:
    :return:
    '''
    first_img = cv2.imread(path%(1))
    gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(first_img)
    markers = np.zeros_like(gray,dtype=np.int32)
    para=[-1,-1]
    cv2.namedWindow('img')
    cv2.imshow('img',first_img)
    cv2.setMouseCallback('img',mouse_click3,para)
    key = 0
    pos=[]

    red_point=[]
    green_point=[]

    label = (0,0,255)

    max_point_dist = 15.0

    # 手动选择标记点
    while key != 27:  # esc
        cv2.circle(mask_img,(para[0],para[1]),7,label,-1)
        point_img=cv2.add(mask_img,first_img)
        cv2.imshow('img',point_img)
        key = cv2.waitKey(30)
        if key == 32:  # space
            pos.append([para[0],para[1]])
            print(pos)
            label = (0,255,0)
            key = 1

    cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

    h = mask_img.shape[0]
    w = mask_img.shape[1]

    for row in range(h):
        for col in range(w):
            if mask_img[row][col][2] == 255:
                markers[row][col]=1
                red_point.append([[col,row]])
            elif mask_img[row][col][1] == 255:
                markers[row][col]=2
                green_point.append([[col,row]])

    markers = cv2.watershed(first_img,markers=markers)

    img_res = np.copy(first_img)
    img_bound = np.copy(first_img)
    # 显示边界
    img_bound[markers==-1] = [0,0,255]
    cv2.imshow('res_bounder',img_bound)

    # 显示分割后的图像
    img_res[markers==1] = [0,0,255]
    cv2.imshow('res_seg',img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    '''
    开始光流追踪序列图
    '''
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1))

    # Take first frame and find corners in it
    first_frame = np.copy(first_img)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    first_frame_mask=np.zeros_like(first_frame)

    p0_red_point = np.array(red_point,dtype=np.float32)
    p0_green_point = np.array(green_point,dtype=np.float32)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    for i in range(2,PIC_NUM+1):
        frame = cv2.imread(path%(i))
        frame_temp = np.copy(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1_red_point, st_red, err_red = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_red_point, None, **lk_params)
        p1_green_point, st_green, err_green = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_green_point, None, **lk_params)

        # Select good points
        red_good_new = p1_red_point[st_red == 1]
        red_good_old = p0_red_point[st_red == 1]

        green_good_new = p1_green_point[st_green == 1]
        green_good_old = p0_green_point[st_green == 1]


        #运用分水岭分割显示图像
        markers_temp = np.zeros_like(frame_gray,dtype=np.int32)

        red_delete_index = []
        green_delete_index = []

        # draw the red points tracks
        for j, (new, old) in enumerate(zip(red_good_new,red_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                red_delete_index.append(j)
            else:
                # 是否画出轨迹
                mask = cv2.line(mask, (a, b), (c, d), (50,20,50), 2)

                frame = cv2.circle(frame, (a, b), 1, (0,0,255), -1)

                if b<512 and a<512:
                    markers_temp[int(b)][int(a)] = 1

        #draw the green points tracks
        for j, (new, old) in enumerate(zip(green_good_new,green_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                green_delete_index.append(j)
            else:
                # 是否画出轨迹
                #mask = cv2.line(mask, (a, b), (c, d), (20,50,20), 2)

                #frame = cv2.circle(frame, (a, b), 1, (0,255,0), -1)

                if b < 512 and a < 512:
                    markers_temp[int(b)][int(a)] = 2

        markers_temp = cv2.watershed(frame_temp,markers=markers_temp)

        #显示分水岭分割后的边界
        frame_temp[markers_temp==-1] =[0,0,255]
        cv2.imshow('optical_bound',frame_temp)


        img = cv2.add(frame, mask)
        cv2.imshow('optical_frame', img)

        #去除移动距离过远的点
        red_good_new = np.delete(red_good_new,red_delete_index,axis=0)
        green_good_new = np.delete(green_good_new,green_delete_index,axis=0)

        old_gray = frame_gray.copy()

        #从新得到的分水岭区域中随机选取标记点进行下一次的跟踪
        red_origin = []
        green_origin = []
        for j in range(h):
            for k in range(w):
                if markers_temp[j][k] == 1:
                    red_origin.append([k,j])
                elif markers_temp[j][k] == 2:
                    green_origin.append([k,j])
        red_rnd_point_tmp = random.sample(red_origin,int(len(red_origin)/2))
        green_rnd_point_tmp = random.sample(green_origin,int(len(green_origin)/3))

        red_rnd_point = np.array(red_rnd_point_tmp,dtype=np.float32)
        green_rnd_point = np.array(green_rnd_point_tmp,dtype=np.float32)

        p0_red_point = red_rnd_point.reshape(-1,1,2)
        p0_green_point = green_rnd_point.reshape(-1,1,2)



        # 是否保存图片
        result_img = np.copy(img)
        result_img[markers_temp==-1] = [0,0,255]
        cv2.imshow('add',result_img)
        cv2.waitKey(3000)
        #cv2.imwrite("E:/WRH_PAPER/optical_random_watershed/res%d.png"%(i),result_img)
        #cv2.imwrite("E:/WRH_PAPER/optical_random_watershed/bound%d.png"%(i),frame_temp)
    cv2.destroyAllWindows()



def rename(path):
    total = 600
    for i in range(total):
        #shutil.copy(os.path.join(path,"%d.tif"%i),os.path.join(path,"%d_label.tif"%i))
        shutil.copy(os.path.join(path,"%d.tif"%(i+2602)),os.path.join(path,"%d.tif"%(i)))
        os.remove(os.path.join(path,"%d.tif"%(i+2602)))
    return


def gammaTrans(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = np.power(img/float(np.max(img)),0.7) * 255.0

    img1 = img1.astype(np.uint8)


    cv2.imshow('src',img)
    cv2.imshow('res',img1)
    get_single_histplot(img)
    get_single_histplot(img1)
    cv2.waitKey(0)

def selfMul(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = np.power(img/float(np.max(img)),2) * 255.0

    img1 = img1.astype(np.uint8)


    cv2.imshow('src',img)
    cv2.imshow('res',img1)
    get_single_histplot(img)
    get_single_histplot(img1)
    cv2.waitKey(0)

def movedir(pathin,labelin,pathout,labelout):
    total = 319
    for i in range(total):
        shutil.copy(pathin%(i),pathout%(i+2170))
        shutil.copy(labelin%(i),labelout%(i+2170))
    return


def watershed_optical_label(imgs_in,imgs_out_path):
    '''
    交互式标注: 分水岭+光流跟踪标记点
    :param imgs_in:
    :param imgs_out_path:
    :return:
    '''
    start_index = 5
    end_index = 319

    first_img = cv2.imread(imgs_in%(start_index))
    first_img_gray = cv2.cvtColor(first_img,cv2.COLOR_BGR2GRAY)
    mask_img = np.zeros_like(first_img)

    para = [0,-1,-1]

    red_point = []
    green_point = []

    i = start_index

    # 处理第一张
    while True:
        img_ = cv2.imread(imgs_in%(i))
        origin_img = np.copy(img_)
        gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        #mask_img = np.zeros_like(img)
        markers = np.zeros_like(gray_,dtype=np.int32)
        cv2.namedWindow('img')
        cv2.imshow('img',img_)
        cv2.setMouseCallback('img',mouse_click3,para)
        key = 0

        red_label = (0,0,255)
        green_label = (0,255,0)
        clear_label = (0,0,0)

        # 手动选择标记点
        while key != 27:  # esc
            if para[0] == 0:
                cv2.circle(mask_img,(para[1],para[2]),1,red_label,-1)
            elif para[0] == 1:
                cv2.circle(mask_img,(para[1],para[2]),2,green_label,1)
            elif para[0] == 2:
                cv2.circle(mask_img,(para[1],para[2]),10,clear_label,-1)
            point_img_=cv2.add(mask_img,img_)
            cv2.imshow('img',point_img_)
            key = cv2.waitKey(30)

        cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

        h = mask_img.shape[0]
        w = mask_img.shape[1]

        for row in range(h):
            for col in range(w):
                if mask_img[row][col][2] == 255:
                    markers[row][col]=1
                    red_point.append([[col,row]]) #将光流追踪点添加进去
                elif mask_img[row][col][1] == 255:
                    markers[row][col]=2
                    green_point.append([[col,row]])

        markers = cv2.watershed(img_,markers=markers)
        img_res = np.copy(img_)

        # 显示边界
        img_[markers==-1] = [0,0,255]
        cv2.imshow('res_bonder_%d'%(i),img_)

        # 显示分割后的图像
        img_res[markers==1] = [255,255,255]
        img_res[markers!=1] = [0,0,0]
        #cv2.imshow('res_seg_%d'%(i),img_res)

        # 显示原图
        #cv2.imshow('origin_%d'%(i),img_bk)
        r_key = cv2.waitKey(0)
        if r_key == 114:  # r_key=='r'
            cv2.destroyAllWindows()
            continue
        else:
            cv2.destroyAllWindows()

        img_gray_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)

        img_origin_mask = cv2.add(origin_img,mask_img)

        cv2.imwrite(os.path.join(imgs_out_path,"%d_origin.tif"%i),origin_img)
        cv2.imwrite(os.path.join(imgs_out_path,"%d_mask.tif"%i),img_origin_mask)
        cv2.imwrite(os.path.join(imgs_out_path,"%d_res.tif"%i),img_gray_res)
        cv2.imwrite(os.path.join(imgs_out_path,"%d_edge.tif"%i),img_)
        break

    # 开始光流追踪
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1))
    max_point_dist = 15.0

    first_frame = np.copy(first_img)
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    first_frame_mask=np.zeros_like(first_frame)

    p0_red_point = np.array(red_point,dtype=np.float32)
    p0_green_point = np.array(green_point,dtype=np.float32)

    # 用来显示轨迹的mask
    mask_track = np.zeros_like(first_img)

    i = start_index+1
    while i != end_index:
        img_cur = cv2.imread(imgs_in%(i))
        img_cur_copy = np.copy(img_cur)
        gray_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY)

        p1_red_point, st_red, err_red = cv2.calcOpticalFlowPyrLK(old_gray, gray_cur, p0_red_point, None, **lk_params)
        p1_green_point, st_green, err_green = cv2.calcOpticalFlowPyrLK(old_gray, gray_cur, p0_green_point, None, **lk_params)

        # Select good points
        red_good_new = p1_red_point[st_red == 1]
        red_good_old = p0_red_point[st_red == 1]

        green_good_new = p1_green_point[st_green == 1]
        green_good_old = p0_green_point[st_green == 1]

        #将mask_img设为共用的
        #mask_img = np.zeros_like(img)

        markers = np.zeros_like(gray_cur,dtype=np.int32)

        red_delete_index = []
        green_delete_index = []

        # draw the red points tracks
        for j, (new, old) in enumerate(zip(red_good_new,red_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                red_delete_index.append(j)
            else:
                # 画出轨迹
                mask_track = cv2.line(mask_track, (a, b), (c, d), (50,20,50), 2)

                #去除旧的点
                mask_img[int(d)][int(c)] = [0,0,0]
                #cv2.circle(mask_img, (c, d),1,(0,0,0),-1)
                #添加新的点
                #cv2.circle(mask_img, (a, b), 1, (0,0,255), -1)
                if b < mask_img.shape[0] and a < mask_img.shape[1]:
                    mask_img[int(b)][int(a)] = [0,0,255]

                #if b<512 and a<512:
                #   markers_temp[int(b)][int(a)] = 1

        # draw the green points tracks
        for j, (new, old) in enumerate(zip(green_good_new,green_good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            #检查这个new点是否距离old点移动过远
            dist = np.sqrt(np.sum(np.square(new.ravel() - old.ravel())))

            #如果过远就舍弃
            if dist > max_point_dist:
                green_delete_index.append(j)
            else:
                # 画出轨迹
                mask_track = cv2.line(mask_track, (a, b), (c, d), (50,20,50), 2)

                #去除旧的点
                mask_img[int(d)][int(c)] = [0,0,0]
                #cv2.circle(mask_img, (c, d),1,(0,0,0),-1)
                #添加新的点
                #cv2.circle(mask_img, (a, b), 1, (0,0,255), -1)
                if b < mask_img.shape[0] and a < mask_img.shape[1]:
                    mask_img[int(b)][int(a)] = [0,255,0]

                #if b<512 and a<512:
                #   markers_temp[int(b)][int(a)] = 1

        #显示原图+标记+轨迹
        cv2.namedWindow('img')
        show_img = cv2.add(mask_img,img_cur)
        show_img = cv2.add(show_img,mask_track)
        cv2.imshow('img',show_img)
        #保存
        cv2.imwrite(os.path.join(imgs_out_path,"%d_track.tif"%i),show_img)
        cv2.waitKey(0)
        cv2.setMouseCallback('img',mouse_click3,para)
        key = 0

        red_label = (0,0,255)
        green_label = (0,255,0)
        clear_label = (0,0,0)

        # 手动选择标记点
        while key != 27:  # esc
            if para[0] == 0:
                cv2.circle(mask_img,(para[1],para[2]),1,red_label,-1)
            elif para[0] == 1:
                cv2.circle(mask_img,(para[1],para[2]),2,green_label,1)
            elif para[0] == 2:
                cv2.circle(mask_img,(para[1],para[2]),10,clear_label,-1)
            point_img=cv2.add(mask_img,img_cur)
            cv2.imshow('img',point_img)
            key = cv2.waitKey(30)

        cv2.circle(mask_img,(0,0),20,(0,0,0),-1)

        h = mask_img.shape[0]
        w = mask_img.shape[1]


        # 清空上一次的red_point
        red_point.clear()
        # 清空上一次的green_point
        green_point.clear()

        for row in range(h):
            for col in range(w):
                if mask_img[row][col][2] == 255:
                    markers[row][col]=1
                    red_point.append([[col,row]]) #将光流追踪点添加进去
                elif mask_img[row][col][1] == 255:
                    markers[row][col]=2
                    green_point.append([[col,row]])

        markers = cv2.watershed(img_cur,markers=markers)
        img_res = np.copy(img_cur)

        # 显示边界
        img_cur[markers==-1] = [0,0,255]
        cv2.imshow('res_bonder_%d'%(i),img_cur)

        # 显示分割后的图像
        #img_res[markers==1] = [0,0,255]
        img_res[markers==1] = [255,255,255]
        img_res[markers!=1] = [0,0,0]
        #cv2.imshow('res_seg_%d'%(i),img_res)

        #cv2.imshow('origin_%d'%(i),img_bk) #显示原图


        #光流后的处理
        p0_red_point = np.array(red_point,dtype=np.float32)
        p0_green_point = np.array(green_point,dtype=np.float32)
        old_gray = gray_cur.copy()

        #如果合适就按esc保存,否则按r重新处理
        r_key = cv2.waitKey(0)
        if r_key == 114:  # r_key=='r'
            cv2.destroyAllWindows()
            continue
        else:
            cv2.destroyAllWindows()

        img_gray_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2GRAY)

        img_origin_mask = cv2.add(img_cur_copy,mask_img)

        cv2.imwrite(os.path.join(imgs_out_path,"%d_mask.tif"%i),img_origin_mask)
        cv2.imwrite(os.path.join(imgs_out_path,"%d_res.tif"%i),img_gray_res)
        cv2.imwrite(os.path.join(imgs_out_path,"%d_edge.tif"%i),img_cur)
        i = i + 1

def shufflefile(imgin,labelin,imgout,labelout):
    '''
    打乱文件顺序
    :param pathin:
    :return:
    '''
    imgnum = 3202
    L = random.sample(range(0,imgnum),imgnum)
    for i in range(imgnum):
        shutil.copy(imgin%(i),imgout%(L[i]))
        shutil.copy(labelin%(i),labelout%(L[i]))
    return


def resGetCanny(origin_path,label_path,output_path):
    #for i in range(1062):
    for i in range(600):
        origin_img = cv2.imread(origin_path%(i))
        label_img = cv2.imread(label_path%(i))
        canny_img = cv2.Canny(label_img,50,150)
        res_img = np.copy(origin_img)
        res_img[canny_img==255]=[0,0,255]
        cv2.imwrite(output_path%(i),res_img)

def showRes(origin_path,label_path,method_path,output_path):
    i = 10
    #i = 24
    origin_img = cv2.imread(origin_path%(i))
    label_img = cv2.imread(label_path%(i))
    method_img = cv2.imread(method_path%(i))
    iou_img = np.zeros_like(origin_img)  # 只标注不重合的地方


    for h in range(method_img.shape[0]):
        for w in range(method_img.shape[1]):
            if method_img[h][w][2]==255:
                method_img[h][w]=[255,255,255]

    for h in range(label_img.shape[0]):
        for w in range(label_img.shape[1]):

            if label_img[h][w][0] ==255 and method_img[h][w][0]!=255:
                iou_img[h][w] = [0,0,255]
            if method_img[h][w][0] ==255 and label_img[h][w][0]!=255:
                iou_img[h][w] = [0,255,0]

            if label_img[h][w][0]==255:     #真实标签设为红色
                label_img[h][w] = [0,0,255]
            if method_img[h][w][0]==255:    #预测标签设为绿色
                method_img[h][w] = [0,255,0]
    true_res_img = cv2.addWeighted(origin_img,1,label_img,0.3,0)
    method_res_img = cv2.addWeighted(origin_img,1,method_img,0.3,0)
    add_res_img = cv2.addWeighted(true_res_img,1,method_img,0.3,0)
    iou_res_img = cv2.addWeighted(origin_img,1,iou_img,0.3,0)



    cv2.imshow("true_res",true_res_img)
    cv2.imshow("method_res",method_res_img)
    cv2.imshow("add_res",add_res_img)
    cv2.imshow("iou_res",iou_res_img)
    cv2.waitKey()
    cv2.imwrite(output_path+"/true_res.tif",true_res_img)
    cv2.imwrite(output_path+"/method_res.tif",method_res_img)
    cv2.imwrite(output_path+"/add_res.tif",add_res_img)
    cv2.imwrite(output_path+"/iou_res.tif",iou_res_img)


def getThreshold(imgin):
    img = cv2.imread(imgin%(1),cv2.COLOR_BGR2GRAY)
    start = time.time()
    ret,res = cv2.threshold(img,111,255,cv2.THRESH_BINARY)
    elapsed = (time.time()-start)
    print('消耗时间为: ',elapsed)
    cv2.imshow('res',res)
    cv2.waitKey()

def red2white(imgpath,savepath):
    for i in range(482,563):
        img = cv2.imread(imgpath%(i),cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img[img!=0] = 255
        cv2.imwrite(savepath%(i),img)



if __name__ == '__main__':
    #lishi_path = "E:/Work/pic/pic/6-5-XY/XY%04d.tif"
    #view_pixel_change(imgs_in)
    #get_histplot()
    #get_XZ(lishi_path)
    #save_XZ(imgs_in,imgs_XZ_out)
    #view_canny(imgs_XZ_out)
    #view_canny(lishi_path)


    #lk_optical_flow(imgs_in)
    #dense_optflow(imgs_in)

    # 基本的分水岭标注方法
    # watershed_seg(imgs_in2,imgs_label_out)

    #movedir(imgs_in,imgs_label_in,imgs_out,imgs_label_out)
    #shufflefile(imgs_in,imgs_label_in,imgs_out,imgs_label_out)

    #gammaTrans(imgs_in1%(1))

    #watershed_optical_seg(imgs_in2)

    #optical_watershed_iter(imgs_in2)
    #seedgrow(imgs_in)

    #optical_watershed_rnd_iter(imgs_in)
    #rename(rename_dir)

    #watershed_optical_label(imgs_in2,imgs_out2)

    #resGetCanny("E:/FIB_SEM_Pic/val/data/%d.tif","E:/FIB_SEM_Pic/val/label/%d.tif","E:/FIB_SEM_Pic/val/canny/%d.tif")
    showRes(res_imgs_in,res_imgs_label_in,method_in,res_imgs_out)
    #showRes(snake_origin_in,snake_label_in,snake_res_in,snake_out)
    #getThreshold(res_imgs_in)

    #red2white("E:/FIB_SEM_Pic/zmtsy1/origin/%d.BMP","E:/FIB_SEM_Pic/zmtsy1/white/%d.BMP")



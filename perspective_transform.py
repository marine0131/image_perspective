# -*- coding:utf-8 -*-
import cv2
import numpy as np
import sys

img = cv2.imread('test.jpg')
# cv2.imshow("original", img)

# expand image
img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
h, w = img.shape[0:2]

anglex = 0
angley = 0
anglez = 0 
fov = 60
r = 0

def rad(x):
    return x * np.pi / 180

def get_warpR():
    global anglex,angley,anglez,fov,w,h,r
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    # rx = np.array([[1, 0, 0, 0],
    #                [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
    #                [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
    #                [0, 0, 0, 1]], np.float32)

    # ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
    #                [0, 1, 0, 0],
    #                [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
    #                [0, 0, 0, 1]], np.float32)

    # rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
    #                [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]], np.float32)

    # r = rx.dot(ry).dot(rz)
    sinx = np.sin(rad(anglex))
    cosx = np.cos(rad(anglex))
    siny = np.sin(rad(angley))
    cosy = np.cos(rad(angley))
    sinz = np.sin(rad(anglez))
    cosz = np.cos(rad(anglez))
    # rotate matrix
    r = np.array([[ cosy*cosz, -cosx*sinz+sinx*siny*cosz, sinx*sinz+cosx*siny*cosz, 0],
                  [ cosy*sinz, cosx*cosz+sinx*siny*sinz, -sinx*cosz+cosx*siny*sinz, 0],
                  [ -siny,     sinx*cosy,                cosx*cosy,                0],
                  [ 0,         0,                        0,                         1]], np.float32)

    # four pair points
    pcenter = np.array([w / 2, h / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    # rotate these points
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # projection to iamge plane
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z+list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z+list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    return warpR

def control(c):
    global anglex,angley,anglez,fov,r

    # 键盘控制
    if 27 == c:  # Esc quit
        sys.exit()
    if c == ord('w'):
        anglex += 1
    if c == ord('s'):
        anglex -= 1
    if c == ord('a'):
        angley += 1
        print(angley)
        # dx=0
    if c == ord('d'):
        angley -= 1
    if c == ord('u'):
        anglez += 1
    if c == ord('p'):
        anglez -= 1
    if c == ord('t'):
        fov += 1
    if c == ord('r'):
        fov -= 1
    if c == ord(' '):
        anglex = angley = anglez = 0
    if c == ord('e'):
        print("======================================")
        print('Rotation Matrix:')
        print(r)
        print('angle alpha(anglex):')
        print(anglex)
        print('angle beta(angley):')
        print(angley)
        print('dz(anglez):')
        print(anglez)


while True:
    warpR = get_warpR()

    result = cv2.warpPerspective(img, warpR, (w, h))
    cv2.namedWindow('result',2)
    cv2.imshow("result", result)
    c = 0xFF & cv2.waitKey(30)
    control(c)

cv2.destroyAllWindows()

# -*- coding:utf-8 -*-
import cv2
import numpy as np
import sys

roll = 0 
pitch = 0
yaw = 0
fov = 60

def rad(x):
    return x * np.pi / 180

def get_warpR(h, w):
    global roll,pitch,yaw,fov
    # rotate matrix of roll pitch yaw
    # rx = np.array([[1, 0, 0, 0],
    #                [0, np.cos(rad(roll)), -np.sin(rad(roll)), 0],
    #                [0, -np.sin(rad(roll)), np.cos(rad(roll)), 0, ],
    #                [0, 0, 0, 1]], np.float32)

    # ry = np.array([[np.cos(rad(pitch)), 0, np.sin(rad(pitch)), 0],
    #                [0, 1, 0, 0],
    #                [-np.sin(rad(pitch)), 0, np.cos(rad(pitch)), 0, ],
    #                [0, 0, 0, 1]], np.float32)

    # rz = np.array([[np.cos(rad(yaw)), np.sin(rad(yaw)), 0, 0],
    #                [-np.sin(rad(yaw)), np.cos(rad(yaw)), 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]], np.float32)

    # r = rx.dot(ry).dot(rz)
    sinx = np.sin(rad(roll))
    cosx = np.cos(rad(roll))
    siny = np.sin(rad(pitch))
    cosy = np.cos(rad(pitch))
    sinz = np.sin(rad(yaw))
    cosz = np.cos(rad(yaw))
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
    global roll,pitch,yaw,fov

    # keyboard control
    if 27 == c:  # Esc quit
        sys.exit()
    if c == ord('w'):
        roll += 1
    if c == ord('s'):
        roll -= 1
    if c == ord('a'):
        pitch += 1
    if c == ord('d'):
        pitch -= 1
    if c == ord('u'):
        yaw += 1
    if c == ord('p'):
        yaw -= 1
    if c == ord('t'):
        fov += 1
    if c == ord('r'):
        fov -= 1
    if c == ord(' '):
        roll = pitch = yaw = 0
    if c == ord('e'):
        print("======================================")
        print('angle alpha(roll):')
        print(roll)
        print('angle beta(pitch):')
        print(pitch)
        print('dz(yaw):')
        print(yaw)

if __name__=="__main__":
    img = cv2.imread('test.jpg')
    # cv2.imshow("original", img)

    # expand image
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)
    h, w = img.shape[0:2]


    # get distance from image to camera
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    while True:
        warpR = get_warpR(h, w)
        result = cv2.warpPerspective(img, warpR, (w, h))
        cv2.namedWindow('result',2)
        cv2.imshow("result", result)
        c = 0xFF & cv2.waitKey(1)
        control(c)

cv2.destroyAllWindows()

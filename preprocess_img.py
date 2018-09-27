import cv2
import numpy as np
from PIL import Image
from os import listdir
import tensorflow as tf
import multiprocessing as mp
from itertools import product
import time

file_path = "./Steve_Jobs.0.jpg"
folder_path = "C:/Users/thang_dinh/Documents/個人データセット/Train_Cartoon"

def convert_img_to_nparray(file_path):
    img = Image.open(file_path, mode="r").convert("RGB")
    img = img.resize((300,300))
    # img.show()
    result = tf.keras.preprocessing.image.img_to_array(img)
    shape = result.shape
    return result

def convert_nparray_to_img(nparray):
    img = tf.keras.preprocessing.image.array_to_img(nparray)
    img.show()

def list_all_file(folder):
    result = []
    dirs = listdir(folder)
    print(dirs)
    for dir in dirs:
        files = listdir(folder+"/"+dir)
        result += [dir +"/"+ file for file in files]
    print("There are {} files in folder {}".format(len(result), folder))
    return result

def create_dataset():
    files_array = list_all_file(folder_path)
    data = []

    #don't use parallel program
    start = time.time()
    for file in files_array:
        result_ = convert_img_to_nparray(folder_path+"/"+file)
    print("Process in {}".format(time.time()-start))

    #parrallel using "Pool" class and "starmap" method of Multiprocess package
    #allow for multiple parameters with tuple type
    start = time.time()
    with mp.Pool(processes=4) as pool:
        result = pool.starmap(convert_img_to_nparray, product([folder_path + "/" + file for file in files_array]))
    print("Process in {}".format(time.time() - start))

    #parrallel using "Pool" and "map" method of Multiprocess package
    # with mp.Pool(processes=4) as pool:
    #     result = pool.map(convert_img_to_nparray, [folder_path + "/" + file for file in files_array])
    #     print(len(result[0]))
    #     convert_nparray_to_img(result[0])

    for arr in result:
        data.append(arr)
    data = np.asarray(data)
    print(type(data))
    return data
if __name__ == "__main__":
    create_dataset()


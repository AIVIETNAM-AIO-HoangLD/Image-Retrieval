import os
import numpy as np
from PIL import Image

class DataLoader:
    def __init__(self, data_dir: str, size: tuple):
        self.data_dir = data_dir
        self.image_paths =[]
        self.size = size
        self.image_path = []
        self.image_np = None

    
    def read_images_from_path(self, size: tuple, path_img: str):
        self.size = size
        img = Image.open(path_img).convert("RGB").resize(self.size)
        return np.array(img)
    
    def folder_to_images(self):
        self.list_dir = [self.data_dir + name for name in os.listdir(self.data_dir)]
        self.image_np = np.zeros(shape=(len(self.list_dir), *self.size, 3), dtype=np.uint8)
        for i, path in enumerate(self.list_dir):
            for pic in os.listdir(path):
                self.image_np[i] = self.read_images_from_path(self.size, str(path + '/' + pic))
                self.image_path.append(str(path + '/' + pic))
                print(str(path + '/' + pic))
                break
        self.image_paths = np.array(self.image_path)
        return self.image_np, self.image_paths

#data_path = r"./data/Dataset/animal/"
#data_loader = DataLoader(data_path, size=(448, 448))
#
#img_np,_ = data_loader.folder_to_images()
#
#print(img_np.shape)
import numpy as np
import os
from PIL import Image
import cv2

from src.data_loader import DataLoader
from src.similarity import Similarity

data_path = r"./data/Dataset/animal/"
data_loader = DataLoader(data_path, size=(448, 448))

img_np,_ = data_loader.folder_to_images()
QUERY = np.array(Image.open(r"/home/hoangLD/Desktop/AIVIETNAM/Module-02/M02W01/Image_Retrieval/data/Dataset/animal/Spider/6208284939_e31f22f9d9_b.jpg").convert("RGB").resize((448, 448)))
retrieval = Similarity(img_np, QUERY)

result= retrieval.get_correlation_coefficient()
result_min_idx = np.argsort(result)[-1]

predict = img_np[result_min_idx].astype(np.uint8)
print(predict.shape)
print(result)
print(result[result_min_idx])
cv2.imshow("",predict)
cv2.waitKey(0)
cv2.destroyAllWindows()

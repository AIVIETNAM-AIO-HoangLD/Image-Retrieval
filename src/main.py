import numpy as np
import os
from PIL import Image
import cv2

from src.data_loader import DataLoader
from src.similarity import Similarity

data_path = r"./data/Dataset/animal/"
data_loader = DataLoader(data_path, size=(448, 448))
QUERY = np.array(Image.open(r"/home/hoangLD/Desktop/AIVIETNAM/Module-02/M02W01/Image_Retrieval/data/Dataset/animal/bear/6951618736_88e63789f2_b.jpg").convert("RGB").resize((448, 448)))
imgs_vec = data_loader.embedding_images()
img_np,_ = data_loader.folder_to_images()
query_vec = data_loader.embedding_images(pic=QUERY)

retrieval = Similarity(imgs_vec, query_vec)
result= retrieval.get_cosine_similarity()
result_min_idx = np.argsort(result)[-1]
#
predict = img_np[result_min_idx].astype(np.uint8)
print(result)
print(result[result_min_idx])
cv2.imshow("",predict)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import os
from PIL import Image
import cv2

from data_loader import DataLoader
from similarity import Similarity

data_path = r"./data/Dataset/animal/"
data_loader = DataLoader(data_path, size=(448, 448))
QUERY = np.array(Image.open(r"/home/hoangLD/Desktop/AIVIETNAM/Module-02/M02W01/Image_Retrieval/data/Dataset/animal/panda/3523034831_4cd64c43dc_b.jpg").convert("RGB").resize((448, 448)))
imgs_vec = data_loader.embedding_images()
img_np,_ = data_loader.folder_to_images()
query_vec = data_loader.embedding_images(pic=QUERY)
retrieval = Similarity(imgs_vec, query_vec)
result= retrieval.get_cosine_similarity()
print(result)
result_min_idx = np.argmax(result)
predict = img_np[result_min_idx,:,:,:].astype(np.uint8)
cv2.imshow(f"Similarity: {result}",predict)

retrieval2 = Similarity(img_np, QUERY)
result2= retrieval2.get_cosine_similarity()
result_min_idx2 = np.argmax(result2)
print(result_min_idx2)
print(result2)
predict2 = img_np[result_min_idx2,:,:,:].astype(np.uint8)
cv2.imshow(f"Similarity: {result2}",predict2)


cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
from data_loader import DataLoader
from similarity import Similarity
from PIL import Image
import cv2
data_path = r"./data/Dataset/animal/"
data_loader = DataLoader(data_path, size=(448, 448))

img_np,_ = data_loader.folder_to_images()
QUERY = np.array(Image.open(r"./data/Dataset/animal/pig/14898204640_7c059c9ac4_b.jpg").convert("RGB").resize((448, 448)))
retrieval = Similarity(img_np, QUERY)

result= retrieval.get_l2_score()
result_min_idx = np.argsort(result)[0]

predict = img_np[result_min_idx].astype(np.uint8)
print(predict.shape)
print(result)
print(result[result_min_idx])
cv2.imshow("",predict)
cv2.waitKey(0)
cv2.destroyAllWindows()

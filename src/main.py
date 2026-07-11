import numpy as np
from data_loader import DataLoader
from similarity import Similarity
from PIL import Image
import matplotlib.pyplot as plt
data_path = r"./data/Dataset/animal/"
data_loader = DataLoader(data_path, size=(448, 448))

img_np,_ = data_loader.folder_to_images()
QUERY = np.array(Image.open(r"/home/hoangLD/Desktop/AIVIETNAM/Module-02/M02W01/Image_Retrieval/data/Dataset/animal/pig/2389683177_2ef554b8d0_b.jpg").convert("RGB").resize((448, 448)))
retrieval = Similarity(img_np, QUERY)

result= retrieval.get_f1_score()
result_max_idx = np.argsort(result)
plt.imshow(img_np[result_max_idx][0].astype(np.uint8))
plt.savefig("result.png")
print(sorted(result))


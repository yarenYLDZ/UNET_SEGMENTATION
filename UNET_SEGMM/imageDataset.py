from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image 
import os
import numpy as np

dataset_path="C:/Users/90538/OneDrive/Desktop/Tez Projesi Veri Setleri/Dataset_BUSI_with_GT/Dataset_BUSI_with_GT/benign"

image_files=os.listdir(dataset_path)
#image_files = sorted(image_files, key=numeric_sort)
image_files=[file for file in image_files if file.endswith(".png")and "mask" not in file]


images = []
masks = []
labels=[]

#aleyna yarennnnn

for file in image_files:
    # Görüntü yükleme

    image_path = os.path.join(dataset_path, file)
    image = Image.open(image_path)  # Görüntüyü renk dönüşümü yapmadan yükle
    image = image.resize((512, 512))  # Görüntüyü 512x512 boyutuna yeniden boyutlandır
    images.append(image)
    
    # Maske yükleme
    mask_path = os.path.join(dataset_path, file.replace(".png", "_mask"))  # Maske dosyasının yolunu oluştur
    mask_path = mask_path + ".png"
    mask = Image.open(mask_path)  # Maskelenmiş görüntüyü yükle
    mask = mask.resize((512, 512))  # Maskelenmiş görüntüyü 512x512 boyutuna yeniden boyutlandır
    masks.append(mask)

    label = file.split("_")[0]  # Örneğin, benign veya malignant
    labels.append(label)


# Görüntü ve maskeleri kontrol edin
# for i in range(len(images)):
#     print("Görüntü boyutu:", images[i].size)
#     print("Maske boyutu:", masks[i].size)

#Dataseti egitim ve test olarak ayirdik
labels = np.array(labels)
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

# images_array = np.array(images)
# masks_array = np.array(masks)

for train_index, test_index in strat_split.split(images, labels):

    train_data = [images[i] for i in train_index]
    train_targets = [masks[i] for i in train_index]
    test_data = [images[i] for i in test_index]
    test_targets = [masks[i] for i in test_index]

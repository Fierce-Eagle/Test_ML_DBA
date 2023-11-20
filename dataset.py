import os
import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import albumentations as A


class DataLoad:
    def __init__(self, path):
        self.image_dir = path

        filenames = os.listdir(self.image_dir)
        labels = [x.split(".")[0].split("_")[2] for x in filenames]

        self.data = pd.DataFrame({"image": filenames, "label": labels})


def split_data(dataset):
    labels = dataset['label']
    x_train, x_temp = train_test_split(dataset, test_size=0.2, stratify=labels, random_state=42)

    label_val_test = x_temp['label']
    x_valid, x_test = train_test_split(x_temp, test_size=0.5, stratify=label_val_test, random_state=42)

    return x_train, x_valid, x_test


class CustomDataset(Dataset):
    def __init__(self, data_frame):
        """
        Класс обработки датасета

        :param data_frame: датасет вида pd.DataFrame({filename: , label: })
        """
        self.data_frame = data_frame

    def __len__(self):
        """
        Число всех картинок в датасете
        :return:
        """
        return len(self.data_frame["image"])

    def __getitem__(self, idx):
        """
        Получение картинки из датасета

        :param idx: позиция картинки в датасете
        :return:
        """
        return self.data_frame["image"][idx], self.data_frame["label"][idx]


def augmented_dataset(path, df_main):
    tmp_df = pd.DataFrame()

    for image_name, label in zip(df_main['image'], df_main['label']):
        # [000] = 2186
        # [001] = 545
        # [010] = 2008
        # [011] = 5
        # [100] = 1226
        # [101] = 8
        # [110] = 22
        # [111] = 0 - нет данных
        if label == '000':
            _label = 0
        elif label == '001':
            _label = 1
        elif label == '010':
            _label = 2
        elif label == '011':
            _label = 3
        elif label == '100':
            _label = 4
        elif label == '101':
            _label = 5
        else:
            _label = 6
        new_label = torch.Tensor([_label])

        img = Image.open(path + image_name)
        _img = np.array(img)
        img.close()
        image_dict = resize_augmentation(image=_img)
        new_image = torch.Tensor(image_dict['image'])
        new_image = np.transpose(new_image, (2, 0, 1))
        tmp_df = pd.concat([tmp_df, pd.DataFrame({'image': [new_image], 'label': [new_label]})], ignore_index=True)

        if _label == 3:  # 5 * 40 = 200
            for i in range(40):

                image_dict = new_image_augmentation(image=_img)
                new_image = torch.Tensor(image_dict['image'])
                new_image = np.transpose(new_image, (2, 0, 1))
                tmp_df = pd.concat([tmp_df, pd.DataFrame({'image': [new_image], 'label': [new_label]})], ignore_index=True)

        elif _label == 5:  # 8 * 25 = 200
            for i in range(25):

                image_dict = new_image_augmentation(image=_img)
                new_image = torch.Tensor(image_dict['image'])
                new_image = np.transpose(new_image, (2, 0, 1))
                tmp_df = pd.concat([tmp_df, pd.DataFrame({'image': [new_image], 'label': [new_label]})], ignore_index=True)

        elif _label == 6:  # 22 * 10 = 220
            for i in range(10):

                image_dict = new_image_augmentation(image=_img)
                new_image = torch.Tensor(image_dict['image'])
                new_image = np.transpose(new_image, (2, 0, 1))
                tmp_df = pd.concat([tmp_df, pd.DataFrame({'image': [new_image], 'label': [new_label]})], ignore_index=True)

    return tmp_df


new_image_augmentation = A.Compose([
    # изменение размеров картинки
    A.Resize(256, 256),
    # применяемые агументации
    A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomShadow(p=0.2),
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
], p=1)

resize_augmentation = A.Compose([
    # изменение размеров картинки
    A.Resize(256, 256),
    # нормализация
    A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
], p=1)

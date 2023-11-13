from torch.utils.data import Dataset
import cv2
import os
import glob
import torch
import numpy as np

class EMODataset(Dataset):
    def __init__(self, img_txt_dir, subset, shape, max_length, padding=False):
        self.img_txt_dir = img_txt_dir # общая директория к обучающим и тестовым данным
        self.subset = subset # имя папки с обучающими или тестовыми данными

        self.shape = shape # параметр обучения - высоты, ширины изображения
        self.max_length = max_length # параметр обучения - максимальная длина последовательности 
        self.padding = padding # параметр обучения - выполнить padding изображения лица нулями mode='constant' (можно рассмотреть другие варианты, например mode='mean') если padding=True, либо resize

        # находим все обучающие или тестовые данные
        self.name_video = os.listdir(os.path.join(self.img_txt_dir, self.subset))
        # выбираем только папки, исключаем .cache, оставить только уникальные имена
        self.name_video = list(set(filter(lambda x: os.path.isdir(os.path.join(self.img_txt_dir, self.subset, x)) and not x.endswith('.cache'), self.name_video)))

    def make_padding(self, img):
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = self.shape[0] / img.shape[0]
            factor_1 = self.shape[1] / img.shape[1]
            factor = min(factor_0, factor_1)
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            diff_0 = self.shape[0] - img.shape[0]
            diff_1 = self.shape[1] - img.shape[1]
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        if img.shape[0:2] != self.shape:
            img = cv2.resize(img, self.shape)   
        return img
    
    def get_sequence_given_length(self, faces):
        """
        # написать метод для получения последовательности фиксированной длины self.max_length, который должен удовлетворять условиям:
        # 1) если текущая длина последовательности больше self.max_length, то обрезаем последовательность до нужной длины
        # 2) если меньше, то дублируем исходную последовательность необходимое количество раз до нужной длины
        """

        current_length = len(faces)
    
        if current_length >= self.max_length:
            faces = faces[:self.max_length]
        elif current_length < self.max_length:
            num_duplicates = (self.max_length // current_length) + 1
            faces = torch.tile(faces, (num_duplicates, 1, 1, 1))[:self.max_length]
        return faces

    def prepare_data(self, name_file):
        # найдем все изображения '*.jpg' относящиеся к текущему видео
        imgs = glob.glob(os.path.join(self.img_txt_dir, self.subset, name_file, '*.jpg'))

        # создаем пустой список для записи областей лиц
        norm_imgs = []
        # обрабатываем каждое изображение последовательно
        for img in imgs:
            # считываем изображение c помощью cv2.imread и конвертируем с BGR на RGB
            im = cv2.imread(img)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # считываем соответствующий текущему изображению текстовый файл '.txt'
            path_txt = os.path.join(self.img_txt_dir, self.subset, name_file, os.path.splitext(os.path.basename(img))[0] + '.txt')
            with open(path_txt) as f:
                txt = f.read()
                # извлекаем из текстового файла метку класса, центр лица по x, центр лица по y, высоту лица, ширину лица
                id_cl, cx, cy, w, h = [float(x) for x in txt.split()]
                # получаем высоту, ширину кадра
                h0, w0 = im.shape[:2]
                
                # находим x,y координаты левого верхнего угла лица
                x = (cx - w/2) * w0
                y = (cy - h/2) * h0

                # вырезаем область лица из исходного изображения по принципу im[startY: endY, startX: endX]
                startX = int(max(0, x))
                startY = int(max(0, y))
                endX = int(min(w0, x + w * w0))
                endY = int(min(h0, y + h * h0))
                im = im[startY:endY, startX:endX]

                # делаем padding нулями или resize изображения области лица
                # если self.padding is true, то выполняем метод self.make_padding()
                # в противном случае выполняем cv2.resize()
                # изменяем dtype массива с int на float32
                if self.padding:
                    im = self.make_padding(im)
                else:
                    
                    im = cv2.resize(im, self.shape)
                    im = im.astype(np.float32)
                
                # нормализовать изображение от 0 до 1
                im = im.astype(np.float32) / 255.0
                # записываем все изображения в список чтобы создать последовательность
                norm_imgs.append(im)
                
        # приводим np.array к torch.tensor
        faces = np.array(norm_imgs)
        faces = torch.from_numpy(faces)
        # написать метод self.get_sequence_given_length() для получения последовательности фиксированной длины self.max_length, который должен удовлетворять условиям:
        # 1) если текущая длина последовательности больше self.max_length, то обрезаем последовательность до нужной длины
        # 2) если меньше, то дублируем исходную последовательность необходимое количество раз до нужной длины
        # вызываем метод self.get_sequence_given_length()
        faces = self.get_sequence_given_length(faces)
        # возвращаем последовательность областей лиц и метку класса типа int
        return faces, int(id_cl)

    def __getitem__(self, index):
        # проходимся по self.name_video по индексу
        # извлекаем последовательность областей лиц и метку класса с помощью метода self.prepare_data()
        faces, id_cl = self.prepare_data(self.name_video[index])
        # изменяем порядок элементов в тензоре областей лиц с (depth, height, width, channels) на (channels, depth, height, width)  
        faces = faces.permute(3, 0, 1, 2)
        # возвращаем последовательность областей лиц и метку класса
        return faces, id_cl
            
    def __len__(self):
        return len(self.name_video)
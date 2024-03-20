import json
import random
# from torch.utils.data import DataLoader
from data.dataset import CocoDetection
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from general_config import constants, general_config

import mindspore as ms
from mindspore.dataset import SequentialSampler
from mindspore import ops
class MyDataLoader:
    def __init__(self, dataset, batch_size=4, shuffle=False):
        """
        初始化 MyDataLoader 类

        参数：
        dataset: 数据集，一个包含样本的列表或数组
        batch_size: 批次大小，即每个批次中包含的样本数，默认为 1
        shuffle: 是否对数据进行洗牌，默认为 False
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(dataset)))
        self.current_index = 0
        self.batch_idx =0
        self.lenth = len(self.dataset)//self.batch_size
        self.set_size = self.lenth*self.batch_size
        if self.shuffle:
            random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if (self.current_index+self.batch_size) > len(self.dataset):
            if self.shuffle:
                random.shuffle(self.indexes)  # 重新洗牌数据集
            # self.current_index = 0  # 重置当前索引
            raise StopIteration
            # raise StopIteration
        self.batch_idx +=1
        batch_indexes = self.indexes[self.current_index:self.current_index + self.batch_size]
        batch_images, label, image_info = self.dataset[batch_indexes]
        batch_idx = self.batch_idx
        self.current_index=self.current_index+self.batch_size
        return batch_idx,batch_images,label,image_info
    
    def __len__(self):
        return self.lenth
    

def get_dataloaders(params):
    ''' creates and returns train and validation data loaders '''

    train_dataloader = get_train_dataloader(params)
    valid_dataloader = get_valid_dataloader(params)

    return train_dataloader, valid_dataloader


def get_test_dev(params):
    test_annotations_path = constants.test_annotations_path
    test_dataset = CocoDetection(root=constants.test_images_folder,
                                 annFile=test_annotations_path,
                                 augmentation=False,
                                 params=params,
                                 run_type="test")

    with open(test_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_test = len(data['images'])
    sampler = SequentialSampler()
    dataloader = ms.dataset.GeneratorDataset(test_dataset, column_names=["data"], sampler=sampler)
    dataloader = dataloader.batch(params.batch_size,drop_remainder=False)

    return dataloader

def get_dataloaders_test(params):
    return get_valid_dataloader(params)


def get_train_dataloader(params):
    train_annotations_path = constants.train_annotations_path
    train_dataset = CocoDetection(root=constants.train_images_folder,
                                  annFile=train_annotations_path,
                                  augmentation=True,
                                  params=params)

    with open(train_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_train = len(data['images'])
    dataloader = MyDataLoader(dataset=train_dataset,batch_size=params.batch_size, shuffle=True)
    return dataloader
    # sampler = SequentialSampler()
    # dataloader = ms.dataset.GeneratorDataset(train_dataset, column_names=["data"],num_parallel_workers=1)
    # dataloader = dataloader.batch(params.batch_size,drop_remainder=True)
    # return train_dataset
    # return DataLoader(train_dataset, batch_size=None,
    #                   shuffle=False, num_workers=general_config.num_workers,
    #                   sampler=BatchSampler(SubsetRandomSampler([i for i in range(nr_images_in_train)]),
    #                                        batch_size=params.batch_size, drop_last=True))


def get_valid_dataloader(params):
    val_annotations_path = constants.val_annotations_path
    validation_dataset = CocoDetection(root=constants.val_images_folder,
                                       annFile=val_annotations_path,
                                       augmentation=False,
                                       params=params)

    with open(val_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_val = len(data['images'])
    dataloader = MyDataLoader(dataset=validation_dataset,batch_size=params.batch_size, shuffle=True)
    # sampler = SequentialSampler()
    # dataloader = ms.dataset.GeneratorDataset(validation_dataset, column_names=["data"], sampler=sampler)
    # dataloader = dataloader.batch(params.batch_size,drop_remainder=False)
    return dataloader
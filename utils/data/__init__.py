'''
Descripttion: 
version: 1.0
Author: ShaHao
Date: 2021-04-15 10:53:26
LastEditors: ShaHao
LastEditTime: 2021-04-24 23:12:46
'''
def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard' or "gmd":
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))
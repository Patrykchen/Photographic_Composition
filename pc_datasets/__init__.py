"""
Created on Tue April 19 2022
@author: Wang Zhicheng
"""

def build_datasets(args):
    if args.dataset_file == 'KU':
        from pc_datasets.KU_PCP.loading_data import loading_data
        return loading_data
    return None

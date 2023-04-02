import os
import numpy as np

DATA_ROOT_DIR = '../../data/traindata_example/'
VGN_TRAIN_ROOT = DATA_ROOT_DIR + 'giga_hemisphere_train_demo'

def add_scenes(root, type, filter_list=None):
    scene_names = []
    splits = os.listdir(root)
    for split in splits:
        if filter_list is not None and split not in filter_list: continue
        scenes = os.listdir(os.path.join(root, split))
        scene_names += [f'vgn_syn/train/{type}/{split}/{fn}/w_0.8' for fn in scenes]
    return scene_names
if os.path.exists(VGN_TRAIN_ROOT):
    vgn_pile_train_scene_names = sorted(add_scenes(os.path.join(VGN_TRAIN_ROOT, 'pile_full'), 'pile'), key=lambda x: x.split('/')[4])
    vgn_pack_train_scene_names = sorted(add_scenes(os.path.join(VGN_TRAIN_ROOT, 'packed_full'), 'packed'), key=lambda x: x.split('/')[4])
    num_scenes_pile = len(vgn_pile_train_scene_names)
    num_scenes_pack = len(vgn_pack_train_scene_names)
    vgn_pack_train_scene_names = vgn_pack_train_scene_names[:num_scenes_pack]
    num_val_pile = 1
    num_val_pack = 1
    print(f"total: {num_scenes_pile + num_scenes_pack} pile: {num_scenes_pile} pack: {num_scenes_pack}")
    vgn_val_scene_names = vgn_pile_train_scene_names[-num_val_pile:]  + vgn_pack_train_scene_names[-num_val_pack:]
    vgn_train_scene_names = vgn_pile_train_scene_names[:-num_val_pile]  + vgn_pack_train_scene_names[:-num_val_pack]

VGN_SDF_DIR = DATA_ROOT_DIR + "giga_hemisphere_train_demo/scenes_tsdf_dep-nor"

VGN_TEST_ROOT = ''
VGN_TEST_ROOT_PILE = os.path.join(VGN_TEST_ROOT,'pile')
VGN_TEST_ROOT_PACK = os.path.join(VGN_TEST_ROOT,'packed')
if os.path.exists(VGN_TEST_ROOT):
    fns = os.listdir(VGN_TEST_ROOT_PILE)
    vgn_pile_test_scene_names = [f'vgn_syn/test/pile//{fn}/w_0.8' for fn in fns]
    fns = os.listdir(VGN_TEST_ROOT_PACK)
    vgn_pack_test_scene_names = [f'vgn_syn/test/packed//{fn}/w_0.8' for fn in fns]

    vgn_test_scene_names = vgn_pile_test_scene_names + vgn_pack_test_scene_names

CSV_ROOT = DATA_ROOT_DIR + 'GIGA_demo'
import pandas as pd
from pathlib import Path
import time
t0 = time.time()
VGN_PACK_TRAIN_CSV = pd.read_csv(Path(CSV_ROOT + '/data_packed_train_processed_dex_noise/grasps.csv'))
VGN_PILE_TRAIN_CSV = pd.read_csv(Path(CSV_ROOT + '/data_pile_train_processed_dex_noise/grasps.csv'))
print(f"finished loading csv in {time.time() - t0} s")
VGN_PACK_TEST_CSV = None 
VGN_PILE_TEST_CSV = None 

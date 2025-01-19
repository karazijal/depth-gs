# %%
from pathlib import Path
import shutil
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image
import json

T = json.load(open('/scratch/local/ssd/laurynas/omni-data-temp/lego_200/transforms.json'))

SEQ = 'lego_200'

train_T = {'frames': []}
test_T = {'frames': []}
train_T['camera_angle_x'] = T['camera_angle_x']
test_T['camera_angle_x'] = T['camera_angle_x']

base = Path(f'/scratch/local/ssd/laurynas/omni-data-temp/{SEQ}/')
train_dir = base / 'train'
test_dir = base / 'test'
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

train_depth_dir = train_dir.parent / 'depths_train'
train_depth_dir.mkdir(exist_ok=True)
train_mask_dir = train_dir.parent / 'masks_train'
train_mask_dir.mkdir(exist_ok=True)
test_depth_dir = test_dir.parent / 'depths_test'
test_depth_dir.mkdir(exist_ok=True)
test_mask_dir = test_dir.parent / 'masks_test'
test_mask_dir.mkdir(exist_ok=True)

TRAIN_IDS = list(range(0, 100, 5))

for idx in tqdm(range(len(T['frames']))):
    img_path = base / f'r_{idx}.png'
    mask = np.array(Image.open(img_path).convert('RGBA'))[..., -1] > 0
    mask = mask.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    depth_path = base / f'r_{idx}_depth_0001.png'
    depth_img = np.array(Image.open(depth_path))
    depth = depth_img.astype(float)[...,0] / 255.0
    depth = 1.0 - depth
    depth = depth * 8.0
    new_frame = copy.deepcopy(T['frames'][idx])
    if idx in TRAIN_IDS:
        shutil.copy2(img_path, train_dir / img_path.name)
        np.save(train_depth_dir / img_path.with_suffix('.npy').name, depth)
        mask.save(train_mask_dir / img_path.name)
        new_frame['file_path'] = './train/' + img_path.stem
        train_T['frames'].append(new_frame)
    else:
        shutil.copy2(img_path, test_dir / img_path.name)
        np.save(test_depth_dir / img_path.with_suffix('.npy').name, depth)
        new_frame['file_path'] = './test/' + img_path.stem
        mask.save(test_mask_dir / img_path.name)
        test_T['frames'].append(new_frame)

with open(train_dir.parent / 'transforms_train.json', 'w') as f:
    json.dump(train_T, f, indent=4)

with open(test_dir.parent / 'transforms_test.json', 'w') as f:
    json.dump(test_T, f, indent=4)
    
# %%

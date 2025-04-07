import sys
import os
import jax
from matplotlib import pyplot as plt
import numpy as np
from scenic.projects.owl_vit import configs
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform
from scenic.projects.owl_vit import models
import time


sys.path.append('~/scenic/scenic/projects/owl_vit/big_vision/')
config = configs.owl_v2_clip_b16.get_config(init_mode='canonical_checkpoint')
module = models.TextZeroShotDetectionModule(
    body_configs=config.model.body,
    objectness_head_configs=config.model.objectness_head,
    normalize=config.model.normalize,
    box_bias=config.model.box_bias)

variables = module.load_variables(config.init_from.checkpoint_path)

filename = 'realsense_rgb.png'
# filename = 'headcamera_rgb.png'
start_time = time.time()
image_uint8 = skimage_io.imread(filename)
image = image_uint8.astype(np.float32) / 255.0

h, w, _ = image.shape
size = max(h, w)
image_padded = np.pad(
    image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

input_image = skimage.transform.resize(
    image_padded,
    (config.dataset_configs.input_size, config.dataset_configs.input_size),
    anti_aliasing=True)

text_queries = ['bowl']
tokenized_queries = np.array([
    module.tokenize(q, config.dataset_configs.max_query_length)
    for q in text_queries
])

tokenized_queries = np.pad(
    tokenized_queries,
    pad_width=((0, 100 - len(text_queries)), (0, 0)),
    constant_values=0)


jitted = jax.jit(module.apply, static_argnames=('train',))
predictions = jitted(
    variables,
    input_image[None, ...],
    tokenized_queries[None, ...],
    train=False)

predictions = jax.tree_util.tree_map(lambda x: np.array(x[0]), predictions )


score_threshold = 0.2

logits = predictions['pred_logits'][..., :len(text_queries)]  
scores = sigmoid(np.max(logits, axis=-1))
labels = np.argmax(predictions['pred_logits'], axis=-1)
boxes = predictions['pred_boxes']

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(input_image, extent=(0, 1, 1, 0))
ax.set_axis_off()

for score, box, label in zip(scores, boxes, labels):
    if score < score_threshold:
        continue
    cx, cy, w, h = box
    print(cx*640, cy*640)
    ax.plot(cx, cy, 'ro')
    ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
    ax.text(
        cx - w / 2,
        cy + h / 2 + 0.015,
        f'{text_queries[label]}: {score:1.2f}',
        ha='left',
        va='top',
        color='red',
        bbox={
            'facecolor': 'white',
            'edgecolor': 'red',
            'boxstyle': 'square,pad=.3'
        })
base_name, extension = os.path.splitext(filename)
save_path = "./realsense_bowl.png"
# save_path = "./headcamera_bowl.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
end_time = time.time()
print(f"Processed {filename} in {end_time - start_time:.2f} seconds")
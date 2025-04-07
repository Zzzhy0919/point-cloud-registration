# Point-Cloud-Registration

## OWL-ViT environment configuration

You can configure the OWL-ViT environment using the following command. (refer to [owl_vit](https://github.com/google-research/scenic/blob/main/README.md#philosophy) and [minimal colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb#scrollTo=kSDsqV0UxbtL))
```bash
conda create --name owl python=3.11
conda activate owl 
git clone https://github.com/google-research/scenic.git
cd ~/scenic
python -m pip install -vq .
python -m pip install -r scenic/projects/owl_vit/requirements.txt

# For GPU support:
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Also install big_vision, which is needed for the mask head:
cd ~/scenic/scenic/projects/owl_vit
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
python -m pip install -r /big_vision/big_vision/requirements.txt
```

Then download the checkpoints([OWLv2 CLIP B/16 ST/FT ens](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b)) and owl_vit_single.py to ` ~/scenic/scenic/projects/owl_vit`.
Remember to modify "checkpoint_path" in `~/anaconda3/envs/owl/lib/python3.11/site-packages/scenic/projects/owl_vit/configs/owl_v2_clip_b16.py`. Replace `checkpoint_path = CHECKPOINTS.get(init_mode, None)` with your own path to save the time of downloading checkpoints from website.

## SAM environment configuration

You can configure the SAM environment using the following command. (refer to [sam](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints))
```bash
conda create --name sam python=3.9
conda activate sam
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx open3d
```
Install PyTorch and TorchVision dependencies following [this](https://pytorch.org/get-started/locally/).

Then download the checkpoints([sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)) and sam_prompt.py and full_point.py to `~/sam`.

## Point Cloud Registration Implementation
```bash
#Modify the path of the point cloud file
python point_cloud.py
#Record the transformation matrix

conda activate owl
cd ~/scenic/scenic/projects/owl_vit
python owl_vit_single.py 
#Record the center point coordinates

conda activate sam
cd ~/sam
#Modify the center point coordinates and the camera intrinsic params
python sam_prompt.py

#Modify the transformation matrix
python full_point.py

# Point-Cloud-Registration
## OWL-ViT environment configuration

You can configure the OWL-ViT environment using the following command. (refer to [owl_vit](https://github.com/google-research/scenic/blob/main/README.md#philosophy) and [minimal colab](https://colab.research.google.com/github/google-research/scenic/blob/main/scenic/projects/owl_vit/notebooks/OWL_ViT_minimal_example.ipynb#scrollTo=kSDsqV0UxbtL))
```bash
git clone https://github.com/google-research/scenic.git
cd ~/scenic
python -m pip install -vq .
python -m pip install -r scenic/projects/owl_vit/requirements.txt

# For GPU support:
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Also install big_vision, which is needed for the mask head:
mkdir /big_vision
git clone https://github.com/google-research/big_vision.git /big_vision
python -m pip install -r /big_vision/big_vision/requirements.txt
```


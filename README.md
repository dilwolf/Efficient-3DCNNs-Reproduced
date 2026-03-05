# Efficient-3DCNNs-Reproduced: An Improved and Extended codebase of original [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs)

> ⚠️ **Work in Progress** — This repository is currently under active development. Training is in progress and results will be shared soon. Stay tuned!

This repository is an improved and extended version of [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs), originally proposed in the paper:

> **Resource Efficient 3D Convolutional Neural Networks**
> Okan Köpüklü, Neslihan Köse, Ahmet Gunduz, Gerhard Rigoll
> [arXiv:1904.02422](https://arxiv.org/abs/1904.02422)

The current focus is on **reproducing and improving ShuffleNetV2 on the Kinetics-600 dataset**. The codebase has been significantly restructured to be cleaner and more extensible compared to the original.

> This repo is also used as the 3D backbone in [YOWOv3-Improved](https://github.com/dilwolf/YOWOv3-Improved).

---

## What's New (so far)

- [x] **Clean codebase** - Significantly restructured and cleaner codebase
- [x] **Reproduction of ShuffleNetV2** - Focused reproduction of ShuffleNetV2 on Kinetics-600
- [x] **Codebase Upgraded** - Updated codebase with PyTorch 2.5+ and also other libraries
- [ ] Training results and pretrained checkpoints (coming soon)
- [ ] Extended model support (planned)

---

## Citation

If you use this repository, please consider citing the original paper:

```bibtex
@inproceedings{kopuklu2019resource,
  title={Resource efficient 3d convolutional neural networks},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Kose, Neslihan and Gunduz, Ahmet and Rigoll, Gerhard},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1910--1919},
  year={2019},
  organization={IEEE}
}
```

---

## Acknowledgements

- [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs) — the original codebase this repo extends
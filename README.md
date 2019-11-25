# DND Submit

Train your own model for denoising and submit the denoised results to [DND](https://noise.visinf.tu-darmstadt.de/).

> Typical result: sRGB track, PSNR > 39, SSIM > 0.95

## Data

Download the [dataset (22 GB)](https://pan.baidu.com/s/13_P1OnSqjVYO2bsei_ZbvQ) (xmfw) and extract the files to `./data/` folder.

## Model

This repo provides the codes of DnCNN, U-Net and ResNet, but you can use your own model by adding it to `./model/` folder.

## Train

Train your model:

```
python train.py --model YourModelName
```

Optional arguments:

```
  --model MODEL          model name
  --ps PS                patch size
  --lr LR                learning rate
  --epochs EPOCHS        sum of epochs
  --freq FREQ            learning rate update frequency
  --save_freq SAVE_FREQ  save result frequency
  --syn                  use synthetic noisy images
```

**In order to reduce the time to read the images, it will save all the images in memory which requires large memory.**

## Submit

Test the trained model on DND and get the denoised results:

```
python submit.py --model YourModelName
```

The results are in `./result/test/bundled/`.

## Reference

1. Guo, Shi, et al. "Toward convolutional blind denoising of real photographs." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
2. Anaya, Josue, and Adrian Barbu. "RENOIR–A dataset for real low-light image noise reduction." Journal of Visual Communication and Image Representation 51 (2018): 144-154.
3. Abdelhamed, Abdelrahman, Stephen Lin, and Michael S. Brown. "A high-quality denoising dataset for smartphone cameras." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
4. Bychkovsky, Vladimir, et al. "Learning photographic global tonal adjustment with a database of input/output image pairs." CVPR 2011. IEEE, 2011.
5. Plotz, Tobias, and Stefan Roth. "Benchmarking denoising algorithms with real photographs." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

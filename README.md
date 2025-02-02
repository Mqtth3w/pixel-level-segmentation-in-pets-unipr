# pixel-level-segmentation-in-pets-unipr

### Project objective:  
-	Pixel level segmentation in PETS dataset 
### Dataset:  
-	Pets dataset provides the segmentation mask of a large variety of pets https://www.robots.ox.ac.uk/~vgg/data/pets/
### Network model:  
-	A segmentation model should be used for this task. UNet is a good choice, but you can experiments with different architectures.
### Detailed information:  
-	Starting from a rgb image of a pet the network should output a segmentation mask of the pet. Only a single pet is present in each image.
-	Intersection over Union, L1 distance are good metrics to evaluate the results.
### Additional notes: 
-	Experiment also with in-the-wild samples (meaning images not extracted from the pets dataset: e.g. found online or taken by a smartphone). Does the network perform well over these images?
-	That happens if more than on animal is in one image? How does the model perform?

---

## Nets I decided to train/finetune and test
- UNet (required)
- DeepLabV3_ResNet101

## Best performances achieved

| | Model | Description | Resolution | IoU | L1 distance | Download | |
|-|-------------------|-------------------|----------|----------|----------|----------|-|
| | UNet | Trained from scratch | 256x256 | 0.8473 | 0.0574 | [weights](./unet/checkpoints/OxfordIIITPet_UNet_wd_1e_4.pth) | |
| | DeepLabV3_ResNet101 | Pretrained on COCO. Backbone frezed except layer_4 | 512x512 | 0.8935 | 0.0387 | [weights](./deeplabv3_resnet101/checkpoints/OxfordIIITPet_DeepLabV3ResNet_lr5e_4.pth) | |
| | | | | | | | |

> [!NOTE]
> These models were trained with A100 40GB, A100 80GB GPUs and data shuffle, so if you train them again you may obtain different results also depending on your hardware.
> Of course there may be a better hyper-parameters configuration, this is not necessarily the best.

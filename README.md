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

## Nets I decided to train and test
- UNet (required)
- DeepLabV3
- FCN

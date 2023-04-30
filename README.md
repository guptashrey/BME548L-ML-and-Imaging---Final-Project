# Multiple Myeloma Cell segmentation
> #### _Archit, Shen, Shrey | Spring '23 | Duke BME548L ML & Imaging Final Project_
&nbsp;

## Project Overview​
Multiple myeloma (MM) is a blood cancer caused by the abnormal expansion of plasma cells in the bone marrow​. Diagnosis of multiple myeloma can be done through various methods, including CBC and counting myeloma plasma cells in aspirate slide images​. Deep learning models and computer vision techniques are being used to detect MM​.  

&nbsp;  
## Our Goal​
- Physical simulations can be applied to raw BMP images to improve the accuracy of myeloma cell segmentation​
- The goal of our work is to make the process of myeloma cell segmentation more robust​  

&nbsp;  
## Data Sources
- SegPC-2021 dataset is being used for this project​
- Collected from subjects suffering from Multiple Myeloma (MM) who were being diagnosed or treated at AIIMS, New Delhi, India.​
- Microscopic images were captured from bone marrow aspirate slides using two cameras:​
    - Olympus camera with image size 2040x1536 pixels​
    - Nikon camera with image size 1920x2560 pixels​
- The dataset contains 775 images that were stain color normalized and divided into three sets:​
    - train (298 images)​
    - validation (200 images)​
    - test (277 images)​
- The ground truth (GT) values are available for the training and validation sets, but not for the test set​  

&nbsp;  
## Data Processing  
- Rescale the images to the dimensions of 1080x1440 pixels since the desired outputs are expected in this size​
- Convert the given ground truth image masks into COCO annotation format​
- Apply image transformations to simulate physical changes in the images​  

&nbsp;  
## Image Transformations
- Illumination​: Create a custom trainable layer that has optimizable weights to simulate the effect of 12 light waves hitting the sample at various angles​
- Contrast normalization​: Enhance the visibility of myeloma cells that have varying contrast levels compared to other cells and tissues
- Morphological operations​: Smooth the edges of the cells using Erosion and Dilation operations​
- Gradient filters​: Apply Sobel Gradient filters to enhance the edges of the cells​
- Manipulating different color channels (RGB)​: Enhance or suppress the effect of either the red, green, or blue channel​
- Blur filters​: Used Gaussian & Median Blur filters to smoothen the image and reduce noise​  

&nbsp;
## Performance Comparison
|IMAGE TRANSFORM|NUCLEUS SEG​ VAL AP ​|CYTOPLASM SEG VAL AP​|TEST​ MEAN IOU​| MODEL WEIGHTS |
|--|:--:|:--:|:--:|:--:|
|None​|80.3993​|44.3472​|0.8557​|[model](https://duke.box.com/s/qowxn0fubdeidfaiyz6vm7myejn6hfir)|
|Illumination Simulation​|77.6479​|41.4616​|0.8469​|[model](https://duke.box.com/s/7tdf2eu7341iqsuxhs7x3v3cylwhldki)|
|Contrast Normalization​|78.2172​|44.0549​|0.8494​|[model](https://duke.box.com/s/h3uznikeylmy68dkq21624qaexs5qz04)|
|Erosion Operation​|78.7668​|43.8360​|0.8465​|[model](https://duke.box.com/s/u5yfvuyimrgyx4ni5n7bpla6v4sg65gv)|
|Dilation Operation​|82.0081​|43.7919​|0.8586​|[model](https://duke.box.com/s/l4x9hnzo37nybgy4f1ogq32hirpbkzuv)|
|Sobel Gradient Filter​|74.1309​|38.5333​|0.8322​|[model](https://duke.box.com/s/tg9u0j4rk3tflxzs3d3je06of1dkinya)|
|Red Channel Enhanced​|79.8766​|42.9748​|0.8588​|[model](https://duke.box.com/s/mn6kej7p7xzdrespc8rqk9ziwck0ormi)|
|Green Channel Enhanced​|79.9117​|42.3286​|0.8665​|[model](https://duke.box.com/s/j991o4ak7jkxvab7rnfi778b92z5tqnq)|
|Blue Channel Enhanced​|79.0457​|43.5166​|0.8553​|[model](https://duke.box.com/s/a2uclp7nbcfb4u68nnnxui6gc2rvt44k)|
|Gaussian Blur Filter​|81.7441​|43.9569​|0.8477​|[model](https://duke.box.com/s/08alil2htadclz4dx2hxkh0zyi0osv34)|
|Median Blur Filter​|81.5294​|46.0385​|0.8696​|​[model](https://duke.box.com/s/0lj0oo8eqf86nrea10ps70jvjx9k47au)|


&nbsp; 
## References 
- Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42 (https://dx.doi.org/10.21227/7np1-2q42).​

- The Definitive Guide to Cell Segmentation Analysis, https://blog.biodock.ai/definitive-guide-to-cell-segmentation-analysis/ (https://blog.biodock.ai/definitive-guide-to-cell-segmentation-analysis/)​

- Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, March 20, 2017, "Mask R-CNN", arXiv, doi: https://doi.org/10.48550/arxiv.1703.06870 (https://doi.org/10.48550/arxiv.1703.06870)​

- Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images (SegPC-2021), accessed March 10, 2023, https://segpc-2021.grand-challenge.org/ (https://segpc-2021.grand-challenge.org/)​

- Digging into Detectron 2 — part 1, accessed April 15, 2023, https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd (https://medium.com/@hirotoschwert/digging-into-detectron-2-47b2e794fabd)

# Instance Segmentation Using ViT-based Mask R-CNN

 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/Instance_Segmentation_Using_ViT_based_Mask_RCNN.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>



<div align="center">
    <img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-3.png" alt="qualitative-3" >
    </img>
    <br />
</div>


Instance segmentation aims at dichotomizing a pixel acting as a sub-object of a unique entity in the scene. One of approaches, by combining object detection and semantic segmentation, is Mask R-CNN. Furthermore, we can also incorporate ViT as the backbone of Mask R-CNN. In this project, the ViT-based Mask R-CNN model is evaluated on the dataset from the Penn-Fudan Database for Pedestrian Detection and Segmentation. With a ratio of 80:10:10, the train, validation, and test sets is distributed.


## Experiment

Leap into this [link](https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/Instance_Segmentation_Using_ViT_based_Mask_RCNN.ipynb) that harbors a Jupyter Notebook of the entire experiment.


## Result

## Quantitative Result

The following table delivers the performance results of ViT-based Mask R-CNN, quantitatively.

Test Metric                    | Score
------------------------------ | -------------
mAP<sup>box</sup>@0.5:0.95     | 96.85%
mAP<sup>mask</sup>@0.5:0.95    | 79.58%


## Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss curves of ViT-based Mask R-CNN on the Penn-Fudan Database for Pedestrian Detection and Segmentation train and validation sets. </p>


## Qualitative Result

Below, the qualitative results are presented.

<p align="center"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-1.png" alt="qualitative-1"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-2.png" alt="qualitative-2"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-3.png" alt="qualitative-3"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-4.png" alt="qualitative-4"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-5.png" alt="qualitative-5"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-6.png" alt="qualitative-6"><img src="https://github.com/reshalfahsi/instance-segmentation-vit-maskrcnn/blob/master/assets/qualitative-7.png" alt="qualitative-7"><br /> Few samples of qualitative results from the ViT-based Mask R-CNN model.</p>


## Credit

- [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
- [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- [Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/pdf/2111.11429.pdf)
- [TorchVision's Mask R-CNN](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py)
- [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)

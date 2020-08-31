# ObjectDetetion
The folder is the object detection using Faster RCNN with Keras and Tensorflow backend.
The orginal code came from https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras
Based on the existed code, I updated some of them. 
The idea of the Faster RCNN is: 
(1) CNN part to get feature map. (There are some pre-trained models such as VGG)
(2) RPN part, which do classificaiton and regression.
(3) ROIpooling part, which resize different anchors to the same size
(4) Classification part. 
Compared with other existed Faster RCNN with Keras, this one just has one training py and each section is easy to understand.
You have two options to run the script. If you choose to run it at your local PC, you need to install Tensorflow-gpu to allow the computer using gpu to speed up. Or you can also use the scirpt on google Colab, which you don't need to install tensorflow-gpu.

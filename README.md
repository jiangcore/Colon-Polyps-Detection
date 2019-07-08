# Deep Learning project for Colon Polyps Detection
# This project uses GPU to train model of SSD-300 and Mask R-CNN
# Result refers to Evaluation for Mask RCNN.ipynb & Evaluation_SSD.ipynb
 

1. Setup the environment
    GCC version: (Ubuntu 7.3.0-27ubuntu1~18.04) 7.3.0
    CMake version: version 3.10.2

    Python version: 3.5
    Is CUDA available: Yes
    CUDA runtime version: Could not collect
    GPU models and configuration:
    GPU 0: GeForce GTX TITAN X
    GPU 1: Tesla K40c
    GPU 2: Tesla K40c

    Nvidia driver version: 396.37
    cuDNN version: /usr/local/cuda-9.2/lib64/libcudnn.so.7.1.4

    Anaconda 4.16.4

 1) install required libaries and applications:
    conda install -c anaconda cython
    conda install numpy
    pip install mmcv
    conda install matplotlib
    conda install scikit-image
    conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -y

2. install code
    git clone https://github.com/open-mmlab/mmdetection.git
     ./compile.sh
     python setup.py install
     cd mmdetection

     unzip DeepLearningProject_source.zip .
     mv -f coco.py mmdet/datasets/
     mv -f convertToCOCOfromSractch_new.py data/Polyp_Aug/
     mv -f mymask_rcnn_r101_fpn_1x.py  configs/
     mv -f myssd300_coco_3k.py  configs/


3. Prepare Data
   #the source code zip file has included all the pictures. So this step can be skipped.
   #cd data/Polyp_Aug/
   #python convertToCOCOfromSractch_new.py

4. Training
   #Training for SSD 300
   CUDA_VISIBLE_DEVICES=1,2 python ./tools/train.py   configs/myssd300_coco_3k.py --work_dir ./work_dir/ssd300_coco_3k --validate --gpus 2

   #Training for Mask R-CNN
   CUDA_VISIBLE_DEVICES=1,2 python ./tools/train.py   configs/mymask_rcnn_r101_fpn_1x.py --work_dir ./work_dir/mymask_rcnn_r101_fpn_3k --validate --gpus 2


5. Evaluation
   #Evaluation for SSD 300
CUDA_VISIBLE_DEVICES=1 python ./tools/test.py  configs/myssd300_coco.py work_dir/ssd300_coco_3k/latest.pth --out work_dir/ssd300_coco/ssd300_new.pkl --eval segm --gpus 1


   #Evaluation for Mask R-CNN
CUDA_VISIBLE_DEVICES=1 python ./tools/test.py  configs/mymask_rcnn_r101_fpn_1x.py work_dir/mymask_rcnn_r101_fpn_3k/latest.pth --out work_dir/mymask_rcnn_r101_fpn_1/maskrcnn.pkl --eval segm --gpus 1

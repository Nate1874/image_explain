# Explaining Image Classifier

The code for our TPAMI 2020 paper:  Interpreting Image Classifiers by Generating Discrete Masks [[paper link]](https://ieeexplore.ieee.org/abstract/document/9214476/?casa_token=yjUDEigqxdIAAAAA:V15jN6r0QM-1WPYR8rzrH1UP3NkhtHO84IgUDFX2XOoNKazVTpUB-6s_M-xSr8IUFVqieYWgQA)

## How to use our code

First, download the ImageNet dataset and set corresponding paths. 

In configure.py, the configurations can be modified. 

The build_imagenet_data.py and data_input.py files provide interface for data prepropossing and data loading. 

The network.py defines the architectures of our model, including the G and D. 

The model.py file include model training and testing. 

In visual folder, we provide code to perform several comparing methods, such as gradients, CAM, guidedBP and IG. 

Reference
---------

    @article{yuan2020interpreting,
      title={Interpreting image classifiers by generating discrete masks},
      author={Yuan, Hao and Cai, Lei and Hu, Xia and Wang, Jie and Ji, Shuiwang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2020},
      publisher={IEEE}
    }




This is the original implementation of the paper "Recovering the Imperfect: Cell Segmentation in the Presence of Dynamically Localized Proteins". It is based upon the Mask R-CNN implementation by Waleed Abdulla (https://github.com/matterport/Mask_RCNN).

# tensorflow

1. Sample pretrained network can be found under:
    /pretrained_net
2. Sample data to train the network is under:
    /sample_data
3. Train Mask R-CNN with hypothesis and aleatoric uncertainty loss:
    maskrcnn_scripts/run_hyp.sh
4. Freeze the Mask R-CNN with hypotheses and train the merging network on top:
    maskrcnn_scripts/run_hyp_merge.sh
5. Each network can be tested with the testing scripts:
    maskrcnn_scripts/test_hyp.sh
    maskrcnn_scripts/test_hyp_merge.sh
6. After the network training is successful, run the propagation algorithm via script:
    propagation_scripts/main_.sh
7. U-Net smooth deformations are not included as the original repository we used has copyleft protection. Any deformation repo in python can be easily integrated into the code.
8. Optical flow implementation is not going to be made public. Using the existing U-Net deformations and the FlowNet implementations, users should obtain optical flow to feed into the propagation algorithm. Alternatively, shift and scale as mentioned in the original paper can be used.

# license

This code is released under MIT license.

# citation

If you use our repository or find it useful in your research, please cite the following paper: 

@InProceedings{10.1007/978-3-030-61166-8_9,
author="{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n
and Marrakchi, Yassine
and Boasiako Antwi, Enoch
and Di Ventura, Barbara
and Brox, Thomas",
title="Recovering the Imperfect: Cell Segmentation in the Presence of Dynamically Localized Proteins",
booktitle="Interpretable and Annotation-Efficient Learning for Medical Image Computing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="85--93",
isbn="978-3-030-61166-8"
}


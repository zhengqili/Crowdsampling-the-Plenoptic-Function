# Crowdsampling the Plenoptic Function 

This repository contains a PyTorch implementation of the paper:

**Crowdsampling The Plenoptic Function**, ECCV 2020. 

[[Project Website]](https://research.cs.cornell.edu/crowdplenoptic/) [[Paper]](https://arxiv.org/pdf/2007.15194.pdf) [[Video]](https://www.youtube.com/watch?v=MAVFKWX8LYo)

[Zhengqi Li](https://www.cs.cornell.edu/~zl548/),
[Wenqi Xian](https://www.cs.cornell.edu/~wenqixian/),
[Abe Davis](http://www.abedavis.com/),
[Noah Snavely](https://www.cs.cornell.edu/~snavely/)


## Dataset
Download and unzip data from the links below: 

* [[Trevi Fountain]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0036.zip)
* [[Piazza Navona]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0057.zip)
* [[Top of the Rock]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0011.zip)
* [[Pantheon]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0023.zip)
* [[Sacre Coeur]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0013.zip)
* [[Lincoln Memorial]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0021.zip)
* [[Eiffel Tower]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0000.zip)
* [[Mount Rushmore]](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/1589.zip)

Read more about the dataset in [Readme file](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/README.txt).

## Dependency
The code is tested with Pytorch >= 1.2, the depdenency library includes
* matplotlib
* opencv
* scikit-image
* scipy
* json


## Pretrained Model
Download and unzip pretrained models from [link](https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/pretrain_models.zip).

To use the pretrained model, put the folders under the project root directory.

### Test the pretrained model:
To run the evaluation, change variable "root" and "data_dir" in script evaluation.py to code directory and data directory respectively. The released code is not highly optimized, so you have to use 4 GPUs with > 11GB memory to run the evaluation. 

| Dataset| Name  | Max depth  | FOV | 
|--------|----------|-------|--------|
| Trevi Fountain | trevi  | 4 |  70 |
| The Pantheon | pantheon | 25 |  65 |
| Top of the Rock | rock  | 75 |  70 |
| Sacre Coeur | coeur | 20 |  65 |
| Piazza Navona | navona  | 25 |  70 |

Follow the commands below:
```bash
   # Usage
   # python evaluation.py --dataset <name> --max_depth <max depth> --ref_fov <fov> --warp_src_img 1
   
   python evaluation.py --dataset trevi --max_depth 4 --ref_fov 70 --warp_src_img 1
```

### Demo of novel view synthesis:
```bash
   # Usage
   # python wander.py --dataset <name> --max_depth <max depth> --ref_fov <fov> --warp_src_img 1 --where_add adain --img_a_name xxx --img_b_name xxx --img_c_name xxx
 
   python wander.py --dataset trevi --max_depth 4 --ref_fov 70 --warp_src_img 1  --where_add adain --img_a_name 5094768508_fa56e355bd.jpg  -
-img_b_name 34558526690_e5ba5b3b9d.jpg --img_c_name 34558526690_e5ba5b3b9d.jpg
```
where 
* img_a_name: image associated with rendering target viewpoint, 
* set img_b_name=img_c_name: image whose apperance we would like to condition on. The results will be saved in folder demo_wander_trevi.

By running the example command, you should get the following result:

![Alt Text](https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/master/demo/ours_34558526690_e5ba5b3b9d.jpg.gif)


### Demo of apperance inteporlation:
```bash
   # Usage
   # python interpolate_appearance.py --dataset <name> --max_depth <max depth> --ref_fov <fov> --warp_src_img 1 --where_add adain --img_a_name xxx --img_b_name xxx --img_c_name xxx
 
   python interpolate_appearance.py --dataset trevi --max_depth 4 --ref_fov 70 --warp_src_img 1  --where_add adain --img_a_name 157303382_3ca2b644c9.jpg  --img_b_name 255196242_3f46e98a0f_o.jpg --img_c_name 157303382_3ca2b644c9.jpg
```
where
* img_a_name: image of starting apperance
* img_b_name: image of end apperance
* img_c_name: image associated with rendering target viewpoint

By running the example command, you can get the following result:
![Alt Text](https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/master/demo/ezgif-6-08f9beb9ee83.gif)


### TO DO: 
* Releasing two-stage training code

## Cite
Please cite our work if you find it useful:
```bash
@inproceedings{li2020crowdsampling,
Author = {Zhengqi Li and Wenqi Xian and Abe Davis and Noah Snavely},
Title = {Crowdsampling the Plenoptic Function},
Year = {2020},
booktitle = {Proc. European Conference on Computer Vision (ECCV)},
}
```

## License
This repository is released under the [MIT license](https://opensource.org/licenses/MIT).



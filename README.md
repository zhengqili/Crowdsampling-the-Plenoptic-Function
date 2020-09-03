# Crowdsampling-the-Plenoptic-Function 

* This is the evaluation code for reproducing the results in the paper Crowdsampling The Plenoptic Function, Li etal. ECCV 2020 (Oral)" .

* Website: https://research.cs.cornell.edu/crowdplenoptic/

* The code skeleton is based on "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix". If you use our code for academic purposes, please consider citing:

    @inproceedings{li2018cgintrinsics,
	  	title={Crowdsampling the Plenoptic Function},
     		author={Li, Zhengqi and Xian, Wenqi and Davis, Abe and Snavely, Noah},
	  	booktitle={European Conference on Computer Vision (ECCV)},
	  	year={2020}
	}
 
* Download and unzip data from 

*  Readme file: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/README.txt
*  Effiel Tower: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0000.zip
*  Top of the Rock: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0011.zip
*  Sacre Coeur: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0013.zip
* Lincoln Memorial: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0021.zip
*  Pantheon: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0023.zip   
*  Trevi Fountain: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0036.zip
*  Piazza Navona: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/0057.zip
*  Mount Rushmore: https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/1589.zip

* Download and unzip pretrained models from https://research.cs.cornell.edu/megadepth/dataset/CrowdSampling/pretrain_models.zip, and put the folders in the same directory as evaluation.py

* To run the evaluation, change variable "root" and "data_dir" in script evaluation.py to code directory and data directory respectively. The released code is not highly optimized, so you have to use 4 GPUs with > 11GB memory to run the evaluation. 

```bash
    Trevi Fountain: python evaluation.py --dataset trevi --max_depth 4 --ref_fov 70 --warp_src_img 1
    The Pantheon: python evaluation.py --dataset pantheon --max_depth 25 --ref_fov 65 --warp_src_img 1
    Top of the Rock: python evaluation.py --dataset rock --max_depth 75 --ref_fov 70 --warp_src_img 1
    Sacre Coeur: python evaluation.py --dataset coeur --max_depth 20 --ref_fov 65 --warp_src_img 1
    Piazza Navona: python evaluation.py --dataset navona --max_depth 25 --ref_fov 70 --warp_src_img 1
```

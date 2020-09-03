# Crowdsampling-the-Plenoptic-FunctionWe release 

evaluation code for reproducing the results since releasing robust training code is not easy given the complexity of the code.
To run the evaluation, 
change variable "root" and "data_dir" in evaluation.py to code directory and data directory respectively.
The code is not highly optimized, so you have to use 4 GPUs with > 11GB memory to run the evaluation 

Trevi Fountain: python evaluation.py --dataset trevi --max_depth 4 --ref_fov 70 --warp_src_img 1
The Pantheon: python evaluation.py --dataset pantheon --max_depth 25 --ref_fov 65 --warp_src_img 1
Top of the Rock: python evaluation.py --dataset rock --max_depth 75 --ref_fov 70 --warp_src_img 1
Sacre Coeur: python evaluation.py --dataset coeur --max_depth 20 --ref_fov 65 --warp_src_img 1
Piazza Navona: python evaluation.py --dataset navona --max_depth 25 --ref_fov 70 --warp_src_img 1

dig +short myip.opendns.com @resolver1.opendns.com > ip.txt
singularity exec --nv -B /fast_scratch -B /data /fast_scratch/triumfmlutils/containers/rl_gym_v0.3.sif /bin/bash

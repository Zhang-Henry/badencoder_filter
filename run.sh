# CUDA_VISIBLE_DEVICES=2 nohup python stealty.py > log/stealth.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python new_stealthy.py > log/new_stealthy.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python new_stealthy.py > log/new_stealthy_CTRL.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python new_stealthy.py > log/new_stealthy_badencoder.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python new_stealthy.py > log/new_stealthy_kelvin.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python new_stealthy.py > log/new_stealthy_xpro2.log 2>&1 &
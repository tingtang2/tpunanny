#!/bin/bash

cd loss-spikes-project/ 
source env-loss-spikes/bin/activate 
cd picodo/ 

python3 main.py \
  +model=gpt3xl +dataset=fw_gpt2 \
  run_name=exp_gpt3xl_seed0 \
  checkpoint.turn_on=true \
  +checkpoint.gcp_bucket=gs://demand-v4-checkpoint-storage/picodo_ckpts/exp_gpt3xl_seed0 \
  checkpoint.start_step=null \
  checkpoint.checkpoint_steps=null \
  checkpoint.checkpoint_every_steps=500 \
  log_logit_grad_stats=True\
  opt.batch_size=64 \
  seed=0 \
  wandb_mode=online \
  --config-name=wortsman_default

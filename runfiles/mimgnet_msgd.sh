# meta-train
python main.py \
  --savedir "./results/mimgnet/msgd" \
  --id_dataset 'mimgnet' \
  --ood_dataset 'cub' \
  --mode 'meta_train' \
  --gpu_id 2 \
  --metabatch 1 \
  --n_steps 5 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_train_iters 80000 \
  --meta_lr 1e-4 \
  --alpha_on

# meta-test
python main.py \
  --savedir "./results/mimgnet/msgd" \
  --id_dataset 'mimgnet' \
  --ood_dataset 'cub' \
  --mode 'meta_test' \
  --gpu_id 2 \
  --metabatch 1 \
  --n_steps 10 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_test_episode 1000 \
  --alpha_on

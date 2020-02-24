# meta-train
python main.py \
  --savedir "./results/cifar/taml" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_train' \
  --gpu_id 2 \
  --metabatch 4 \
  --n_steps 5 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_train_iters 50000 \
  --meta_lr 1e-3 \
  --alpha_on \
  --omega_on \
  --gamma_on \
  --z_on

# meta-test
python main.py \
  --savedir "./results/cifar/taml" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_test' \
  --gpu_id 2 \
  --metabatch 4 \
  --n_steps 10 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_test_episode 1000 \
  --alpha_on \
  --omega_on \
  --gamma_on \
  --z_on \
  --n_mc_samples 10

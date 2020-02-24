# meta-train
python main.py \
  --savedir "./results/cifar/msgd" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_train' \
  --gpu_id 1 \
  --metabatch 4 \
  --n_steps 5 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_train_iters 50000 \
  --meta_lr 1e-3 \
  --alpha_on

# meta-test
python main.py \
  --savedir "./results/cifar/msgd" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_test' \
  --gpu_id 1 \
  --metabatch 4 \
  --n_steps 10 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_test_episode 1000 \
  --alpha_on

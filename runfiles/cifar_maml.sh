# meta-train
python main.py \
  --savedir "./results/cifar/maml" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_train' \
  --gpu_id 0 \
  --metabatch 4 \
  --n_steps 5 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_train_iters 50000 \
  --inner_lr 0.5 \
  --meta_lr 1e-3

# meta-test
python main.py \
  --savedir "./results/cifar/maml" \
  --id_dataset 'cifar' \
  --ood_dataset 'svhn' \
  --mode 'meta_test' \
  --gpu_id 0 \
  --metabatch 4 \
  --n_steps 10 \
  --way 5 \
  --max_shot 50 \
  --query 15 \
  --n_test_episode 1000 \
  --inner_lr 0.5

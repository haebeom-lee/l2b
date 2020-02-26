# meta-train
python main.py \
 --savedir "./results/multi/msgd" \
 --id_dataset 'aircraft, quickdraw, vgg_flower' \
 --ood_dataset 'traffic, fashion-mnist' \
 --mode 'meta_train' \
 --gpu_id 0 \
 --metabatch 3 \
 --n_steps 5 \
 --way 10 \
 --max_shot 50 \
 --query 15 \
 --n_train_iters 60000 \
 --meta_lr 1e-3 \
 --alpha_on 

# meta-test
python main.py \
 --savedir "./results/multi/msgd" \
 --id_dataset 'aircraft, quickdraw, vgg_flower' \
 --ood_dataset 'traffic, fashion-mnist' \
 --mode 'meta_test' \
 --gpu_id 0 \
 --metabatch 3 \
 --n_steps 10 \
 --way 10 \
 --max_shot 50 \
 --query 15 \
 --n_test_episode 1000 \
 --alpha_on 


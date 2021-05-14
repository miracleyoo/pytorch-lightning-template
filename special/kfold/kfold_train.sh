for i in {0..4}
do
   echo "Working on $i th fold."
   python main.py --gpus=1 --train_type=sr --data_dir=data/ref --batch_size=128 --model_name=simple_net --layer_num=5 --kfold=5 --fold_num=$i --log_dir=sr
done
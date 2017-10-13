#rm nohup.out
env=LunarLander-v2
workers=3
ps=2
for ((i=0; i<$ps; i++)); do
   python train.py ps $i --worker_num $workers  --env $env --ps_num $ps &
done

mode=train

for ((i=0; i<$workers; i++)); do
    python train.py worker $i --mode $mode  --env $env --worker_num $workers --ps_num $ps  &
done

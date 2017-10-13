#rm nohup.out
env=BipedalWalker-v2
workers=3
ps=1
mode=debug-light
python train.py ps 0 --worker_num $workers  --env $env --ps_num $ps &
python train.py worker 0 --animate --mode $mode  --env $env --worker_num $workers --ps_num $ps  &
python train.py worker 1 --worker_num $workers --mode $mode --env $env --ps_num $ps &
python train.py worker 2 --worker_num $workers --mode $mode --env $env --ps_num $ps &

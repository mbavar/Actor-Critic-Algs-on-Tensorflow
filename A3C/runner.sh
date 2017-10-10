#rm nohup.out
env=Pendulum-v0
workers=2
ps=1
python run.py ps 0 --worker_num $workers  --env $env --ps_num $ps &
python run.py worker 0 --animate  --env $env --worker_num $workers --ps_num $ps  &
python run.py worker 1 --worker_num $workers --env $env --ps_num $ps &
#python run.py worker 2 --worker_num $workers --env $env --ps_num $ps &

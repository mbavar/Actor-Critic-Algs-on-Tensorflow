rm nohup.out
python run.py ps 0 --worker_num 2 --env CartPole-v0 --ps_num 1 &
python run.py worker 0 --animate --env CartPole-v0 --worker_num 2 --ps_num 1 &
python run.py worker 1 --env CartPole-v0 --worker_num 2 --ps_num 1 & 



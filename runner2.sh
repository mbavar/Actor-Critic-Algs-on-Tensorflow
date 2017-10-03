rm nohup.out
python run.py ps 0 --worker_num 3  --ps_num 1 &
python run.py worker 0 --animate --worker_num 3 --ps_num 1 &
python run.py worker 1 --worker_num 3 --ps_num 1 & 
python run.py worker 2 --worker_num 3 --ps_num 1 &



rm nohup.out
python run.py ps 0 --worker_num 2 --ps_num 1 &
python run.py worker 0 --worker_num 2 --ps_num 1 &
python run.py worker 1 --worker_num 2 --ps_num 1 & 



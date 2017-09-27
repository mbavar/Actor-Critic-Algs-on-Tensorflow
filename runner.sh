rm nohup.out
nohup python run.py ps 0 &
nohup python run.py ps 1 & 
nohup python run.py worker 0 &
nohup python run.py worker 1 &
nohup python run.py worker 2 &
nohup python run.py worker 3 &



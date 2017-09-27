rm nohup.out
nohup python run.py ps 0 &
nohup python run.py worker 0 &
nohup python run.py worker 1 &



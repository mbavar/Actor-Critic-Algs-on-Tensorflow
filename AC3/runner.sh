rm nohup.out
python run.py ps 0 &
python run.py ps 1 & 
python run.py worker 0 &
python run.py worker 1 &
python run.py worker 2 &
python run.py worker 3 &



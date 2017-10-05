from process import process_fn
import util as U
import argparse


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("job", choices=["ps", "worker"])
    parser.add_argument("task", type=int )
    parser.add_argument("--outdir", default='log.txt')
    parser.add_argument("--animate", default=False, action='store_true')
    parser.add_argument("--env", default='Pendulum-v0')
    parser.add_argument("--seed", default=12321)
    parser.add_argument("--tboard", default=False)
    parser.add_argument("--worker_num",default=4, type=int)
    parser.add_argument("--ps_num", default=2, type=int)
    parser.add_argument("--initport", default=8597)
    parser.add_argument("--checkpoint", default='./tmp/checkpoints/')
    args = parser.parse_args()
    
    ANIMATE = args.animate and args.task == 0 and  args.job == 'worker'
    INITPORT = args.initport
    CLUSTER = dict()
    workers = []
    ps_ = []
    for i in range(args.ps_num):
        ps_.append('localhost:{}'.format(INITPORT+i))
    for i in range(args.worker_num):
        workers.append("localhost:{}".format(i+args.ps_num+INITPORT))
    CLUSTER['worker'] = workers
    CLUSTER['ps'] = ps_
    LOG_FILE = args.outdir.split('.')[0] + '_{}.'.format(args.task) + args.outdir.split('.')[1] if args.job == 'worker' else  'no logging for ps'
    RANDOM_SEED = args.seed + args.task

    logger = U.Logger(logfile=LOG_FILE) if args.job == 'worker' else None
    print("Starting {} {} with log at {}".format(args.job, args.task, LOG_FILE))
    process_fn(cluster=CLUSTER, task_id=args.task, job=args.job , logger=logger, 
                env_id=args.env, animate=ANIMATE, random_seed=RANDOM_SEED, save_path=args.checkpoint)



if __name__ == '__main__':
    main()


#all_vars = tf.trainable_variables()
#u = [v for v in all_vars if 'Critic' in v.name]

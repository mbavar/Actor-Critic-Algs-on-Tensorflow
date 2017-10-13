from process import process_fn
from process import test_process
import util as U
import argparse
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("job", choices=["ps", "worker"])
    parser.add_argument("task", type=int )
    parser.add_argument("--animate", default=False, action='store_true')
    parser.add_argument("--env", default='Pendulum-v0')
    parser.add_argument("--seed", default=12321)
    parser.add_argument("--tboard", default=False)
    parser.add_argument("--worker_num",default=4, type=int) #worker jobs
    parser.add_argument("--ps_num", default=2, type=int)  #ps jobs
    parser.add_argument("--initport", default=1738)   #starting ports for cluster
    parser.add_argument("--save_every", default=600)  #save frequency
    parser.add_argument("--outdir", default=os.path.join('tmp', 'logs'))  # file for the statistics of training
    parser.add_argument("--checkpoint_dir", default=os.path.join('tmp', 'checkpoints'))   #where to save checkpoint
    parser.add_argument("--frames", default=1)    #how many recent frames to send to model 
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--num_episodes", default=3, help="For Test mode. How many episodes to render.")
    parser.add_argument("--mode", choices=["train", "debug-light", "debug-full", "test"], default="train") #how verbose to print to stdout
    parser.add_argument("--desired_kl", default=0.002)   #An important param to tune. The learning rate is adjusted when KL dist falls 
                                                         #far above or below the desired_kl

    args = parser.parse_args()

    if args.mode == 'test':
        model_path = args.load_model or tf.train.latest_checkpoint(checkpoint_dir)
        if not model_path:
            raise ValueError('No model found. Use --load_model to specify the model.')
        print("\n************Test Mode**********\nUsing model path {}\n\n".format(model_path))   
        test_process(random_seed=args.seed, animate=args.animate, env_id=args.env, model_path=model_path, 
                     stack_frames=args.frames, num_episodes=args.num_episodes)
        return

    
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
    LOG_FILE = os.path.join(args.outdir, 'worker_{}.log'.format(args.task)) if args.job == 'worker' else  'N/A'
    RANDOM_SEED = args.seed + args.task
    checkpoint_basename = 'model' + '-'+ args.env.split('-')[0] 

    logger = U.Logger(logfile=LOG_FILE) if args.job == 'worker' else None
    print("Starting {} {} with log at {}".format(args.job, args.task, LOG_FILE))
    process_fn(cluster=CLUSTER, task_id=args.task, job=args.job , logger=logger, 
                env_id=args.env, animate=ANIMATE, random_seed=RANDOM_SEED, save_path=args.checkpoint_dir, stack_frames=args.frames,
                save_every=args.save_every, run_mode=args.mode, desired_kl=args.desired_kl, checkpoint_basename=checkpoint_basename)



if __name__ == '__main__':
    main()


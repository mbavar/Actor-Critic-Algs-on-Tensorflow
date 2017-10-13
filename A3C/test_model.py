from process import test_process
import util as U
import argparse
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("env", default='Pendulum-v0')
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--no_animation", default=False, action='store_true')
    parser.add_argument("--seed", default=12321, type=int)
    parser.add_argument("--checkpoint_dir", default=os.path.join('tmp', 'checkpoints'))   #where to save checkpoint
    parser.add_argument("--frames", default=1, type=int)    #how many recent frames to send to model 
    parser.add_argument("--num_episodes", default=3, help="For Test mode. How many episodes to render.", type=int)
    args = parser.parse_args()

    model_path = args.load_model or tf.train.latest_checkpoint(checkpoint_dir)
    if not model_path:
        raise ValueError('No model found. Use --load_model to specify the model.')
    print("\n************Test Mode**********\nUsing model path {}\n\n".format(model_path))   
    test_process(random_seed=args.seed, animate=not args.no_animation, env_id=args.env, model_path=model_path, 
                 stack_frames=args.frames, num_episodes=args.num_episodes)


if __name__ == '__main__':
    main()


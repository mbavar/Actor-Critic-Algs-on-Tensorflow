from run_AC import test_process
import argparse



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("env")
    parser.add_argument("model_path")
    parser.add_argument("--no_animation", default=False, action='store_true')
    parser.add_argument("--seed", default=12321, type=int)
    parser.add_argument("--frames", default=1, type=int)    #how many recent frames to send to model 
    parser.add_argument("--num_episodes", default=3, help="For Test mode. How many episodes to render.", type=int)
    args = parser.parse_args()

    print("\n************Test Mode**********\nUsing model path {}\n\n".format(args.model_path))   
    test_process(random_seed=args.seed, animate=not args.no_animation, env_id=args.env, model_path=args.model_path, 
                 stack_frames=args.frames, num_episodes=args.num_episodes)


if __name__ == '__main__':
    main()


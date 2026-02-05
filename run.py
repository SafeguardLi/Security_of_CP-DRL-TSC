import os, sys, time

from src.argparse import parse_cl_args
from src.distprocs import DistProcs
from src.helper_funcs import get_fp

def main():
    start_t = time.time()
    print('start running main...')
    args = parse_cl_args()
   
    distprocs = DistProcs(args, args.tsc, args.mode)
    distprocs.run()
    print(args)

    args_config_file = get_fp(args,'args_config.txt')
    with open(args_config_file,'w') as out_file:
        out_file.write(str(args))

    print('...finish running main')
    print('run time : '+str((time.time()-start_t)/60) + ' min(s)')

if __name__ == '__main__':
    main()

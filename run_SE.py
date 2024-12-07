import numpy as np
import pandas as pd

from state_evolution import StateEvolution

import argparse

parser = argparse.ArgumentParser(description='args for xor theory prediction')

parser.add_argument('--out_step_csv',default='test_output/xor_step_default.csv',help='output result csv for each step')
parser.add_argument('--out_final_csv',default='test_output/xor_final_default.csv',help='output result csv for final result')
parser.add_argument('--out_txt',default='test_output/xor_out.txt',help='output txt for test output')
parser.add_argument('--start_alpha', type=float, 
                    help='starting alpha')
parser.add_argument('--end_alpha', type=float,
                    help='ending alpha')
parser.add_argument('--alpha_step',type=float,default=0.5,help='step for each alpha')
parser.add_argument('--delta',help='delta for the variance of the cluster',type=float,default=0.1)
parser.add_argument('--lamb',help='lambba for the l2 regularization strength',type=float,default=0.1)
parser.add_argument('--max_step',help='max iteration step',type=int,default=500)
parser.add_argument('--tol',help='tolerance for convergence',type=float,default=0.001)
parser.add_argument('--damping',help='damping of iteration',type=float,default=0.9)
parser.add_argument('--integral_steps',help='number of MC for integral calculation',type=int,default=10000)
parser.add_argument('--seeds',help='number of iteration for each alpha',type=int,default=1)
parser.add_argument('--printout',help='wether or not print out at each step',type=bool,default=False)
parser.add_argument('--printout_csv',help='csv to printout proximals at each step',default='test_output/xor_printout.csv')
parser.add_argument('--error_evaluation',help='wether or not evaluate error at each step',default=False,type=bool)
parser.add_argument('--resume',help='whether or not resume the previous run', default=False,type=bool)
args = parser.parse_args()

K = 2
C = 4

if __name__ == "__main__":
    if args.alpha_step == 0:
        alphas = [args.start_alpha]
    else:
        if args.start_alpha <= args.end_alpha:
            alphas = np.arange(args.start_alpha,args.end_alpha,args.alpha_step)
        else:
            alphas = np.arange(args.end_alpha,args.start_alpha,args.alpha_step)
            alphas = np.flip(alphas)
    
    resume = args.resume
    printout = args.printout

    if resume and printout:
        df_printout = pd.read_csv(args.printout_csv)
        idx = df_printout.index[-1]
        
        V00, V01, V10, V11 = df_printout.loc[idx,'V00'], df_printout.loc[idx,'V01'], df_printout.loc[idx,'V10'], df_printout.loc[idx,'V11']
        Q00, Q01, Q10, Q11 = df_printout.loc[idx,'Q00'], df_printout.loc[idx,'Q01'], df_printout.loc[idx,'Q10'], df_printout.loc[idx,'Q11']
        M00, M01, M02, M03 = df_printout.loc[idx,'M00'], df_printout.loc[idx,'M01'], df_printout.loc[idx,'M02'], df_printout.loc[idx,'M03']
        M10, M11, M12, M13 = df_printout.loc[idx,'M10'], df_printout.loc[idx,'M11'], df_printout.loc[idx,'M12'], df_printout.loc[idx,'M13']
        b0, b1 = df_printout.loc[idx,'b0'], df_printout.loc[idx,'b1']

        print(M00, M01, M02, M03)
        initV = np.array([[V00,V01],[V10,V11]])
        initQ = np.array([[Q00,Q01],[Q10,Q11]])
        initM = np.array([[M00,M01,M02,M03],[M10,M11,M12,M13]])
        initb = np.array([b0,b1])
    else:
        initV = np.eye(K)
        initQ = np.eye(K)
        initM = np.array([[1,-1,1,-1],[1,-1,-1,1]])
        # initM = np.zeros([K,C])
        initb = np.zeros([K])
    
    print("initV:",initV)
    print("initQ:",initQ)
    print("initM:",initM)
    print("initb:",initb)

    SE = StateEvolution(delta=args.delta, lamb=args.lamb, tol=args.tol, integral_steps=args.integral_steps, max_steps=args.max_step,
                        initV=initV, initQ=initQ, initM=initM, initb=initb,
                        out_step_csv=args.out_step_csv, out_final_csv=args.out_final_csv, out_txt=args.out_txt, printout_csv=args.printout_csv,
                        damping=args.damping, printout=printout, error_evaluation=args.error_evaluation, resume=resume)
    
    SE.iterate_follow(alphas)
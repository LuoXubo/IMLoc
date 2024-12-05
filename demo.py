"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/12/02 20:14:06
"""

from imloc_eval import IMLoc
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    evaluator = IMLoc(args.config)
    evaluator.eval_imloc()
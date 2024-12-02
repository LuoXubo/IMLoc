"""
@Description :   
@Author      :   Xubo Luo 
@Time        :   2024/12/02 20:14:06
"""

from imloc_eval import IMLoc

evaluator = IMLoc('config.yaml')

evaluator.eval_imloc('results')

"""
@Description :   Image matching for localization
@Author      :   Xubo Luo 
@Time        :   2024/12/02 20:01:56
"""

import numpy as np
import cv2
import os
import yaml

import imloc
from imloc.utils.data_io import lprint
from imloc.utils.model_helper import parse_model_config

class IMLoc:
    def __init__(self, config_path = 'config.yaml'):
        self.database_root = None
        self.query_list = None
        self.query_root = None
        self.methods = []
        self.benchmark = 'imloc'
        self.root_dir = None
        
        self.load_config(config_path)
        self.load_dataset()
        
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.database_root = config['database_root']
        self.query_root = config['query_root']
        self.query_list = os.path.join(self.query_root, 'images.txt')
        self.methods = config['methods']
        self.benchmark = config['benchmark']
        self.root_dir = config['root_dir']
    
    def load_dataset(self):
        # Load database
        self.database = sorted(os.listdir(self.database_root))
        # Load query
        with open(self.query_list, 'r') as f:
            self.query = f.readlines()
            f.close()
            
        print(f'>>>>> Database: {self.database_root} ({len(self.database)} images)')
        print(f'>>>>> Query: {self.query_list} ({len(self.query)} images)')
        
    def warp_coordinate(self, H, start_h, start_w):
        center = np.array([512, 512, 1], dtype=np.float32).reshape(3, 1)
        center_projection = np.dot(H, center)
        center_projection = center_projection / center_projection[2]
        center_projection = center_projection[:2].reshape(2)
        center_projection = center_projection + np.array([start_h, start_w])
        return center_projection
    
    def eval_imloc(self, save_results):
        for method in self.methods:
            result_str = ""
            args = parse_model_config(method, self.benchmark, self.root_dir)
            class_name = args['class']
            
            # One log file per method
            if save_results:
                if not os.path.exists(save_results):
                    os.makedirs(save_results)
                log_file = os.path.join(save_results, f"{class_name}.txt")
                log = open(log_file, "a")
                lprint_ = lambda ms: lprint(ms, log)

            # Load model
            model = imloc.__dict__[class_name](args)
            print(f'>>>>> {class_name} loaded!')
            
            current_index = 0
            for query_info in self.query:
                timestamp, query_path = query_info.split('\n')[0].split(' ')
                
                max_matches = 0
                best_H = None

                for database_index in range(max(0, current_index-2), min(len(self.database), current_index+3)):
                    database_path = self.database[database_index]
                    
                    # matches = np.concatenate([p1s, p2s], axis=1)
                    matches, kpts1, kpts2, scale = model.match_pairs(os.path.join(self.database_root, database_path), 
                                                                     os.path.join(self.query_root, query_path))
                    
                    if len(matches) < 4 or len(matches) < max_matches:
                        continue
                    
                    H_pred, inliers = cv2.findHomography(matches[:, :2], matches[:, 2:4], cv2.RANSAC, 2.0)
                    max_matches = len(matches)
                    best_H = H_pred
                    current_index = database_index
                
                start_h, start_w = self.database[current_index].split('.')[0].split('_')[1:]
                x, y = self.warp_coordinate(best_H, float(start_w), float(start_h))
                result_str += f'{timestamp} {x} {y} 0 0.707107 0.0 -0.0 -0.707107\n'
                print(f'{timestamp} in map:{current_index} with {max_matches} matches, X:{x}, Y:{y}')
            
            if save_results:
                lprint_(result_str)
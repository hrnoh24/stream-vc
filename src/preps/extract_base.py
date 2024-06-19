import os, glob
import torch
from multiprocessing import Process
from functools import partial

from abc import abstractmethod

class BaseExtractor:
    def __init__(self, 
                 root_dir, 
                 data_ext="wav",
                 num_workers=8, 
                 device="cuda"):
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.device = device
        
        self.filelist = glob.glob(os.path.join(root_dir, f"**/*.{data_ext}"), recursive=True)

    @abstractmethod
    def _run(self, rank, filelist):
        raise NotImplementedError

    def run(self):
        if self.num_workers > 1:
            num_gpus = torch.cuda.device_count()
            processes = []
            for i in range(self.num_workers):
                data = self.filelist[i::self.num_workers]
                p = Process(target=partial(self._run, (i%num_gpus), data))
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
        else:
            self._run(0, self.filelist)
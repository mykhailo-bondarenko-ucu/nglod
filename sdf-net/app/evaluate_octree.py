import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import glob
import logging as log
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from lib.datasets import *
from lib.models import *
from lib.tracer import *
from lib.validator import *

from sklearn.metrics import f1_score

import numpy as np

from lib.options import parse_options

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)


from lib.torchgp import load_obj, point_sample, sample_surface, compute_sdf, normalize, sample_uniform, area_weighted_distribution
from lib.utils import setparam

def sample_near_surface_2(
    V : torch.Tensor,
    F : torch.Tensor, 
    num_samples: int,
    noise_std=1e-2,
):
    """Sample points near the mesh surface.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        num_samples (int): number of surface samples
        distrib: distribution to use. By default, area-weighted distribution is used
    """
    distrib = area_weighted_distribution(V, F)
    samples = sample_surface(V, F, num_samples, distrib)[0]
    variance = noise_std ** 2
    samples += torch.randn_like(samples) * variance
    return samples


class TestingDataset(Dataset):
    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode: str = None,
        num_samples = None,
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.sample_mode = sample_mode
        self.num_samples = setparam(args, num_samples, 'num_samples')

        assert sample_mode in ['near', 'rand']
        
        self.V, self.F = load_obj(self.dataset_path)
        self.V, self.F = normalize(self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""
        if self.sample_mode == 'near':
            self.pts = sample_near_surface_2(self.V, self.F, self.num_samples)
        elif self.sample_mode == 'rand':
            self.pts = sample_uniform(self.num_samples).to(self.V.device)
        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())
        self.d = self.d[...,None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        return self.pts[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""
        return 1


class OccupancyTester(object):
    def __init__(self, args, args_str):
        """Constructor.
        
        Args:
            args (Namespace): parameters
            args_str (str): string representation of all parameters
            model_name (str): model nametag
        """

        self.args = args 
        self.args_str = args_str
        
        # Set device to use
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        # Initialize
        self.set_dataset()
        self.set_network()
        self.set_logger()
        
    #######################
    # __init__ helper functions
    #######################

    def set_dataset(self):
        self.near_surf_ds = TestingDataset(self.args, sample_mode='near')

        log.info("Near Surface Size: {}".format(len(self.near_surf_ds)))
        
        self.near_surf_dl = DataLoader(self.near_surf_ds, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)

        log.info("Loaded Near Surface Dataset")

        self.uniform_ds = TestingDataset(self.args, sample_mode='rand')

        log.info("Near Surface Size: {}".format(len(self.uniform_ds)))
        
        self.uniform_dl = DataLoader(self.uniform_ds, batch_size=self.args.batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=0)

        log.info("Loaded Near Surface Dataset")
            
    def set_network(self):
        """
        Override this function if using a custom network, that does not use the default args based
        initialization, or if you need a custom network initialization scheme.
        """
        self.net = globals()[self.args.net](self.args)
        if self.args.jit:
            self.net = torch.jit.script(self.net)

        if self.args.pretrained:
            self.net.load_state_dict(torch.load(self.args.pretrained))

        self.net.to(self.device)

        log.info("Total number of parameters: {}".format(sum(p.numel() for p in self.net.parameters())))

    def set_logger(self):
        """
        Override this function to use custom loggers.
        """
        if self.args.exp_name:
            self.log_fname = self.args.exp_name
        else:
            self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(self.args.logs, self.log_fname)
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Parameters', self.args_str)

        log.info('Model configured and ready to go')

    #######################
    # occupancy tests
    #######################
        
    def test_occupancy(self):
        for (dl, test_name, test_f1_thr) in [
            (self.uniform_dl, "Volume", 0.95),
            (self.near_surf_dl, "Surface", 0.9),
        ]:
            pred_occupancies_by_lod = [
                np.array([], dtype=np.bool8)
                for lod in range(self.args.num_lods)
            ]
            gt_occupancies = np.array([], dtype=np.bool8)

            for data in dl:
                pts = data[0].to(self.device)
                sdf_gts = data[1].to(self.device)

                gt_occupancies = np.append(gt_occupancies, sdf_gts.cpu().numpy().flatten() < 0)

                sdf_preds_by_lod = []
                with torch.no_grad():
                    for lod in range(self.args.num_lods):
                        sdf_preds_by_lod.append(self.net.sdf(pts, lod=lod))

                for sdf_pred, lod in zip(sdf_preds_by_lod, range(self.args.num_lods)):
                    # l2_loss = ((sdf_pred - sdf_gts)**2).sum()
                    pred_occupancies_by_lod[lod] = np.append(
                        pred_occupancies_by_lod[lod],
                        sdf_pred.cpu().numpy().flatten() < 0
                    )
            
            all_pass = True

            for lod in range(self.args.num_lods):
                test_f1 = f1_score(gt_occupancies, pred_occupancies_by_lod[lod])
                test_pass = test_f1 > test_f1_thr
                all_pass = all_pass and test_pass
                log.info(f"{test_name} points Occupancy F1, LOD {lod}: {test_f1:.2f} - {'Pass' if test_pass else 'Fail'}")

            log.info(f"{test_name} TESTS: {'PASS' if all_pass else 'FAIL'}")


def main():
    args, args_str = parse_options()
    log.info(f'Parameters: \n{args_str}')
    log.info(f'Training on {args.dataset_path}')
    model = OccupancyTester(args, args_str)
    model.test_occupancy()


if __name__ == "__main__":
    main()

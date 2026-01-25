"""
Logging and visualization utilities.
"""

import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd

from . import utils as Util


class InfoLogger:
    """
    Logger that records to file, only works on GPU 0.

    Args:
        opt: Configuration dictionary with 'global_rank', 'path', and 'phase' keys
    """

    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'],
                          level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0:  # Only log on GPU 0
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)

            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper

    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """Set up logger."""
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)


class VisualWriter:
    """
    TensorBoard writer with results saving functionality.

    Args:
        opt: Configuration dictionary
        logger: InfoLogger instance
    """

    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank == 0:
            log_dir = str(log_dir)

            # Retrieve visualization writer
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = (
                    "Warning: visualization (Tensorboard) is configured to use, but currently not installed. "
                    "Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to "
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the config."
                )
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        """Set current iteration info."""
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        """Save result images to disk."""
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        try:
            names = results['name']
            outputs = Util.postprocess(results['result'])
            for i in range(len(names)):
                Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
        except Exception:
            raise NotImplementedError(
                'You must specify the context of name and result in save_current_results functions of model.'
            )

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            print('Close the Tensorboard SummaryWriter.')

    def __getattr__(self, name):
        """
        Return add_data() methods of tensorboard with additional information.
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # Add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(
                    "type object '{}' has no attribute '{}'".format(self.selected_module, name)
                )
            return attr


class LogTracker:
    """
    Record training numerical indicators.

    Args:
        *keys: Metric names to track
        phase: 'train' or 'val'
    """

    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """Reset all metrics."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """Update metric with new value."""
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """Get average value for metric."""
        return self._data.average[key]

    def result(self):
        """Get all results as dictionary."""
        return {'{}/{}'.format(self.phase, k): v for k, v in dict(self._data.average).items()}

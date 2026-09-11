import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import neuralhydrology.training.loss as loss
from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import load_basin_file, load_scaler
from neuralhydrology.evaluation import get_tester
from neuralhydrology.evaluation.tester import BaseTester
from neuralhydrology.modelzoo import get_model
from neuralhydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging

import pickle


LOGGER = logging.getLogger(__name__)


class BaseTrainer(object):
    """Default class to train a model.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._scaler = {}
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch
        self._is_distributed = cfg.is_distributed
        self._rank = cfg.distributed_rank
        self._world_size = cfg.distributed_world_size
        self._local_rank = cfg.distributed_local_rank
        self._is_main_process = self._rank == 0
        if not self._is_main_process:
            self._disable_pbar = True

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        log_file = "output.log" if self._is_main_process else f"output_rank{self._rank}.log"
        setup_logging(str(self.cfg.run_dir / log_file))
        LOGGER.info(f"### Folder structure created at {self.cfg.run_dir}")

        if self.cfg.is_continue_training:
            LOGGER.info(f"### Continue training of run stored in {self.cfg.base_run_dir}")

        if self.cfg.is_finetuning:
            LOGGER.info(f"### Start finetuning with pretrained model stored in {self.cfg.base_run_dir}")

        LOGGER.info(f"### Run configurations for {self.cfg.experiment_name}")
        for key, val in self.cfg.as_dict().items():
            LOGGER.info(f"{key}: {val}")

        self._set_random_seeds()
        self._set_device()


    def _get_dataset(self) -> BaseDataset:
        return get_dataset(cfg=self.cfg, period="train", is_train=True, scaler=self._scaler)

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(model=self.model, cfg=self.cfg)

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self):
        self.loss_obj.set_regularization_terms(get_regularization_obj(cfg=self.cfg))

    def _get_tester(self) -> BaseTester:
        return get_tester(cfg=self.cfg, run_dir=self.cfg.run_dir, period="validation", init_model=False)

    def _get_data_loader(self, ds: BaseDataset) -> torch.utils.data.DataLoader:
        sampler = None
        if self._is_distributed:
            sampler = DistributedSampler(ds,
                                         num_replicas=self._world_size,
                                         rank=self._rank,
                                         shuffle=True,
                                         drop_last=False)

        return DataLoader(ds,
                          batch_size=self.cfg.batch_size,
                          shuffle=sampler is None,
                          sampler=sampler,
                          num_workers=self.cfg.num_workers,
                          collate_fn=ds.collate_fn)

    def _get_model_without_ddp(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _get_distributed_safe_train_dataset(self) -> BaseDataset:
        """Create/load the train dataset without concurrent raw NetCDF reads on all DDP ranks."""
        if (not self._is_distributed) or (self.cfg.train_data_file is not None):
            return self._get_dataset()

        train_data_file = self.cfg.train_dir / "train_data.p"
        original_save_train_data = self.cfg.save_train_data
        original_train_data_file = self.cfg.train_data_file

        if self._rank == 0:
            self.cfg.update_config({"save_train_data": True})
            ds = self._get_dataset()
            self.cfg.update_config({"save_train_data": original_save_train_data})
            self._distributed_barrier()
            return ds

        self._distributed_barrier()
        self.cfg.update_config({"train_data_file": train_data_file})
        ds = self._get_dataset()
        self.cfg.update_config({"train_data_file": original_train_data_file})
        return ds

    def _distributed_barrier(self):
        if not self._is_distributed:
            return

        if torch.cuda.is_available() and self._local_rank is not None:
            try:
                dist.barrier(device_ids=[self._local_rank])
                return
            except TypeError:
                # Older PyTorch versions do not support barrier(device_ids=...).
                pass
        dist.barrier()

    def _freeze_model_parts(self):
        # freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        unresolved_modules = []

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self.cfg.finetune_modules, list):
            for module_part in self.cfg.finetune_modules:
                if module_part in self.model.module_parts:
                    module = getattr(self.model, module_part)
                    for param in module.parameters():
                        param.requires_grad = True
                else:
                    unresolved_modules.append(module_part)
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self.cfg.finetune_modules.items():
                if module_group in self.model.module_parts:
                    if isinstance(module_parts, str):
                        module_parts = [module_parts]
                    for module_part in module_parts:
                        module = getattr(self.model, module_group)[module_part]
                        for param in module.parameters():
                            param.requires_grad = True
                else:
                    unresolved_modules.append(module_group)
        if unresolved_modules:
            LOGGER.warning(f"Could not resolve the following module parts for finetuning: {unresolved_modules}")

    def initialize_training(self):
        """Initialize the training class.

        This method will load the model, initialize loss, regularization, optimizer, dataset and dataloader,
        tensorboard logging, and Tester class.
        If called in a ``continue_training`` context, this model will also restore the model and optimizer state.
        """
        if self.cfg.is_finetuning:
            # Load scaler from pre-trained model.
            self._scaler = load_scaler(self.cfg.base_run_dir)

        # Initialize dataset before the model is loaded.
        ds = self._get_distributed_safe_train_dataset()
        if len(ds) == 0:
            raise ValueError("Dataset contains no samples.")
        self.loader = self._get_data_loader(ds=ds)

        self.model = self._get_model().to(self.device)
        if self.cfg.checkpoint_path is not None:
            LOGGER.info(f"Starting training from Checkpoint {self.cfg.checkpoint_path}")
            self.model.load_state_dict(torch.load(str(self.cfg.checkpoint_path), map_location=self.device))
        elif self.cfg.checkpoint_path is None and self.cfg.is_finetuning:
            # the default for finetuning is the last model state
            checkpoint_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            LOGGER.info(f"Starting training from checkpoint {checkpoint_path}")
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))


        # Jay: Load the pre-trained weight
        # print("\n")
        # print(self.model.experts)
        # print("\n")

        # print("\n")
        # print("Loading pre-trained basic LSTM...")
        # state_dict_lstm1 = torch.load("pretrained/lstm1/model_epoch000.pt", map_location=self.device)

        # state_dict_lstm1 = {k.replace("lstm.", ""): v for k, v in state_dict_lstm1.items() if k.startswith("lstm.")}

        # self.model.experts[0].load_state_dict(state_dict_lstm1)
        # self.model.experts[1].load_state_dict(state_dict_lstm1)


        # # Freeze lstm_basic
        # for param in self.model.lstm_basic.parameters():
        #     param.requires_grad = False

        # # Freeze lstm_peak
        # for param in self.model.lstm_peak.parameters():
        #     param.requires_grad = False


        # Freeze model parts from pre-trained model.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        if self._is_distributed:
            if self.device.type != "cuda":
                raise RuntimeError("Distributed training for this example expects CUDA devices.")
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.device.index],
                                                 output_device=self.device.index)

        self.optimizer = self._get_optimizer()
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add possible regularization terms to the loss function.
        self._set_regularization()

        # restore optimizer and model state if training is continued
        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # set epoch and iteration step counter to continue from the selected checkpoint
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                warn_msg = [
                    f"Validation set to validate every {self.cfg.validate_every} epoch(s), but ",
                    "'validate_n_random_basins' not set or set to zero. Will validate on the entire validation set."
                ]
                LOGGER.warning("".join(warn_msg))
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=self.cfg.target_noise_std)
            self._target_mean = torch.from_numpy(
                ds.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values).to(self.device)
            self._target_std = torch.from_numpy(
                ds.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values).to(self.device)
            

        # Define Learning Rate Scheduler
        self._ema_beta = 0.9
        self._loss_ema = None
        self.scheduler = ReduceLROnPlateau(
                            self.optimizer,
                            mode="min",
                            factor=0.1,       # reduce LR by 10x
                            patience=5,       # wait 5 epochs with no improvement
                            threshold=1e-3,   # require ~0.1% improvement
                            threshold_mode="rel",
                            cooldown=2,       # small cooldown after a reduce
                            min_lr=1e-6       # floor
                        )


    def train_and_validate(self):
        """Train and validate the model.

        Train the model for the number of epochs specified in the run configuration, and perform validation after every
        ``validate_every`` epochs. Model and optimizer state are saved after every ``save_weights_every`` epochs.
        """

        # training & evaluation records
        train_mse = []
        # valid_nse = []
        # valid_mse = []
        # valid_rmse = []
        # valid_kge = []
        # valid_fhv = []
        # valid_flv = []
        # valid_pearsonr = []
        # valid_pmape = []
        # valid_ptime = []

        # Saving criteria
        save_threshold = 1000000

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            epoch_start_time = time.time()
            # if epoch in self.cfg.learning_rate.keys():
            #     LOGGER.info(f"Setting learning rate to {self.cfg.learning_rate[epoch]}")
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.cfg.learning_rate[epoch]

            success = self._train_epoch(epoch=epoch)
            success = self._sync_training_success(success)
            if not success:
                if self._is_main_process:
                    elapsed_seconds = time.time() - epoch_start_time
                    print(f"Epoch {epoch} time: {elapsed_seconds:.2f} seconds")
                break

            avg_losses = self.experiment_logger.summarise()
            avg_losses = self._sync_loss_dict(avg_losses)
            loss_str = ", ".join(f"{k}: {v:.5f}" for k, v in avg_losses.items())
            if self._is_main_process:
                LOGGER.info(f"Epoch {epoch} average loss: {loss_str}")

            # if epoch % self.cfg.save_weights_every == 0:
            #     self._save_weights_and_optimizer(epoch)

            self._distributed_barrier()

            if self._is_main_process and (self.validator is not None) and (epoch % self.cfg.validate_every == 0):
                self.validator.evaluate(epoch=epoch,
                                        save_results=self.cfg.save_validation_results,
                                        save_all_output=self.cfg.save_all_output,
                                        metrics=self.cfg.metrics,
                                        model=self._get_model_without_ddp(),
                                        experiment_logger=self.experiment_logger.valid())

                valid_metrics = self.experiment_logger.summarise()
                print_msg = f"Epoch {epoch} average validation loss: {valid_metrics['avg_total_loss']:.5f}"
                if self.cfg.metrics:
                    print_msg += f" -- Median validation metrics: "
                    print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
                    LOGGER.info(print_msg)

            self._distributed_barrier()


            # Save best checkpoint based on the validation NSE
            # if epoch % self.cfg.save_weights_every == 0:
            #     if valid_metrics['avg_total_loss'] <= total_loss_threshold:
            #         total_loss_threshold = valid_metrics['avg_total_loss']
            #         self._save_weights_and_optimizer(0)


            train_loss = avg_losses["avg_loss"]

            if train_loss <= save_threshold:
                save_threshold = train_loss
                if self._is_main_process:
                    self._save_weights_and_optimizer(0)
            

            # Learning Rate: ReduceOnPlateau
            if self._loss_ema is None:
                self._loss_ema = train_loss
            else:
                self._loss_ema = self._ema_beta*self._loss_ema + (1-self._ema_beta)*train_loss
            
            self.scheduler.step(self._loss_ema)
            if self._is_main_process:
                LOGGER.info(f"epoch={epoch} loss={train_loss:.6f} ema={self._loss_ema:.6f} lr={self.optimizer.param_groups[0]['lr']:.2e}")

            if self.cfg.model == "moe_tau":
                model = self._get_model_without_ddp()
                if self._is_main_process:
                    print("Current tau: {}\n".format(model.gating_net.current_tau))

            # record training info
            train_mse.append(train_loss)

            if self._is_main_process:
                elapsed_seconds = time.time() - epoch_start_time
                print(f"Epoch {epoch} time: {elapsed_seconds:.2f} seconds")

            # valid_nse.append(valid_metrics["NSE"])
            # valid_mse.append(valid_metrics["MSE"])
            # valid_rmse.append(valid_metrics["RMSE"])
            # valid_kge.append(valid_metrics["KGE"])
            # valid_fhv.append(valid_metrics["FHV"])
            # valid_flv.append(valid_metrics["FLV"])
            # valid_pearsonr.append(valid_metrics["Pearson-r"])
            # valid_pmape.append(valid_metrics["Peak-MAPE"])
            # valid_ptime.append(valid_metrics["Peak-Timing"])

        if self._is_main_process:
            self.plot_loss(train_mse)


        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

        # Save training & validation records
        # with open("./records/train_mse.pkl", "wb") as file:
        #     pickle.dump(train_mse, file)

        # with open("./records/valid_nse.pkl", "wb") as file:
        #     pickle.dump(valid_nse, file)
        
        # with open("./records/valid_mse.pkl", "wb") as file:
        #     pickle.dump(valid_mse, file)

        # with open("./records/valid_rmse.pkl", "wb") as file:
        #     pickle.dump(valid_rmse, file)

        # with open("./records/valid_kge.pkl", "wb") as file:
        #     pickle.dump(valid_kge, file)
        
        # with open("./records/valid_fhv.pkl", "wb") as file:
        #     pickle.dump(valid_fhv, file)
        
        # with open("./records/valid_flv.pkl", "wb") as file:
        #     pickle.dump(valid_flv, file)
        
        # with open("./records/valid_pearsonr.pkl", "wb") as file:
        #     pickle.dump(valid_pearsonr, file)
        
        # with open("./records/valid_pmape.pkl", "wb") as file:
        #     pickle.dump(valid_pmape, file)
        
        # with open("./records/valid_ptime.pkl", "wb") as file:
        #     pickle.dump(valid_ptime, file)


    def plot_loss(self, loss_values, filename_prefix="loss_plot"):
        epochs = range(1, len(loss_values) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_values, linestyle="-", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Mean-Squared Error (MSE)")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()

        # Add timestamp to avoid overwriting
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"

        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


    def _get_start_epoch_number(self):
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = [x for x in sorted(list(self.cfg.run_dir.glob('model_epoch*.pt')))][-1]
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self):
        if self.cfg.continue_from_epoch is not None:
            epoch = f"{self.cfg.continue_from_epoch:03d}"
            weight_path = self.cfg.base_run_dir / f"model_epoch{epoch}.pt"
        else:
            weight_path = [x for x in sorted(list(self.cfg.base_run_dir.glob('model_epoch*.pt')))][-1]
            epoch = weight_path.name[-6:-3]

        optimizer_path = self.cfg.base_run_dir / f"optimizer_state_epoch{epoch}.pt"

        LOGGER.info(f"Continue training from epoch {int(epoch)}")
        self._get_model_without_ddp().load_state_dict(torch.load(weight_path, map_location=self.device))
        self.optimizer.load_state_dict(torch.load(str(optimizer_path), map_location=self.device))

    def _save_weights_and_optimizer(self, epoch: int):
        weight_path = self.cfg.run_dir / f"model_epoch{epoch:03d}.pt"
        torch.save(self._get_model_without_ddp().state_dict(), str(weight_path))

        optimizer_path = self.cfg.run_dir / f"optimizer_state_epoch{epoch:03d}.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.experiment_logger.train()
        if isinstance(self.loader.sampler, DistributedSampler):
            self.loader.sampler.set_epoch(epoch)

        # process bar handle
        n_iter = min(self._max_updates_per_epoch, len(self.loader)) if self._max_updates_per_epoch is not None else None
        pbar = tqdm(self.loader, file=sys.stdout, disable=self._disable_pbar, total=n_iter)
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        nan_count = 0
        # y_list = []
        for i, data in enumerate(pbar):
            if self._max_updates_per_epoch is not None and i >= self._max_updates_per_epoch:
                break

            for key in data.keys():
                if not key.startswith('date'):
                    data[key] = data[key].to(self.device)

            # apply possible pre-processing to the batch before the forward pass
            data = self._get_model_without_ddp().pre_model_hook(data, is_train=True)

            # get predictions
            # print("\n")
            # print(data.keys())
            # print("x_d: {}\n".format(data["x_d"]))
            # print("y: {}\n".format(data["y"]))
            # print("date: {}".format(data["date"]))
            # print("\n")

            # y_last = data["y"].squeeze(-1)[:, -1]
            # y_list.append(y_last)

            predictions = self.model(data)

            if self.noise_sampler_y is not None:
                for key in filter(lambda k: 'y' in k, data.keys()):
                    noise = self.noise_sampler_y.sample(data[key].shape)
                    # make sure we add near-zero noise to originally near-zero targets
                    data[key] += (data[key] + self._target_mean / self._target_std) * noise.to(self.device)

            loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    # raise RuntimeError(f"Loss was NaN for {nan_count} times in a row. Stopped training.")
                    return False
                LOGGER.warning(f"Loss is Nan; ignoring step. (#{nan_count}/{self._allow_subsequent_nan_losses})")
            else:
                nan_count = 0

                # delete old gradients
                self.optimizer.zero_grad()

                # get gradients
                loss.backward()

                if self.cfg.clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradient_norm)

                # update weights
                self.optimizer.step()

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            self.experiment_logger.log_step(**{k: v.item() for k, v in all_losses.items()})

        # y_list = torch.cat(y_list)                 # -> shape [N]

        # mean = y_list.mean()
        # std  = y_list.std(unbiased=False)        # use unbiased=True for sample std (PyTorch<2.0 default)

        # print("mean: {}".format(mean))
        # print("std: {}".format(std))

        return True

    def _set_random_seeds(self):
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))

        # fix random seeds for various packages
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self):
        if self.cfg.device is not None:
            if self.cfg.device.startswith("cuda"):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id >= torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"### Device {self.device} will be used for training")

    def _create_folder_structure(self):
        # create as subdirectory within run directory of base run
        if self.cfg.is_continue_training:
            folder_name = f"continue_training_from_epoch{self._epoch:03d}"

            # store dir of base run for easier access in weight loading
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # create as new folder structure
        else:
            now = datetime.now()
            day = f"{now.day}".zfill(2)
            month = f"{now.month}".zfill(2)
            hour = f"{now.hour}".zfill(2)
            minute = f"{now.minute}".zfill(2)
            second = f"{now.second}".zfill(2)
            run_name = self.cfg.run_name or f'{self.cfg.experiment_name}_{day}{month}_{hour}{minute}{second}'

            # if no directory for the runs is specified, a 'runs' folder will be created in the current working dir
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / "runs" / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # create folder + necessary subfolder
        if self._is_main_process:
            if not self.cfg.run_dir.is_dir():
                self.cfg.train_dir = self.cfg.run_dir / "train_data"
                self.cfg.train_dir.mkdir(parents=True)
            else:
                raise RuntimeError(f"There is already a folder at {self.cfg.run_dir}")
        else:
            self.cfg.train_dir = self.cfg.run_dir / "train_data"

        self._distributed_barrier()

        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / "img_log"
            if self._is_main_process:
                self.cfg.img_log_dir.mkdir(parents=True)
            self._distributed_barrier()

    def _sync_loss_dict(self, losses: Dict[str, float]) -> Dict[str, float]:
        if not self._is_distributed:
            return losses

        synced_losses = {}
        for key, value in losses.items():
            tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced_losses[key] = (tensor / self._world_size).item()
        return synced_losses

    def _sync_training_success(self, success: bool) -> bool:
        if not self._is_distributed:
            return success

        tensor = torch.tensor(1 if success else 0, dtype=torch.int32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

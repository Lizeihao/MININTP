import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil
from typing import Optional
import math
import json
import argparse
import logging
import torch
from torch import nn
import deepspeed
from deepspeed import get_accelerator
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.ntp_dataset import PretrainDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

class Trainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.current_epoch = 0

        # Initialize model and dataset
        self.init_components()
        # Initialize distributed training
        self.deepspeed_init()

        # Print model parameters (only on main process)
        if self.model_engine.global_rank == 0:
            # Get trainable parameters (in billions)
            self.trainable_params = round(
                sum(
                    p.numel() for p in self.model_engine.module.parameters() if p.requires_grad
                ) / 1e9, 
                3
            )
            logger.info(f"Trainable parameters: {self.trainable_params:.3f}B")

        # Auto-detect latest checkpoint
        if not self.args.load_checkpoint:
            latest_ckpt = self._find_latest_checkpoint()
            if latest_ckpt:
                self.args.load_checkpoint = latest_ckpt
                logger.info(f"Detected latest checkpoint: {latest_ckpt}")

        # Load checkpoint (if exists)
        self.load_checkpoint()

        self.output_dir = self.args.save_dir + f"/miniNTP-{self.args.model_number}"

    def deepspeed_init(self):
        """Initialize Deepspeed engine"""
        ds_config = json.load(open(self.args.deepspeed_config, 'r', encoding="utf-8"))
        ds_config["train_micro_batch_size_per_gpu"] = self.args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
        ds_config["optimizer"]["params"]["lr"] = self.args.learning_rate
        ds_config["scheduler"]["params"]["total_num_steps"] = self.args.total_num_steps
        ds_config["scheduler"]["params"]["warmup_num_steps"] = self.args.num_warmup_steps

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=parameters,
            config_params=ds_config
        )

        self.train_loader = self.model_engine.deepspeed_io(self.train_dataset)
        self.test_loader = self.model_engine.deepspeed_io(self.test_dataset)

    def init_components(self):
        """Initialize model, dataset and other components"""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('/home/kas/minimind/model')
        
        # Initialize model
        config = MiniMindConfig(
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            num_attention_heads=self.args.num_attention_heads,
            num_key_value_heads=self.args.num_attention_heads
        )
        self.model = MiniMindForCausalLM(config)
        
        # Initialize dataset
        self.train_dataset = PretrainDataset(
            self.args.pretrain_data_path,
            self.tokenizer,
            max_length=self.args.max_train_seq_len,
            data_size=self.args.data_size
        )
        self.test_dataset = PretrainDataset(
            self.args.test_data_path,
            self.tokenizer,
            max_length=self.args.max_test_seq_len
        )

        world_size = torch.distributed.get_world_size()
        num_update_steps_per_epoch = math.ceil(len(self.train_dataset) / (self.args.per_device_train_batch_size * world_size * self.args.gradient_accumulation_steps))
        num_training_steps = self.args.epochs * num_update_steps_per_epoch
        self.args.total_num_steps = num_training_steps

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Automatically find the latest checkpoint in the save directory"""
        if not os.path.exists(self.output_dir):
            return None
        
        checkpoint_dirs = []
        for dir_name in os.listdir(self.output_dir):
            dir_path = os.path.join(self.output_dir, dir_name)
            if os.path.isdir(dir_path):
                # Parse global_step from directory name
                step = self._extract_step_from_dirname(dir_name)
                if step is not None:
                    checkpoint_dirs.append((step, dir_path))
        
        if not checkpoint_dirs:
            return None
        
        # Sort by global_step in descending order
        checkpoint_dirs.sort(reverse=True, key=lambda x: x[0])
        return checkpoint_dirs[0][1]
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Clean up old checkpoints (keep specified number)"""
        checkpoint_dirs = []
        for dir_name in os.listdir(self.output_dir):
            dir_path = os.path.join(self.output_dir, dir_name)
            if os.path.isdir(dir_path):
                step = self._extract_step_from_dirname(dir_name)
                if step is not None:
                    checkpoint_dirs.append((step, dir_path))
        
        if len(checkpoint_dirs) > keep:
            checkpoint_dirs.sort(reverse=True, key=lambda x: x[0])
            
            # Remove checkpoints exceeding the keep limit
            for step, path in checkpoint_dirs[keep:]:
                try:
                    logger.info(f"Removing old checkpoint: {os.path.basename(path)}")
                    shutil.rmtree(path)
                except:
                    logger.info(f"Failed to remove checkpoint: {os.path.basename(path)}")

    def load_checkpoint(self):
        """Load checkpoint (with auto-detection support)"""
        if self.args.load_checkpoint:
            load_path, tag = os.path.split(self.args.load_checkpoint)

            _, client_state = self.model_engine.load_checkpoint(
                load_dir=load_path,
                tag=tag,
                load_optimizer_states=True,
                load_lr_scheduler_states=True
            )
            
            if client_state:
                self.current_epoch = client_state.get("epoch", 0)
                self.global_step = client_state.get("global_step", 0) + 1
                logger.info(f"Successfully loaded checkpoint | Epoch: {self.current_epoch} | Step: {self.global_step}")
            
            torch.distributed.barrier()

    def _extract_step_from_dirname(self, dir_name: str) -> Optional[int]:
        """Extract global_step from directory name (supports multiple naming formats)"""
        match = re.search(r'(?:^|_)step(\d+)', dir_name)
        return int(match.group(1)) if match else None

    def save_checkpoint(self, custom_tag: Optional[str] = None):
        """Save checkpoint and clean up old files"""
        tag = custom_tag or f"epoch{self.current_epoch}_step{self.global_step}"
            
        self.model_engine.save_checkpoint(
            save_dir=self.output_dir,
            tag=tag,
            client_state={
                "epoch": self.current_epoch,
                "global_step": self.global_step
            }
        )
            
        self._cleanup_old_checkpoints()
    
    def save_final_model(self):
        """Save final model after training"""
        if self.model_engine.global_rank == 0:
            # Get original model (unwrap Deepspeed)
            model_to_save = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine

            state_dict = model_to_save.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
            
            model_to_save.save_pretrained(
                self.output_dir,
                state_dict=state_dict,
                safe_serialization=False 
            )
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"Final model saved to: {self.output_dir}")

        # Also save Deepspeed checkpoint
        self.save_checkpoint(custom_tag="final_step{}".format(self.global_step))

    def train(self, wandb):
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        total_loss = 0.0

        while self.current_epoch < self.args.epochs:
            self.model_engine.train()
            for _, batch in enumerate(self.train_loader):
                if self.global_step == 0:
                    print_rank_0(f"Batch_train_size is: {len(batch[0])}", self.model_engine.global_rank)
                # Move data to device
                inputs, labels, mask = [t.to(self.model_engine.local_rank) for t in batch]
                
                # Forward pass
                outputs = self.model_engine(inputs)
                    
                # Calculate masked loss
                loss = loss_fct(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)
                ).view(labels.size())
                loss = (loss * mask).sum() / mask.sum()
                # loss += outputs.aux_loss
                total_loss += loss.float()
                
                self.model_engine.backward(loss)
                self.model_engine.step()
                
                # Logging (using accumulated total loss)
                if self.global_step % self.args.log_interval == 0:
                    total_loss = get_all_reduce_mean(total_loss).item()

                    if self.global_step == 0:
                        avg_train_loss = total_loss
                    else:
                        avg_train_loss = total_loss / self.args.log_interval

                    total_loss = 0.0

                    # Evaluate on validation set
                    test_loss, perplexity = self.evaluate(self.test_loader)

                    lr = self.optimizer.param_groups[0]['lr']
                        
                    # Log output
                    if self.model_engine.global_rank == 0:
                        logger.info(
                            f"Epoch: {self.current_epoch+1}/{self.args.epochs} | "
                            f"Step: {self.global_step} | "
                            f"Train Loss: {avg_train_loss:.3f} | "
                            f"Test Loss: {test_loss:.3f} | "
                            f"PPL: {perplexity:.3f} | "
                            f"LR: {lr:.6e}"
                        )
                        
                    # WandB logging (if enabled)
                    if self.args.use_wandb and self.model_engine.global_rank == 0:
                        wandb.log({
                            "train_loss": avg_train_loss,
                            "test_loss": test_loss,
                            "gen_gap": abs(avg_train_loss - test_loss),
                            "ppl": perplexity,
                            "global_step": self.global_step
                        })

                self.global_step += 1

                # Save checkpoint
                if self.global_step % self.args.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()

                if self.global_step >= self.args.total_steps:
                    break

            self.current_epoch += 1

        # Save final model after training
        self.save_final_model()

    def evaluate(self, dataloader):
        """Model evaluation"""
        self.model_engine.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx == 0 and self.global_step == 0:
                    print_rank_0(f"Batch_eval_size is: {len(batch[0])}", self.model_engine.global_rank)
                inputs, labels, mask = [t.to(self.model_engine.local_rank) for t in batch]
                outputs = self.model_engine(inputs)
                loss = nn.CrossEntropyLoss(reduction='none')(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.view(-1)
                ).view(labels.size())
                loss = (loss * mask).sum() / mask.sum()
                eval_loss += loss.float() / len(dataloader)
        eval_loss = get_all_reduce_mean(eval_loss)
        try:
            perplexity = torch.exp(eval_loss).item()
        except OverflowError:
            perplexity = float("inf")

        return eval_loss.item(), perplexity

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialize trainer
    if args.local_rank == -1:
        args.device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        args.device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()
    
    args.global_rank = torch.distributed.get_rank()

    trainer = Trainer(args)
    # Initialize wandb
    if args.use_wandb:
        import wandb
        WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
        if WANDB_API_KEY:
            wandb.login(key=WANDB_API_KEY)
        if trainer.model_engine.global_rank == 0:
            args.wandb_run_name = f"MiniLLM-{args.model_number}-{trainer.trainable_params}B\
                    -Epoch-{trainer.current_epoch}-step-{trainer.global_step}-m-{args.max_train_seq_len}\
                    -d-{args.hidden_size}-L-{args.num_hidden_layers}-H-{args.num_attention_heads}-s-{args.data_size}"
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        else:
            wandb = None
    else:
        wandb = None
    # Start training
    trainer.train(wandb)

def parse_args():
    parser = argparse.ArgumentParser(description="Deepspeed NTP-Pretraining")
    
    parser.add_argument('--local_rank', type=int, default=-1)

    # Model parameters
    parser.add_argument("--model_number", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--log_interval", type=int, default=32)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=20,
        help="Number of steps for the warmup in the lr scheduler.")
    
    # Data parameters
    parser.add_argument("--pretrain_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--max_train_seq_len", type=int, default=512)
    parser.add_argument("--max_test_seq_len", type=int, default=512)
    parser.add_argument("--data_size", type=float, default=1.0)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    # Path parameters
    parser.add_argument("--save_dir", type=str, default="../output/pretrain")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # wandb
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="miniNTP-Pretrain")
    
    # Deepspeed configuration
    parser.add_argument("--deepspeed_config", type=str, required=True)
    
    # seed
    parser.add_argument("--seed", type=int, default=1234)
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
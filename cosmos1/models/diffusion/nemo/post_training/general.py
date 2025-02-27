# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import nemo_run as run
from huggingface_hub import snapshot_download
from nemo.collections import llm
from nemo.collections.diffusion.models.model import DiT7BConfig, DiT14BConfig, DiT7BExtendConfig
from nemo.collections.diffusion.train import pretrain, videofolder_datamodule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


@run.cli.factory(target=llm.train)
def cosmos_diffusion_7b_text2world_lora() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT7BConfig)

    # Trainer setup
    recipe.trainer.max_steps = 1000
    recipe.optim.config.lr = 1e-6

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    recipe.trainer.strategy.sequence_parallel = False
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    recipe.trainer.strategy.grad_reduce_in_fp32 = False
    recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = False
    
    # Activation Checkpointing
    # recipe.model.config.recompute_granularity = "full"
    # recipe.model.config.recompute_method = "uniform"
    # recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = ""  # path to folder with processed dataset

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint
    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_7b_text2world_lora"
    recipe.trainer.strategy.sequence_parallel = False
    recipe.model_transform = run.Config(llm.peft.LoRA,
        target_modules=['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'],
        dim=256,
    )

    return recipe

@run.cli.factory(target=llm.train)
def cosmos_diffusion_7b_video2world_finetune() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT7BExtendConfig)
    recipe.model.config.vae_path = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")

    # Trainer setup
    recipe.trainer.max_steps = 1000
    recipe.optim.config.lr = 1e-6

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = False
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    
    recipe.trainer.strategy.grad_reduce_in_fp32 = False
    recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = False
    
    # # Activation Checkpointing
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = ""  # path to folder with processed dataset
    
    # validation setup
    recipe.trainer.num_sanity_val_steps = 8
    recipe.trainer.limit_val_batches = 8
    recipe.trainer.check_val_every_n_epoch = 5

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = "/home/anw2067/scratch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-7B-Video2World-nemo-zarr"
    recipe.resume.resume_if_exists = False
    recipe.trainer.strategy.save_ckpt_format = "torch_dist"

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_7b_video2world_finetune"

    return recipe

@run.cli.factory(target=llm.train)
def cosmos_diffusion_7b_text2world_finetune() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT7BConfig)

    # Trainer setup
    recipe.trainer.max_steps = 1000
    recipe.optim.config.lr = 1e-6

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = False
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True
    
    # # Activation Checkpointing
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = ""  # path to folder with processed dataset

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Text2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint
    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_7b_text2world_finetune"

    return recipe


@run.cli.factory(target=llm.train)
def cosmos_diffusion_14b_text2world_finetune() -> run.Partial:
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT14BConfig)

    # Trainer setup
    recipe.trainer.max_steps = 1000
    recipe.optim.config.lr = 1e-6

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    # Activation Checkpointing
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = ""  # path to folder with processed dataset

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download("nvidia/Cosmos-1.0-Diffusion-14B-Text2World", allow_patterns=["nemo/*"]), "nemo"
    )  # path to diffusion model checkpoint

    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_14b_text2world_finetune"

    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=cosmos_diffusion_7b_text2world_finetune)

from nemo.collections.diffusion.models.model import DiT7BExtendConfig, DiTCrossAttentionModel7BExtend, DiT7BConfig
from cosmos1.models.diffusion.networks.general_dit_video_conditioned import VideoExtendGeneralDIT
import os
import tempfile
from argparse import ArgumentParser

import torch
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.pytorch.plugins.environments import TorchElasticEnvironment
from lightning.pytorch.trainer.trainer import Trainer
from omegaconf import OmegaConf

from nemo.collections.multimodal.models.text_to_image.controlnet.controlnet import MegatronControlNet
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import AppState, logging
from nemo.utils.distributed import initialize_distributed
from pprint import pprint
try:
    from megatron.core import parallel_state
    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/home/anw2067/scratch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-7B-Video2World/model.pt", help="Path to checkpoint.")

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    # parser.add_argument("--nemo_file_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=False, default=1)
    parser.add_argument("--tensor_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument("--pipeline_model_parallel_size", type=int, required=False, default=1)
    parser.add_argument(
        "--pipeline_model_parallel_split_rank",
        type=int,
        required=False,
        default=None,
        help="If pipeline parallel size > 1, this is the rank at which the encoder ends and the decoder begins.",
    )
    parser.add_argument("--local_rank", type=int, required=False, default=os.getenv('LOCAL_RANK', -1))
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")
    parser.add_argument("--model_type", type=str, required=False, default="stable_diffusion")
    parser.add_argument("--nemo_clip_path", type=str, required=False, help="Path to clip ckpt file in .nemo format")

    args = parser.parse_args()
    return args

def merge_state_dicts(target_dict, source_dict):
    """
    Merges state dicts using target keys and source tensors.
    """
    output_dict = {}
    
    def merge_qkv(q, k, v, block_size=128):
        num_blocks = q.shape[0] // block_size
        reshaped = torch.stack([
            q.reshape(num_blocks, block_size, -1),
            k.reshape(num_blocks, block_size, -1),
            v.reshape(num_blocks, block_size, -1)
        ], dim=1)
        return reshaped.reshape(-1, reshaped.shape[-1])

    def merge_kv(k, v, block_size=128):
        num_blocks = k.shape[0] // block_size
        reshaped = torch.stack([
            k.reshape(num_blocks, block_size, -1),
            v.reshape(num_blocks, block_size, -1)
        ], dim=1)
        return reshaped.reshape(-1, reshaped.shape[-1])
    
    def merge_block(target_dict, source_dict, block_idx):
        tgt_base = f"module.decoder.layers.{block_idx}"
        src_base = f"net.blocks.block{block_idx}.blocks"
        
        # Full self attention
        q = source_dict[f"{src_base}.0.block.attn.to_q.0.weight"]
        k = source_dict[f"{src_base}.0.block.attn.to_k.0.weight"]
        v = source_dict[f"{src_base}.0.block.attn.to_v.0.weight"]
        output_dict[f"{tgt_base}.full_self_attention.linear_qkv.weight"] = merge_qkv(q, k, v)
        
        # Cross attention
        q = source_dict[f"{src_base}.1.block.attn.to_q.0.weight"]
        k = source_dict[f"{src_base}.1.block.attn.to_k.0.weight"]
        v = source_dict[f"{src_base}.1.block.attn.to_v.0.weight"]
        output_dict[f"{tgt_base}.cross_attention.linear_kv.weight"] = merge_kv(k, v)
        
        # Direct mappings
        mappings = {
            f"{src_base}.0.block.attn.to_q.1.weight": f"{tgt_base}.full_self_attention.q_layernorm.weight",
            f"{src_base}.0.block.attn.to_k.1.weight": f"{tgt_base}.full_self_attention.k_layernorm.weight",
            f"{src_base}.0.block.attn.to_out.0.weight": f"{tgt_base}.full_self_attention.linear_proj.weight",
            f"{src_base}.1.block.attn.to_q.0.weight": f"{tgt_base}.cross_attention.linear_q.weight",
            f"{src_base}.1.block.attn.to_q.1.weight": f"{tgt_base}.cross_attention.q_layernorm.weight",
            f"{src_base}.1.block.attn.to_k.1.weight": f"{tgt_base}.cross_attention.k_layernorm.weight",
            f"{src_base}.1.block.attn.to_out.0.weight": f"{tgt_base}.cross_attention.linear_proj.weight",
            f"{src_base}.2.block.layer1.weight": f"{tgt_base}.mlp.linear_fc1.weight",
            f"{src_base}.2.block.layer2.weight": f"{tgt_base}.mlp.linear_fc2.weight"
        }
        
        for src, dst in mappings.items():
            output_dict[dst] = source_dict[src]
            
        # AdaLN mapping
        # AdaLN mappings
        output_dict[f"{tgt_base}.adaLN.adaLN_modulation.1.weight"] = torch.cat([
            source_dict[f"{src_base}.0.adaLN_modulation.1.weight"],
            source_dict[f"{src_base}.1.adaLN_modulation.1.weight"],
            source_dict[f"{src_base}.2.adaLN_modulation.1.weight"]
        ], dim=0)
        
        # AdaLN modulation.2 mapping
        target_shape = target_dict[f"{tgt_base}.adaLN.adaLN_modulation.2.weight"].shape
        output_dict[f"{tgt_base}.adaLN.adaLN_modulation.2.weight"] = torch.zeros(target_shape)
        output_dict[f"{tgt_base}.adaLN.adaLN_modulation.2.weight"][:12288, :256] = source_dict[f"{src_base}.0.adaLN_modulation.2.weight"]
        output_dict[f"{tgt_base}.adaLN.adaLN_modulation.2.weight"][12288:2*12288, 256:2*256] = source_dict[f"{src_base}.1.adaLN_modulation.2.weight"]
        output_dict[f"{tgt_base}.adaLN.adaLN_modulation.2.weight"][2*12288:, 2*256:] = source_dict[f"{src_base}.2.adaLN_modulation.2.weight"]
            
    # Process all blocks
    for i in range(28):
        merge_block(target_dict, source_dict, i)
        
    # Copy remaining direct mappings
    mappings = {
        "net.x_embedder.proj.1.weight": "module.x_embedder.proj.1.weight",
        "net.pos_embedder.seq": "module.pos_embedder.seq",
        "net.extra_pos_embedder.pos_emb_h": "module.extra_pos_embedder.pos_emb_h",
        "net.extra_pos_embedder.pos_emb_w": "module.extra_pos_embedder.pos_emb_w",
        "net.extra_pos_embedder.pos_emb_t": "module.extra_pos_embedder.pos_emb_t",
        "net.t_embedder.1.linear_1.weight": "module.t_embedder.1.linear_1.weight",
        "net.t_embedder.1.linear_2.weight": "module.t_embedder.1.linear_2.weight",
        "net.final_layer.linear.weight": "module.final_layer.linear.weight",
        "net.final_layer.adaLN_modulation.1.weight": "module.final_layer.adaLN_modulation.1.weight",
        "net.final_layer.adaLN_modulation.2.weight": "module.final_layer.adaLN_modulation.2.weight",
        "net.affline_norm.weight": "module.affline_norm.weight",
        "logvar.0.freqs": "module.logvar.0.freqs",
        "logvar.0.phases": "module.logvar.0.phases",
        "logvar.1.weight": "module.logvar.1.weight"
    }
    
    for src, dst in mappings.items():
        output_dict[dst] = source_dict[src]
        
    # Ensure the output_dict tensors have the same shape as target_dict tensors
    for key in target_dict:
        if key in output_dict:
            if output_dict[key].shape != target_dict[key].shape:
                print(f"Shape mismatch for {key}: {output_dict[key].shape} vs {target_dict[key].shape}")
                output_dict[key] = torch.zeros_like(target_dict[key])
        else:
            print(f"Missing key in output_dict: {key}")
            output_dict[key] = torch.zeros_like(target_dict[key])
    assert target_dict.keys() == output_dict.keys()
    print("Done matching keys")
    
    return output_dict

def convert(local_rank, rank, world_size, args):
    app_state = AppState()
    app_state.data_parallel_rank = 0
    num_nodes = world_size // args.gpus_per_node
    if args.bcp:
        trainer = Trainer(
            devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu', plugins=[TorchElasticEnvironment()]
        )
    else:
        trainer = Trainer(devices=args.gpus_per_node, num_nodes=num_nodes, accelerator='gpu')

    app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    app_state.tensor_model_parallel_size = args.tensor_model_parallel_size

    # no use atm, use to split ranks in encoder/decoder models.
    if args.pipeline_model_parallel_size > 1 and args.model_type in []:
        if args.pipeline_model_parallel_split_rank is not None:
            app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_split_rank
        else:
            if args.pipeline_model_parallel_size % 2 != 0:
                raise ValueError(
                    f"Pipeline model parallel size {args.pipeline_model_parallel_size} must be even if split rank is not specified."
                )
            else:
                # If split rank is not set, then we set it to be pipeline_model_parallel_size // 2 - this is because in most cases we have the same number of enc/dec layers.
                app_state.pipeline_model_parallel_split_rank = args.pipeline_model_parallel_size // 2
    else:
        app_state.pipeline_model_parallel_split_rank = None

    app_state.model_parallel_size = app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=app_state.tensor_model_parallel_size,
        pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
    )

    app_state.pipeline_model_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    app_state.tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
    
    app_state.is_model_being_restored = True
    
    ### Now load the full model and full checkpoint and copy the weights
    from nemo.collections.diffusion.models.model import DiTModel
    from nemo.lightning.io.pl import MegatronCheckpointIO
    model = DiTModel(DiT7BExtendConfig())
    model.configure_model()
    nemo_dict = model.state_dict()
    nemo_dict = {k: v for k, v in nemo_dict.items() if '_extra_state' not in k}
    
    with open("nemo_dict.txt", "w") as f:
        for k, v in nemo_dict.items():
            f.write(f"{k}: {v.shape}\n")
    
    torch_dict = torch.load("/home/anw2067/scratch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-7B-Video2World/model.pt")

    with open("torch_dict.txt", "w") as f:
        for k, v in torch_dict.items():
            f.write(f"{k}: {v.shape}\n")

    merged_dict = merge_state_dicts(nemo_dict, torch_dict)    
    
    # Load the merged state dict into the DiTModel
    res = model.load_state_dict(merged_dict, strict=False)
    # Ignore missing keys if they contain '_extra_state'
    missing_keys = [key for key in res.missing_keys if '_extra_state' not in key]
    if missing_keys:
        print(f"Missing keys (excluding '_extra_state'): {missing_keys}")
    print("State dict loaded into the model successfully.")
    
    # wrap model.module in megatronparallel for state_dict
    from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel, aggregate_moe_loss_stats
    megatron_parallel = MegatronParallel(
        model
    )
    
    # reinstate the 'module.' in sharded_state_dict
    sharded_state_dict = megatron_parallel.sharded_state_dict()
    
    # Save the merged state dict using MegatronCheckpointIO
    checkpoint_io = MegatronCheckpointIO('zarr')
    output_checkpoint_path = "/home/anw2067/scratch/Cosmos/checkpoints/Cosmos-1.0-Diffusion-7B-Video2World-nemo-zarr"
    checkpoint_io.save_checkpoint(sharded_state_dict, output_checkpoint_path)
    print(f"Checkpoint saved to {output_checkpoint_path}")
    
    

if __name__ == '__main__':
    for k, v in os.environ.items():
        if 'SLURM' in k:
            del os.environ[k]
    
    args = get_args()
    local_rank, rank, world_size = initialize_distributed(args)
    convert(local_rank, rank, world_size, args)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


def setup_distribution():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def get_mixed_precision_policy():
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )


def wrap_flux_model(model, local_rank):
    # Define FSDP wrapping policy
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            # Add your transformer block classes here if any
        },
    )

    fsdp_config = dict(
        auto_wrap_policy=transformer_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=get_mixed_precision_policy(),
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True
    )

    return FSDP(model, **fsdp_config)


def main():
    # Initialize distributed setup
    local_rank, world_size = setup_distribution()
    config = TrainingConfig()

    # Device assignments
    encoder_device = torch.device("cuda:0")
    flux_device = torch.device(f"cuda:{local_rank}")

    # Initialize models with proper device placement
    if local_rank == 0:
        t5 = load_t5(encoder_device, max_length=128)
        clip = load_clip(encoder_device)
        ae = load_ae("flux-dev-fill", device=encoder_device)
    else:
        t5, clip, ae = None, None, None

    # Load Flux model
    name = "flux-dev-fill"
    if local_rank == 0:
        ckpt_path = configs[name].ckpt_path
        if (ckpt_path is None and configs[name].repo_id is not None
                and configs[name].repo_flow is not None):
            ckpt_path = hf_hub_download(
                configs[name].repo_id, configs[name].repo_flow)
    else:
        ckpt_path = None

    # Broadcast ckpt_path to all ranks
    if world_size > 1:
        object_list = [ckpt_path]
        dist.broadcast_object_list(object_list, src=0)
        ckpt_path = object_list[0]

    # Initialize Flux model
    model = Flux(params=configs[name].params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        if local_rank == 0:
            print_load_warning(missing, unexpected)

    # Apply LoRA modifications
    lora_rank = 16
    lora_scale = 1.0
    replace_linear_with_lora(
        model, lora_rank, lora_scale, recursive=False,
        keys_override=["single_blocks", "final_layer"],
        keys_ignore=["img_in", "txt_in"]
    )

    # Wrap model with FSDP
    model = wrap_flux_model(model, local_rank)

    # Setup dataset and dataloader with DistributedSampler
    dataset = FluxFillDataset(root="cube-dataset")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Initialize optimizer
    optimizer = bitsandbytes.optim.AdamW8bit(
        model.parameters(), lr=config.learning_rate
    )

    # Training loop
    for epoch in range(config.num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            img, mask, coords = batch
            img = img.to(flux_device)
            mask = mask.to(flux_device)

            # Prepare inputs on encoder_device (GPU 0)
            if local_rank == 0:
                inputs = prepare_fill(
                    t5=t5, clip=clip, ae=ae,
                    img=img, mask=mask,
                    prompt="a photo of sks"
                )
                # Move inputs to flux_device after preparation
                inputs = {k: v.to(flux_device) for k, v in inputs.items()}

            # Broadcast inputs to all ranks
            if world_size > 1:
                object_list = [inputs]
                dist.broadcast_object_list(object_list, src=0)
                inputs = object_list[0]

            # Convert inputs to bfloat16
            inputs = {k: v.to(torch.bfloat16) for k, v in inputs.items()}
            guids = torch.ones(inputs["img"].shape[0], dtype=torch.bfloat16,
                               device=flux_device)

            # Forward pass
            pred = model(
                img=torch.cat((inputs["img"], inputs["img_cond"]), dim=-1),
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                y=inputs["vec"],
                timesteps=inputs["t"],
                guidance=guids,
                homo_pos_map=None
            )

            pred = unpack(pred, 512, 512)
            loss = torch.pow(pred - inputs["vt"], 2).mean()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if local_rank == 0:
                print(f"Step loss: {loss.item():.4f}")
            epoch_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}, "
                  f"Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint on rank 0
        if local_rank == 0 and (epoch + 1) % config.save_every == 0:
            save_file(
                model.state_dict(),
                f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            )


if __name__ == "__main__":
    main()

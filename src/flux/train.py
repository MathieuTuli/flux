from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file as load_sft, save_file
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

import torch

from flux.util import (
    load_t5, load_clip, load_ae,
    configs, optionally_expand_state_dict,
    print_load_warning
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.gpu_split import split_flux_model_to_gpus, GPUSplitConfig
from flux.model import Flux
from flux.train_utils import (
    prepare_fill_for_train,
    OptimalTransportPath,
    FluxFillDataset
)


@dataclass
class TrainingConfig:
    outdir = "shapes-without-sphere-pe-many-epochs"
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_epochs: int = 5000
    save_every: int = 2500
    num_steps: int = 50
    guidance: float = 1.0
    seed: int = 420


def main():
    config = TrainingConfig()
    config.outdir = Path(config.outdir)
    config.outdir.mkdir(exist_ok=True, parents=True)
    offload = False
    device = torch.device("cuda:0")

    # Initialize models
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    ae = load_ae(
        "flux-dev-fill", device="cpu" if offload else device)

    name = "flux-dev-fill"
    hf_download = True
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(
            configs[name].repo_id, configs[name].repo_flow)

    model = Flux(
        params=configs[name].params)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(
            ckpt_path, device="cpu")
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(
            sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)

    lora_rank = 16
    lora_scale = 1.
    replace_linear_with_lora(model, lora_rank, lora_scale, recursive=True,
                             # keys_override=["single_blocks", "final_layer"],
                             # keys_ignore=["img_in", "txt_in", "final_layer"],
                             )

    model.requires_grad_(False)

    def set_requires_grad_recursive(module, name=''):
        if isinstance(module, LinearLora):
            module.requires_grad_(False)
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
        for child_name, child in module.named_children():
            child_full_name = f"{name}.{child_name}" if name else child_name
            set_requires_grad_recursive(child, child_full_name)

    set_requires_grad_recursive(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    gpu_config = GPUSplitConfig(
        gpu_ids=[1, 2, 3, 4, 5],  # List of GPU IDs to use
        max_params_per_gpu=5e9,  # Maximum parameters per GPU
        base_gpu=1  # GPU to place non-distributed components
    )
    model = split_flux_model_to_gpus(model, gpu_config)
    model = model.to()

    dataset = FluxFillDataset(root="datasets/shapes")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Training setup
    # optimizer = bitsandbytes.optim.AdamW8bit(
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate)

    path = OptimalTransportPath(sig_min=0)

    # Training loop
    grad_accum_steps = min(config.batch_size, len(dataset))
    print("Setting grad accum step:", grad_accum_steps)
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.
        for step, batch in enumerate(dataloader):

            if offload:
                t5, clip, ae = t5.to(device), clip.to(device), ae.to(device)
            img, mask, sphere_coords = batch
            img, mask = img.to(device), mask.to(device)
            inputs = prepare_fill_for_train(
                t5=t5, clip=clip, ae=ae,
                img=img, mask=mask,
                prompt="a photo of sks",
                path=path
            )
            guids = torch.ones(inputs["img"].shape[0], dtype=torch.bfloat16,
                               device=device)
            if offload:
                t5, clip, ae = t5.cpu(), clip.cpu(), ae.cpu()
                torch.cuda.empty_cache()
                model = model.to(device)

            for k in inputs.keys():
                inputs[k] = inputs[k].to(device)
                inputs[k] = inputs[k].to(torch.bfloat16)
            pred = model(
                img=torch.cat((inputs["img"], inputs["img_cond"]), dim=-1),
                img_ids=inputs["img_ids"],
                txt=inputs["txt"],
                txt_ids=inputs["txt_ids"],
                y=inputs["vec"],
                timesteps=inputs["t"],
                guidance=guids,
                sphere_coords=None # sphere_coords,
            ).to("cuda:0")
            # pred = unpack(pred, 512, 512).to("cuda:0")
            loss = torch.nn.functional.mse_loss(
                pred, inputs["vt"], reduction="mean")
            # loss += (loss * inputs["weights"] / grad_accum_steps)
            loss += loss / grad_accum_steps
            del inputs
            loss.backward()
            print(f"Step loss: {loss.item():.4f}")
            if ((step + 1) % grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()

            if offload:
                model.cpu()
                torch.cuda.empty_cache()
        print(
            f"Epoch {epoch+1}/{config.num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        if config.save_every > 0 and (epoch + 1) % config.save_every == 0:
            # torch.save(
            #     model.state_dict(),
            #     f"lora_checkpoint_epoch_{epoch+1}.safetensors"
            # )
            sd = model.state_dict()
            lora_dict = {k: sd[k] for k in sd.keys() if "lora_" in k}
            save_file(model.state_dict(), config.outdir /
                      f"lora_checkpoint_epoch_{epoch+1}.safetensors")
    save_file(model.state_dict(), config.outdir / "lora_last.safetensors")


if __name__ == "__main__":
    main()

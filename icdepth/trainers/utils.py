import os

import torch
from accelerate import Accelerator
from tqdm import tqdm


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def trainable_param_names(self):
        return {name for name, param in self.named_parameters() if param.requires_grad}

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict = {
                (name[len(remove_prefix):] if name.startswith(remove_prefix) else name): param
                for name, param in state_dict.items()
            }
        return state_dict


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, save_steps=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.save_steps = save_steps
        self.num_steps = 0

    def on_step_end(self, accelerator, model):
        self.num_steps += 1
        if self.save_steps is not None and self.num_steps % self.save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            accelerator.print(f"Saved {path}")


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 8,
):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            model_logger.on_step_end(accelerator, model)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        model_logger.on_epoch_end(accelerator, model, epoch_id)

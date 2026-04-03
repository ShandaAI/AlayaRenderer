import os, torch, csv, time
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Resume from checkpoint
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None) if args is not None else None
    start_epoch = 0
    global_step = 0
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        ckpt_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(ckpt_state_path):
            ckpt_state = torch.load(ckpt_state_path, map_location="cpu")
            optimizer.load_state_dict(ckpt_state["optimizer"])
            scheduler.load_state_dict(ckpt_state["scheduler"])
            start_epoch = ckpt_state["epoch"] + 1
            global_step = ckpt_state["global_step"]
            model_logger.num_steps = global_step
            if accelerator.is_main_process:
                print(f"Resumed from checkpoint: epoch={start_epoch}, global_step={global_step}")

    # Loss logging
    loss_log_path = os.path.join(model_logger.output_path, "loss_log.csv")
    if accelerator.is_main_process:
        os.makedirs(model_logger.output_path, exist_ok=True)
        append_mode = start_epoch > 0 and os.path.exists(loss_log_path)
        loss_log_file = open(loss_log_path, "a" if append_mode else "w", newline="")
        loss_writer = csv.writer(loss_log_file)
        if not append_mode:
            loss_writer.writerow(["epoch", "step", "global_step", "loss"])

    # wandb logging (optional)
    use_wandb = False
    if accelerator.is_main_process:
        try:
            import wandb
            if os.environ.get("WANDB_DISABLED", "").lower() == "true":
                pass
            elif wandb.run is not None:
                use_wandb = True
            else:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "diffsynth-training"),
                    name=os.environ.get("WANDB_NAME", None),
                    dir=os.environ.get("WANDB_DIR", None),
                    config=vars(args) if args is not None else {},
                )
                use_wandb = True
        except ImportError:
            pass

    t_data_start = time.time()
    for epoch_id in range(start_epoch, num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_id}")
        for step, data in enumerate(pbar):
            data_time = time.time() - t_data_start
            with accelerator.accumulate(model):
                t_train_start = time.time()
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()
                global_step += 1
                loss_val = loss.item()
                torch.cuda.synchronize()
                train_time = time.time() - t_train_start
                pbar.set_postfix(loss=f"{loss_val:.6f}", data=f"{data_time:.1f}s", train=f"{train_time:.1f}s")
                if accelerator.is_main_process:
                    loss_writer.writerow([epoch_id, step, global_step, loss_val])
                    loss_log_file.flush()
                    if use_wandb:
                        wandb.log({"loss": loss_val, "epoch": epoch_id, "lr": scheduler.get_last_lr()[0],
                                   "data_time": data_time, "train_time": train_time}, step=global_step)
            t_data_start = time.time()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        # Save training state for resuming
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(model_logger.output_path, "training_state")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch_id,
                "global_step": global_step,
            }, os.path.join(ckpt_dir, "training_state.pt"))

    if accelerator.is_main_process:
        loss_log_file.close()
        if use_wandb:
            wandb.finish()
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)

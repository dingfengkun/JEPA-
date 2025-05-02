import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import argparse
import numpy as np
from datetime import datetime

from models import HJEPA
from dataset import create_wall_dataloader
from schedulers import Scheduler, LRSchedule
from evaluator import ProbingEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train H-JEPA model for wall environment")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay rate for target encoders")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Output embedding dimension")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate model every n epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda/cpu)")
    parser.add_argument("--data_path", type=str, default=".", help="Path to dataset")
    parser.add_argument("--schedule", type=str, default="Cosine", help="LR schedule (Constant/Cosine)")
    parser.add_argument("--quick_debug", action="store_true", help="Quick debug mode (fewer iterations)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")
    
    return parser.parse_args()


def get_device(args):
    """Check for GPU availability."""
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(args, device):
    """Load training and evaluation data."""
    data_path = args.data_path
    
    # Training dataset
    train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        batch_size=args.batch_size,
        train=True,
    )
    
    # Probing datasets
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        batch_size=args.batch_size,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        batch_size=args.batch_size,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        batch_size=args.batch_size,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }
    
    return train_ds, probe_train_ds, probe_val_ds


def initialize_model(args, device):
    """Initialize the H-JEPA model."""
    model = HJEPA(
        context_encoder_dim=args.hidden_dim,
        target_encoder_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        action_dim=2,
        device=device,
        ema_decay=args.ema_decay
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        #'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler': scheduler,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """Load model checkpoint (assuming only model state dict was saved)."""
    try:
        # --- Load the checkpoint ---
        checkpoint = torch.load(load_path, map_location=device)
        
        # --- Load model state ---
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # --- Get epoch ---
        start_epoch = checkpoint['epoch'] + 1
        
        # --- Optimizer and scheduler states are NOT loaded ---
        print(f"Loaded model state dict from epoch {checkpoint['epoch']}. Optimizer and scheduler will be re-initialized.")
        
        # Return None for scheduler to signal re-initialization needed
        return model, optimizer, start_epoch, None 
        
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {load_path}")
        # Decide how to handle this - exit, or start from scratch?
        raise 
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise


def train(args):
    """Main training function."""
    # Setup device and seed
    device = get_device(args)
    torch.manual_seed(42)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"hjepa_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Tensorboard logs will be saved to {log_dir}")
    
    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    train_ds, probe_train_ds, probe_val_ds = load_data(args, device)
    
    # Initialize model
    model = initialize_model(args, device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Setup learning rate scheduler
    schedule = LRSchedule.Cosine if args.schedule == "Cosine" else LRSchedule.Constant
    scheduler = Scheduler(
        schedule=schedule,
        base_lr=args.lr,
        data_loader=train_ds,
        epochs=args.epochs,
        optimizer=optimizer,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        model, optimizer, start_epoch, loaded_scheduler = load_checkpoint(
            model, optimizer, args.resume_from, device
        )
        if loaded_scheduler:
            scheduler = loaded_scheduler
    
    # Training loop
    global_step = start_epoch * len(train_ds)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {
            'total_loss': [],
            'short_term_loss': [],
            'long_term_loss': [],
            'reg_loss': [],
            'agent_loss': [],
            'wall_loss': [],
        }
        
        progress_bar = tqdm(train_ds, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Get batch data
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, 2]
            
            # Training step
            losses = model.training_step(states, actions, optimizer)
            
            # Update learning rate
            lr = scheduler.adjust_learning_rate(global_step)
            
            # Log losses
            for loss_name, loss_value in losses.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name].append(loss_value)
                    writer.add_scalar(f"train/{loss_name}", loss_value, global_step)
                else:
                    tqdm.write(f"Warning: Unexpected loss key '{loss_name}' returned from training_step.")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total_loss'],
                'lr': lr
            })
            
            global_step += 1
            
            # Debug mode: run only a few steps
            if args.quick_debug and step >= 5:
                break
        
        # Log epoch metrics
        for loss_name, loss_values in epoch_losses.items():
            if loss_values:
                avg_loss = np.mean(loss_values)
                print(f"Epoch {epoch+1} {loss_name}: {avg_loss:.4f}")
                writer.add_scalar(f"epoch/{loss_name}", avg_loss, epoch)
            else:
                print(f"Epoch {epoch+1} {loss_name}: No data collected")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"hjepa_epoch_{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        
        # --- Evaluation Phase ---
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            print(f"\nEvaluating model at epoch {epoch+1}...")
            model.eval() # Set main model to eval mode (important!)
            
            # --- Step 1: Train the prober (Gradients NEEDED for prober) ---
            # Initialize evaluator here or ensure it's accessible
            evaluator = ProbingEvaluator(
                device=device,
                model=model, # Pass the main model in eval mode
                probe_train_ds=probe_train_ds,
                probe_val_ds=probe_val_ds,
                quick_debug=args.quick_debug,
            )
            print("Training prober...")
            # Ensure this function doesn't internally use torch.no_grad() wrongly
            prober = evaluator.train_pred_prober() 
                                             
            # --- Step 2: Evaluate using the trained prober (Gradients NOT needed) ---
            print("Evaluating prober...")
            with torch.no_grad(): # Use no_grad ONLY for the final evaluation
                avg_losses = evaluator.evaluate_all(prober=prober)
            
            # Log evaluation metrics
            for probe_attr, loss in avg_losses.items():
                writer.add_scalar(f"eval/{probe_attr}_loss", loss, epoch)
                print(f"Eval - {probe_attr} loss: {loss:.4f}") # Print eval results
            
            avg_eval_loss = sum(avg_losses.values()) / len(avg_losses)
            writer.add_scalar(f"eval/average_loss", avg_eval_loss, epoch)
            print(f"Eval - Average loss: {avg_eval_loss:.4f}")

            # Set model back to train mode if continuing training
            # model.train() # Already handled at the start of the next epoch loop

    # Final evaluation (similar structure)
    print("\nFinal evaluation...")
    model.eval()
    # Re-initialize evaluator for final run? Or reuse if state doesn't matter.
    final_evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=args.quick_debug,
    )
    print("Training final prober...")
    final_prober = final_evaluator.train_pred_prober()
    print("Evaluating final prober...")
    with torch.no_grad():
        final_avg_losses = final_evaluator.evaluate_all(prober=final_prober)
    for probe_attr, loss in final_avg_losses.items():
         print(f"Final Eval - {probe_attr} loss: {loss:.4f}")

    # Save final model
    final_path = os.path.join(args.save_dir, "hjepa_final.pt")
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, final_path)
    
    print(f"Training completed. Final model saved to {final_path}")
    writer.close()
    
    return model


if __name__ == "__main__":
    args = parse_args()
    train(args)

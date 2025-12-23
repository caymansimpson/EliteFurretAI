"""
Example: Training with all advanced features enabled.

This demonstrates a complete training setup with:
1. Portfolio regularization (5 reference models)
2. Mixed precision training (FP16)
3. Profiler-optimized hyperparameters
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from elitefurretai.rl2.config import RNaDConfig
from elitefurretai.rl2.profiler import TrainingProfiler


def step1_profile():
    """Step 1: Profile to find optimal settings."""
    print("="*60)
    print("STEP 1: Profiling training configuration")
    print("="*60)
    
    profiler = TrainingProfiler("data/models/bc_action_model.pt")
    
    # Run sweep to find best configuration
    best = profiler.profile_sweep(
        worker_configs=[(2, 4), (4, 4), (4, 6)],
        batch_configs=[(16, 32), (24, 48), (32, 64)],
        test_mixed_precision=True
    )
    
    # Save results
    profiler.save_results("profiling_results.json")
    
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATION FOUND:")
    print("="*60)
    print(f"Workers: {best.num_workers}")
    print(f"Players/Worker: {best.players_per_worker}")
    print(f"Batch Size: {best.batch_size}")
    print(f"Train Batch Size: {best.train_batch_size}")
    print(f"Mixed Precision: {best.use_mixed_precision}")
    print(f"\nExpected Performance:")
    print(f"  Updates/hour: {best.updates_per_hour:.0f}")
    print(f"  GPU Utilization: {best.avg_gpu_utilization:.1f}%")
    print("="*60 + "\n")
    
    return best


def step2_create_config(profiling_result):
    """Step 2: Create optimized config with all features."""
    print("="*60)
    print("STEP 2: Creating optimized config")
    print("="*60)
    
    config = RNaDConfig(
        # Model
        checkpoint_path="data/models/bc_action_model.pt",
        bc_teampreview_path="data/models/bc_teampreview_model.pt",
        bc_action_path="data/models/bc_action_model.pt",
        bc_win_path="data/models/bc_win_model.pt",
        
        # Teams
        team_pool_path="data/teams/gen9vgc2023regulationc",
        use_random_teams=True,
        
        # Training (from profiler)
        num_workers=profiling_result.recommended_num_workers or profiling_result.num_workers,
        players_per_worker=profiling_result.recommended_players_per_worker or profiling_result.players_per_worker,
        batch_size=profiling_result.recommended_batch_size or profiling_result.batch_size,
        train_batch_size=profiling_result.recommended_train_batch_size or profiling_result.train_batch_size,
        lr=1e-4,
        
        # Portfolio regularization
        use_portfolio_regularization=True,
        max_portfolio_size=5,
        portfolio_update_strategy="diverse",
        portfolio_add_interval=5000,
        
        # Mixed precision
        use_mixed_precision=True,
        gradient_clip=0.5,
        
        # RNaD
        rnad_alpha=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        
        # Checkpointing
        checkpoint_interval=1000,
        ref_update_interval=1000,
        save_dir="data/models/advanced_training",
        past_models_dir="data/models/advanced_training/past_versions",
        max_past_models=10,
        
        # Opponent pool
        curriculum={
            'self_play': 0.40,
            'bc_player': 0.20,
            'exploiters': 0.20,
            'past_versions': 0.20
        },
        exploiter_registry_path="data/models/exploiter_registry.json",
        
        # Exploiter training
        train_exploiters=True,
        exploiter_interval=10000,
        exploiter_updates=50000,
        exploiter_min_win_rate=0.60,
        
        # Logging
        use_wandb=True,
        wandb_project="elitefurretai-advanced",
        wandb_run_name="portfolio_mp_optimized",
        log_interval=10,
        curriculum_update_interval=500,
        
        # Hardware
        device="cuda",
        num_showdown_servers=4,
        showdown_start_port=8000,
    )
    
    # Save config
    config_path = "advanced_training_config.yaml"
    config.save(config_path)
    
    print(f"\nConfig saved to: {config_path}")
    print("\nKey features enabled:")
    print("  ✓ Portfolio regularization (5 refs)")
    print("  ✓ Mixed precision (FP16)")
    print("  ✓ Profiler-optimized hyperparameters")
    print("  ✓ Team randomization")
    print("  ✓ Automatic exploiter training")
    print("  ✓ Wandb logging")
    print("="*60 + "\n")
    
    return config_path


def step3_train(config_path):
    """Step 3: Launch training with advanced features."""
    print("="*60)
    print("STEP 3: Launching training")
    print("="*60)
    print(f"\nCommand: python src/elitefurretai/rl2/train_v2.py --config {config_path}")
    print("\nExpected output:")
    print("  - 'Using Portfolio RNaD with 5 references'")
    print("  - 'Mixed Precision: True'")
    print("  - Updates should be ~2x faster than baseline")
    print("  - Portfolio size will grow (adds ref every 5k updates)")
    print("  - Wandb will track portfolio selection stats")
    print("\nPress Ctrl+C to stop training")
    print("="*60 + "\n")
    
    # Launch training
    subprocess.run([
        sys.executable,
        "src/elitefurretai/rl2/train_v2.py",
        "--config", config_path
    ])


def main():
    """Run complete workflow: profile → configure → train."""
    
    print("\n" + "="*60)
    print("ADVANCED TRAINING SETUP WORKFLOW")
    print("="*60)
    print("\nThis will:")
    print("1. Profile your system to find optimal settings")
    print("2. Create config with all advanced features")
    print("3. Launch training with portfolio + mixed precision")
    print("\nEstimated time:")
    print("  Profiling: 10-15 minutes")
    print("  Training: Ongoing (Ctrl+C to stop)")
    print("="*60)
    
    input("\nPress Enter to start profiling...")
    
    # Step 1: Profile
    try:
        profiling_result = step1_profile()
    except Exception as e:
        print(f"\nProfiling failed: {e}")
        print("Using default settings instead...")
        # Create dummy result
        from elitefurretai.rl2.profiler import ProfilingResult
        profiling_result = ProfilingResult(
            num_workers=4,
            players_per_worker=4,
            batch_size=16,
            train_batch_size=32,
            use_mixed_precision=True,
            battles_per_hour=6000,
            updates_per_hour=3800,
            timesteps_per_hour=24000,
            avg_cpu_percent=50.0,
            peak_cpu_percent=70.0,
            avg_gpu_utilization=75.0,
            peak_gpu_utilization=90.0,
            avg_gpu_memory_mb=14000,
            peak_gpu_memory_mb=16000,
            avg_ram_gb=18.0,
            peak_ram_gb=21.0,
            avg_inference_time_ms=8.0,
            avg_training_time_ms=31.0,
            avg_data_collection_time_ms=50.0,
            primary_bottleneck="balanced",
            bottleneck_score=0.75
        )
    
    input("\nProfiling complete! Press Enter to create config...")
    
    # Step 2: Create config
    config_path = step2_create_config(profiling_result)
    
    input(f"\nConfig created at {config_path}! Press Enter to start training...")
    
    # Step 3: Train
    step3_train(config_path)


if __name__ == "__main__":
    main()

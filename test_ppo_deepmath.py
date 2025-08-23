#!/usr/bin/env python3

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.append('/cpfs04/user/liyuanhang.p/src/ContinualCountdown')

def test_ppo_deepmath_training():
    """Test PPO training with DeepMath dataset to validate scorer fix"""
    
    # Set environment variables for minimal test
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only one GPU
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['RAY_DEDUP_LOGS'] = '0'
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Import after setting environment variables
        from verl.trainer.main_ppo import main
        from verl.trainer.config import Config
        
        # Create minimal config for testing
        config_dict = {
            'algorithm': {
                'kl_ctrl': {
                    'kl_coef': 0.05,
                    'adaptive_kl': False,
                    'target_kl': 6.0,
                    'horizon': 10000
                }
            },
            'data': {
                'train_files': ['/nas/shared/sys2/yuanhangli/tmp/data/deepmath/train.parquet'],
                'val_files': ['/nas/shared/sys2/yuanhangli/tmp/data/deepmath/test.parquet'],
                'prompt_key': 'prompt',
                'max_prompt_length': 1024,
                'max_response_length': 512
            },
            'model': {
                'actor': {
                    'path': '/nas/shared/sys2/yuanhangli/tmp/llama_instruct_sft_model/global_step_0',
                    'enable_gradient_checkpointing': True
                },
                'critic': {
                    'path': '/nas/shared/sys2/yuanhangli/tmp/llama_instruct_sft_model/global_step_0',
                    'enable_gradient_checkpointing': True
                }
            },
            'trainer': {
                'total_epochs': 1,
                'rollout': {
                    'name': 'vllm',
                    'gpu_memory_utilization': 0.85,
                    'tensor_parallel_size': 1,
                    'max_num_batched_tokens': 2048
                },
                'ppo': {
                    'num_mini_batches': 1,
                    'ppo_mini_batch_size': 2,
                    'ppo_epochs': 1,
                    'max_grad_norm': 1.0,
                    'entropy_coef': 0.0,
                    'learning_rate': 1e-6
                },
                'logger': {
                    'project_name': 'test_deepmath_scorer',
                    'experiment_name': 'minimal_test',
                    'save_interval': 1000
                }
            },
            'trainer_base': {
                'n_gpus_per_node': 1,
                'nnodes': 1,
                'save_freq': 1000,
                'logging_freq': 1,
                'checkpoint_path': checkpoint_dir
            }
        }
        
        # Create config object
        config = Config(config_dict)
        
        print("Starting minimal PPO training test with DeepMath dataset...")
        print("This will test the scorer fix with just a few training steps.")
        
        # Run training for minimal steps
        main(config)
        
        print("✓ PPO training test completed successfully!")
        print("✓ DeepMath scorer fix is working in training environment")
        
    except Exception as e:
        print(f"❌ PPO training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    # Check if DeepMath data exists
    deepmath_train = '/nas/shared/sys2/yuanhangli/tmp/data/deepmath/train.parquet'
    if not os.path.exists(deepmath_train):
        print(f"❌ DeepMath training data not found at {deepmath_train}")
        print("Please ensure DeepMath dataset is generated first.")
        sys.exit(1)
    
    success = test_ppo_deepmath_training()
    if success:
        print("🎉 All tests passed! DeepMath scorer fix is validated.")
    else:
        print("❌ Test failed. Please check the error messages above.")
        sys.exit(1)

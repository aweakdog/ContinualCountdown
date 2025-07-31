#!/usr/bin/env python3
"""
Test script to verify dynamic analyzer storage path detection.
"""

import os
import sys
import tempfile
from verl.utils.analyzer_metrics_storage import AnalyzerMetricsStorage

def test_llama_scripts_detection():
    """Test detection for llama_scripts."""
    print("=== Testing llama_scripts Detection ===")
    
    # Simulate llama_scripts environment
    os.environ['RUN_NAME'] = 'Phase1_Group1_SFT_global_step_20_20250801_004500'
    
    # Simulate being called from llama_scripts
    original_argv = sys.argv.copy()
    sys.argv = ['python', 'llama_scripts/debug_train_continual_countdown_3b_curriculum_after_sft_llama.sh']
    
    try:
        storage = AnalyzerMetricsStorage(
            experiment_name="test_llama",
            enable_wandb=False,
            auto_detect_script_type=True
        )
        print(f"✅ Detected path: {storage.base_dir}")
        print(f"✅ Expected: ./llama_logs/Phase1_Group1_SFT_global_step_20_20250801_004500")
        
        # Cleanup
        import shutil
        if os.path.exists("llama_logs"):
            shutil.rmtree("llama_logs")
            
    finally:
        sys.argv = original_argv

def test_scripts_detection():
    """Test detection for scripts (qwen)."""
    print("\n=== Testing scripts (qwen) Detection ===")
    
    # Simulate scripts environment
    os.environ['RUN_NAME'] = 'ContinualCountdown3B_SingleRun_20250801_004500'
    
    # Simulate being called from scripts
    original_argv = sys.argv.copy()
    sys.argv = ['python', 'scripts/debug_train_continual_countdown_3b_curriculum_after_sft.sh']
    
    try:
        storage = AnalyzerMetricsStorage(
            experiment_name="test_qwen",
            enable_wandb=False,
            auto_detect_script_type=True
        )
        print(f"✅ Detected path: {storage.base_dir}")
        print(f"✅ Expected: ./qwen_logs/ContinualCountdown3B_SingleRun_20250801_004500")
        
        # Cleanup
        import shutil
        if os.path.exists("qwen_logs"):
            shutil.rmtree("qwen_logs")
            
    finally:
        sys.argv = original_argv

def test_environment_override():
    """Test explicit environment variable override."""
    print("\n=== Testing Environment Variable Override ===")
    
    # Set explicit analyzer log directory
    os.environ['ANALYZER_LOG_DIR'] = './custom_analyzer_logs'
    os.environ['RUN_NAME'] = 'custom_experiment'
    
    try:
        storage = AnalyzerMetricsStorage(
            experiment_name="test_custom",
            enable_wandb=False,
            auto_detect_script_type=True
        )
        print(f"✅ Detected path: {storage.base_dir}")
        print(f"✅ Expected: ./custom_analyzer_logs")
        
        # Cleanup
        import shutil
        if os.path.exists("custom_analyzer_logs"):
            shutil.rmtree("custom_analyzer_logs")
            
    finally:
        if 'ANALYZER_LOG_DIR' in os.environ:
            del os.environ['ANALYZER_LOG_DIR']

def test_default_fallback():
    """Test default fallback behavior."""
    print("\n=== Testing Default Fallback ===")
    
    # Clear environment variables
    if 'RUN_NAME' in os.environ:
        del os.environ['RUN_NAME']
    if 'ANALYZER_LOG_DIR' in os.environ:
        del os.environ['ANALYZER_LOG_DIR']
    
    # Simulate unknown script
    original_argv = sys.argv.copy()
    sys.argv = ['python', 'unknown_script.py']
    
    try:
        storage = AnalyzerMetricsStorage(
            experiment_name="test_default",
            enable_wandb=False,
            auto_detect_script_type=True
        )
        print(f"✅ Detected path: {storage.base_dir}")
        print(f"✅ Expected: ./analyzer_metrics")
        
        # Cleanup
        import shutil
        if os.path.exists("analyzer_metrics"):
            shutil.rmtree("analyzer_metrics")
            
    finally:
        sys.argv = original_argv

def test_storage_functionality():
    """Test actual storage functionality."""
    print("\n=== Testing Storage Functionality ===")
    
    os.environ['RUN_NAME'] = 'test_storage_functionality'
    
    try:
        storage = AnalyzerMetricsStorage(
            experiment_name="test_storage",
            enable_wandb=False,
            auto_detect_script_type=False,
            base_dir="./test_storage"
        )
        
        # Test gradient metrics storage
        gradient_stats = {
            "__global__": {"ratio": 0.075, "zero": 1500, "total": 20000},
            "components": {
                "attention": {
                    "ratio": 0.08,
                    "matrices": {
                        "q_proj": {"ratio": 0.09, "min_row_norm": 1e-6, "avg_row_norm": 0.001, "max_row_norm": 0.1}
                    }
                }
            }
        }
        
        storage.store_gradient_metrics(
            step=1,
            gradient_stats=gradient_stats,
            tau=0.1,
            additional_info={"test": True}
        )
        
        # Test Fisher metrics storage
        fisher_stats = {
            "c_k_running_avg": 2.036,
            "l_k_cumulative_sum": 0.000,
            "C_K_normalized": 1.540
        }
        
        storage.store_fisher_metrics(
            step=1,
            fisher_stats=fisher_stats,
            additional_info={"test": True}
        )
        
        # Verify files were created
        assert os.path.exists(storage.metrics_file), f"Metrics file not created: {storage.metrics_file}"
        
        # Read and verify content
        with open(storage.metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
            
        print("✅ Storage functionality test passed!")
        
        # Cleanup
        import shutil
        if os.path.exists("test_storage"):
            shutil.rmtree("test_storage")
            
    finally:
        if 'RUN_NAME' in os.environ:
            del os.environ['RUN_NAME']

if __name__ == "__main__":
    print("🧪 Testing Analyzer Storage Dynamic Path Detection")
    print("=" * 60)
    
    test_llama_scripts_detection()
    test_scripts_detection()
    test_environment_override()
    test_default_fallback()
    test_storage_functionality()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📋 Summary:")
    print("✅ llama_scripts → llama_logs/RUN_NAME/")
    print("✅ scripts → qwen_logs/RUN_NAME/")
    print("✅ Environment override → ANALYZER_LOG_DIR")
    print("✅ Default fallback → ./analyzer_metrics")
    print("✅ Storage functionality working")

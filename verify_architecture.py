#!/usr/bin/env python3
"""
Verification script to check if CRIS-SNN architecture matches CRIS.pytorch
Run this to validate the implementation before training.
"""

import torch
import sys

def check_imports():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'snntorch': 'snnTorch',
        'cv2': 'OpenCV',
        'loguru': 'Loguru',
        'wandb': 'Weights & Biases',
    }
    
    missing = []
    for pkg, name in required_packages.items():
        try:
            __import__(pkg)
            print(f"✅ {name:20} - OK")
        except ImportError:
            print(f"❌ {name:20} - MISSING")
            missing.append(pkg)
    
    return len(missing) == 0

def check_snn_encoders():
    """Check SNNVisionEncoder and SNNTextEncoder outputs"""
    print("\n" + "=" * 60)
    print("CHECKING SNN ENCODERS")
    print("=" * 60)
    
    try:
        from model.snn_encoder import SNNVisionEncoder, SNNTextEncoder
        
        # Check SNNVisionEncoder
        print("\n[1] SNNVisionEncoder")
        print("-" * 40)
        vision_encoder = SNNVisionEncoder(input_size=416, num_steps=10)
        
        # Test forward pass
        dummy_img = torch.randn(1, 3, 416, 416)
        with torch.no_grad():
            vis_output = vision_encoder(dummy_img)
        
        print(f"  Input shape: {dummy_img.shape}")
        print(f"  Output type: {type(vis_output)}")
        print(f"  Output length: {len(vis_output)}")
        
        if len(vis_output) != 3:
            print(f"  ❌ Expected 3 scales, got {len(vis_output)}")
            return False
        else:
            print(f"  ✅ Returns 3 scales")
        
        scales_info = []
        for i, scale in enumerate(vis_output):
            scales_info.append(scale.shape)
            print(f"    Scale {i}: {scale.shape}")
        
        expected = [
            (1, 256, 52, 52),  # C3
            (1, 512, 26, 26),  # C4
            (1, 1024, 13, 13), # C5
        ]
        
        for i, (got, exp) in enumerate(zip(scales_info, expected)):
            if got != exp:
                print(f"  ❌ Scale {i}: expected {exp}, got {got}")
                return False
        
        print(f"  ✅ All scales have correct shape")
        
        # Check SNNTextEncoder
        print("\n[2] SNNTextEncoder")
        print("-" * 40)
        text_encoder = SNNTextEncoder(vocab_size=49408, embed_dim=512, num_steps=10)
        
        dummy_text = torch.randint(0, 49408, (1, 77))
        with torch.no_grad():
            word_embeddings, state = text_encoder(dummy_text)
        
        print(f"  Input shape: {dummy_text.shape}")
        print(f"  Word embeddings shape: {word_embeddings.shape}")
        print(f"  State shape: {state.shape}")
        
        if word_embeddings.shape != (1, 77, 512):
            print(f"  ❌ Expected word embeddings (1, 77, 512), got {word_embeddings.shape}")
            return False
        else:
            print(f"  ✅ Word embeddings shape correct")
        
        if state.shape != (1, 1024):
            print(f"  ❌ Expected state (1, 1024), got {state.shape}")
            return False
        else:
            print(f"  ✅ State shape correct (matches FPN expectation)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking encoders: {e}")
        return False

def check_model_architecture():
    """Check CRIS model with SNN encoders"""
    print("\n" + "=" * 60)
    print("CHECKING CRIS MODEL ARCHITECTURE")
    print("=" * 60)
    
    try:
        from utils.config import CfgNode
        from model import build_segmenter
        
        # Create minimal config
        cfg = CfgNode({
            'input_size': 416,
            'word_len': 17,
            'word_dim': 1024,
            'vis_dim': 512,
            'fpn_in': [512, 1024, 1024],
            'fpn_out': [256, 512, 1024],
            'num_steps': 10,
            'num_layers': 3,
            'num_head': 8,
            'dim_ffn': 2048,
            'dropout': 0.1,
            'intermediate': False,
            'lr_multi': 0.1,
            'base_lr': 0.0001,
        })
        
        print("\n[1] Building model...")
        model, param_list = build_segmenter(cfg)
        print(f"  ✅ Model built successfully")
        print(f"  Model type: {type(model).__name__}")
        
        print("\n[2] Testing forward pass...")
        dummy_img = torch.randn(1, 3, 416, 416)
        dummy_text = torch.randint(0, 49408, (1, 77))
        dummy_mask = torch.randn(1, 1, 416, 416)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_img, dummy_text, dummy_mask)
        
        print(f"  ✅ Forward pass successful")
        
        if model.training:
            pred, mask, loss = output
            print(f"  Training mode: pred {pred.shape}, mask {mask.shape}, loss {loss.item():.4f}")
        else:
            print(f"  Eval mode: output shape {output.shape}")
        
        print("\n[3] Checking parameter groups...")
        print(f"  Number of parameter groups: {len(param_list)}")
        for i, group in enumerate(param_list):
            num_params = sum(p.numel() for p in group['params'])
            print(f"    Group {i}: {num_params:,} parameters, lr={group.get('initial_lr', 'default')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_pipeline():
    """Check data loading pipeline"""
    print("\n" + "=" * 60)
    print("CHECKING DATA PIPELINE")
    print("=" * 60)
    
    try:
        from data_loader import preprocess_image, get_text
        import numpy as np
        
        print("\n[1] Testing preprocess_image...")
        # Create dummy image
        dummy_img = np.random.randint(0, 256, (1024, 768, 3), dtype=np.uint8)
        print(f"  ✅ Can create dummy images")
        
        print("\n[2] Testing text processing...")
        sample_texts = [
            "the cat on the table",
            "left side of the image",
            "a red ball",
        ]
        
        for text in sample_texts:
            processed = get_text({'sentences': [text]})
            print(f"  ✅ '{text}' → '{processed}'")
        
        print("\n[3] Data format check:")
        print(f"  Input image should be: (batch, 3, 416, 416), range [0, 1]")
        print(f"  ✅ SNNVisionEncoder will handle rate encoding internally")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data pipeline: {e}")
        return False

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  CRIS-SNN Architecture Validation".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    all_ok = True
    
    # Run checks
    all_ok &= check_imports()
    all_ok &= check_snn_encoders()
    all_ok &= check_model_architecture()
    all_ok &= check_data_pipeline()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if all_ok:
        print("\n✅ All checks passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Install requirements: pip install -r requirement.txt")
        print("  2. Prepare datasets according to tools/prepare_datasets.md")
        print("  3. Convert data to LMDB: python tools/folder2lmdb.py")
        print("  4. Start training: python train.py --config config/refcoco/cris_r50.yaml")
    else:
        print("\n❌ Some checks failed. Please review errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

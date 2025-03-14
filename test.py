import os
import torch
import numpy as np
import argparse
from module.m2bnet import m2bnet
from data_loader import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Test script for generating confidence and detecting beats")
    parser.add_argument('--checkpoint', required=True, help="Checkpoint path to load the model from")
    parser.add_argument('--test', required=True, help="Test file path")
    parser.add_argument('--test_data', required=True, help="Test data dir")
    parser.add_argument('--output_dir', default="results/test", help="Output directory to save the test results")
    return parser.parse_args()

def get_model(checkpoint_path):
    net = m2bnet()
    if torch.cuda.is_available():
        net.cuda()
    print(f"[*] Loading model from: {checkpoint_path}")
    net.load_state_dict(torch.load(checkpoint_path))
    return net

def test(model, test_loader, output_dir):
    model.eval()

    with torch.no_grad():
        # Get the first batch from the test loader
        batch_data = next(iter(test_loader))
        batch_keypoints, _, batch_beat, batch_mask = batch_data
        batch_keypoints, batch_beat, batch_mask = batch_keypoints.float(), batch_beat.float(), batch_mask.long()

        if torch.cuda.is_available():
            batch_keypoints, batch_beat, batch_mask = batch_keypoints.cuda(), batch_beat.cuda(), batch_mask.cuda()

        # Get the predictions
        predictions = model(batch_keypoints)

        # For testing, just take the first sample in the batch
        pred_beat = predictions[0, :, 0].cpu().numpy()

        # Apply threshold to get rhythm beats (0.6 threshold)
        rhythm_beats = (pred_beat >= 0.6).astype(int)

        # Print the rhythm beats prediction (frame-level)
        print("Predicted Rhythm Beats (Frame-level):")
        print(rhythm_beats)

def main():
    args = parse_args()
    
    # Setup
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    model = get_model(args.checkpoint)

    # Create data loader
    test_loader = get_dataloaders(args.test, args.test_data, batch_size=1)  # Batch size 1 for testing single example

    # Perform testing
    test(model, test_loader, args.output_dir)

if __name__ == '__main__':
    main()

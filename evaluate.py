import os
import torch
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt
from module.m2bnet import m2bnet
from metrics_with_window import evaluate_motion_rhythm
from data_loader import get_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for checkpoint")
    parser.add_argument('--checkpoint', required=True, help="Checkpoint path to load the model from")
    parser.add_argument('--test',  required=True, help="Test file path")
    parser.add_argument('--test_data', required=True, help="Test data dir")
    parser.add_argument('--output_dir',  default="results/eval", help="Output directory to save evaluation results")
    return parser.parse_args()

def get_model(checkpoint_path):
    net = m2bnet()
    if torch.cuda.is_available():
        net.cuda()
    print(f"[*] Loading model from: {checkpoint_path}")
    net.load_state_dict(torch.load(checkpoint_path))
    return net

def evaluate(model, test_loader, output_dir):
    model.eval()
    total_predictions, total_targets = [], []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_keypoints, _, batch_beat, batch_mask = batch_data
            batch_keypoints, batch_beat, batch_mask = batch_keypoints.float(), batch_beat.float(), batch_mask.long()

            if torch.cuda.is_available():
                batch_keypoints, batch_beat, batch_mask = batch_keypoints.cuda(), batch_beat.cuda(), batch_mask.cuda()

            out = model(batch_keypoints)

            total_predictions.append(out.cpu().numpy())
            total_targets.append(batch_beat.cpu().numpy())

    total_predictions = np.concatenate(total_predictions, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    
    accuracy, recall, f1_score, pr_auc = evaluate_motion_rhythm(total_targets[:,:,0], total_predictions[:,:,0], save_path=os.path.join(output_dir, 'pr_curve.png'))
    print(f"Precision = {accuracy:.3f}, Recall = {recall:.3f}, F1 Score = {f1_score:.3f}, pr_auc = {pr_auc:.3f}")

def main():
    args = parse_args()
    
    # Setup
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    model = get_model(args.checkpoint)

    # Create data loader
    test_loader = get_dataloaders(args.test, args.test_data, batch_size=32)

    # Perform evaluation
    evaluate(model, test_loader, args.output_dir)

if __name__ == '__main__':
    main()

"""
Debug script for temporal perception training
1. Inspect V5 sequential dataset structure
2. Visualize V6 frame-to-frame transforms
3. Verify coordinate transform correctness
"""

import os
import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- CONFIGURATION ---
MAP_SIZE = 40
MAP_RES = 0.05


def euler_from_quat(quat):
    """Extract yaw from quaternion (w, x, y, z format)."""
    if len(quat.shape) == 1:
        quat = quat.unsqueeze(0)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def transform_height_map(prev_map, prev_pos, prev_yaw, curr_pos, curr_yaw):
    """Transform previous height map from prev robot frame to current robot frame."""
    B, _, H, W = prev_map.shape
    device = prev_map.device
    
    delta_pos_world = curr_pos[:, :2] - prev_pos[:, :2]
    delta_yaw = curr_yaw - prev_yaw
    
    cos_prev = torch.cos(-prev_yaw)
    sin_prev = torch.sin(-prev_yaw)
    delta_x_robot = delta_pos_world[:, 0] * cos_prev - delta_pos_world[:, 1] * sin_prev
    delta_y_robot = delta_pos_world[:, 0] * sin_prev + delta_pos_world[:, 1] * cos_prev
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    half_size = MAP_SIZE * MAP_RES / 2.0
    grid_meters = grid * half_size
    
    cos_delta = torch.cos(-delta_yaw).view(B, 1, 1, 1)
    sin_delta = torch.sin(-delta_yaw).view(B, 1, 1, 1)
    
    grid_x_rot = grid_meters[..., 0] * cos_delta.squeeze(-1) - grid_meters[..., 1] * sin_delta.squeeze(-1)
    grid_y_rot = grid_meters[..., 0] * sin_delta.squeeze(-1) + grid_meters[..., 1] * cos_delta.squeeze(-1)
    
    grid_x_trans = grid_x_rot - delta_x_robot.view(B, 1, 1)
    grid_y_trans = grid_y_rot - delta_y_robot.view(B, 1, 1)
    
    grid_transformed = torch.stack([
        grid_x_trans / half_size,
        grid_y_trans / half_size
    ], dim=-1)
    
    transformed_map = F.grid_sample(
        prev_map, grid_transformed,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )
    
    return transformed_map, delta_x_robot, delta_y_robot, delta_yaw


def inspect_dataset(h5_path, seq_len=16, stride=8):
    """Inspect dataset structure and sequence creation."""
    print(f"\n{'='*60}")
    print(f"DATASET INSPECTION: {h5_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(h5_path, 'r') as f:
        episodes = list(f.keys())
        print(f"Number of episodes: {len(episodes)}")
        
        # Sample first episode
        ep = f[episodes[0]]
        print(f"\nFirst episode: {episodes[0]}")
        print(f"  Keys: {list(ep.keys())}")
        
        for key in ep.keys():
            shape = ep[key].shape
            dtype = ep[key].dtype
            print(f"  {key}: shape={shape}, dtype={dtype}")
        
        # Calculate sequences
        total_sequences = 0
        ep_lengths = []
        for ep_name in episodes:
            if "gt_height" in f[ep_name]:
                ep_len = f[ep_name]["gt_height"].shape[0]
                ep_lengths.append(ep_len)
                n_seqs = max(0, (ep_len - seq_len) // stride + 1)
                total_sequences += n_seqs
        
        print(f"\n--- Sequence Statistics (seq_len={seq_len}, stride={stride}) ---")
        print(f"Total frames: {sum(ep_lengths)}")
        print(f"Total sequences: {total_sequences}")
        print(f"Episode lengths: min={min(ep_lengths)}, max={max(ep_lengths)}, mean={np.mean(ep_lengths):.1f}")
        
        # Time analysis
        fps = 50  # Assuming 50Hz
        seq_time = seq_len / fps
        print(f"\nAt {fps}Hz:")
        print(f"  seq_len={seq_len} = {seq_time:.2f} seconds")
        print(f"  stride={stride} = {stride/fps:.2f} seconds between sequence starts")
        print(f"  Overlap: {max(0, seq_len - stride)} frames = {max(0, seq_len - stride)/fps:.2f}s")
        
        # Sample a sequence
        print(f"\n--- Sample Sequence ---")
        ep = f[episodes[0]]
        start = 0
        end = seq_len
        
        pos_seq = ep["robot_pos"][start:end]
        quat_seq = ep["robot_quat"][start:end]
        
        print(f"Position changes over {seq_len} frames:")
        pos_delta = pos_seq[-1] - pos_seq[0]
        print(f"  Total delta: x={pos_delta[0]:.3f}m, y={pos_delta[1]:.3f}m, z={pos_delta[2]:.3f}m")
        print(f"  Distance: {np.linalg.norm(pos_delta[:2]):.3f}m")
        
        # Per-frame deltas
        frame_deltas = np.linalg.norm(np.diff(pos_seq[:, :2], axis=0), axis=1)
        print(f"  Per-frame movement: mean={frame_deltas.mean()*100:.2f}cm, max={frame_deltas.max()*100:.2f}cm")
        
        return pos_seq, quat_seq


def find_interesting_frames(h5_path, min_variance=0.001):
    """Find frames with interesting terrain (high height variance)."""
    print(f"\n{'='*60}")
    print(f"SEARCHING FOR INTERESTING TERRAIN")
    print(f"{'='*60}\n")
    
    interesting = []
    
    with h5py.File(h5_path, 'r') as f:
        for ep_idx, ep_name in enumerate(list(f.keys())[:20]):  # Check first 20 episodes
            gt = f[ep_name]["gt_height"][:] * 0.001
            
            for frame_idx in range(len(gt)):
                var = np.var(gt[frame_idx])
                if var > min_variance:
                    interesting.append({
                        'episode': ep_idx,
                        'ep_name': ep_name,
                        'frame': frame_idx,
                        'variance': var,
                        'height_range': gt[frame_idx].max() - gt[frame_idx].min()
                    })
    
    # Sort by variance
    interesting.sort(key=lambda x: x['variance'], reverse=True)
    
    print(f"Found {len(interesting)} interesting frames (variance > {min_variance})")
    if interesting:
        print("\nTop 10 most interesting:")
        for i, item in enumerate(interesting[:10]):
            print(f"  {i+1}. Episode {item['episode']} ({item['ep_name']}), Frame {item['frame']}: "
                  f"var={item['variance']:.4f}, range={item['height_range']*100:.1f}cm")
    
    return interesting


def visualize_frame_transform(h5_path, episode_idx=0, frame_start=50, n_frames=5, auto_find=True):
    """Visualize frame-to-frame coordinate transforms for V6 debugging."""
    print(f"\n{'='*60}")
    print(f"FRAME TRANSFORM VISUALIZATION")
    print(f"{'='*60}\n")
    
    with h5py.File(h5_path, 'r') as f:
        episodes = list(f.keys())
        
        # Auto-find interesting terrain if requested
        if auto_find:
            print("Auto-finding interesting terrain...")
            best_var = 0
            best_ep = 0
            best_frame = 50
            
            for ep_idx, ep_name in enumerate(episodes[:30]):
                gt = f[ep_name]["gt_height"][:] * 0.001
                for frame_idx in range(min(len(gt) - n_frames, 500)):
                    var = np.var(gt[frame_idx])
                    if var > best_var:
                        best_var = var
                        best_ep = ep_idx
                        best_frame = frame_idx
            
            episode_idx = best_ep
            frame_start = best_frame
            print(f"Found: Episode {episode_idx}, Frame {frame_start} (variance={best_var:.4f})")
        
        ep_name = episodes[episode_idx]
        ep = f[ep_name]
        
        gt_seq = torch.tensor(ep["gt_height"][frame_start:frame_start+n_frames]).float() * 0.001
        pos_seq = torch.tensor(ep["robot_pos"][frame_start:frame_start+n_frames]).float()
        quat_seq = torch.tensor(ep["robot_quat"][frame_start:frame_start+n_frames]).float()
        
        yaw_seq = euler_from_quat(quat_seq)
        if len(yaw_seq.shape) > 1:
            yaw_seq = yaw_seq.squeeze()
    
    print(f"Visualizing Episode {episode_idx}, Frames {frame_start}-{frame_start+n_frames-1}")
    print(f"Height range in sequence: {gt_seq.min()*100:.1f}cm to {gt_seq.max()*100:.1f}cm")
    
    fig = plt.figure(figsize=(16, 4*n_frames))
    
    for t in range(1, n_frames):
        # Get previous and current
        prev_map = gt_seq[t-1:t].unsqueeze(1)  # (1, 1, H, W)
        curr_map = gt_seq[t:t+1].unsqueeze(1)
        
        prev_pos = pos_seq[t-1:t]
        curr_pos = pos_seq[t:t+1]
        prev_yaw = yaw_seq[t-1:t]
        curr_yaw = yaw_seq[t:t+1]
        
        # Transform
        transformed, dx, dy, dyaw = transform_height_map(
            prev_map, prev_pos, prev_yaw, curr_pos, curr_yaw
        )
        
        # Calculate difference
        diff = torch.abs(transformed - curr_map)
        valid_mask = (transformed != 0).float()
        
        # Plot
        row = t - 1
        
        ax1 = fig.add_subplot(n_frames-1, 5, row*5 + 1)
        ax1.imshow(prev_map.squeeze().numpy(), cmap='viridis', vmin=-0.5, vmax=0.5)
        ax1.set_title(f'Frame {t-1} (prev)')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(n_frames-1, 5, row*5 + 2)
        ax2.imshow(transformed.squeeze().numpy(), cmap='viridis', vmin=-0.5, vmax=0.5)
        ax2.set_title(f'Prev→Curr transform')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(n_frames-1, 5, row*5 + 3)
        ax3.imshow(curr_map.squeeze().numpy(), cmap='viridis', vmin=-0.5, vmax=0.5)
        ax3.set_title(f'Frame {t} (curr GT)')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(n_frames-1, 5, row*5 + 4)
        ax4.imshow(diff.squeeze().numpy(), cmap='hot', vmin=0, vmax=0.1)
        ax4.set_title(f'|Transform - GT|')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(n_frames-1, 5, row*5 + 5)
        ax5.imshow(valid_mask.squeeze().numpy(), cmap='gray')
        ax5.set_title(f'Valid mask')
        ax5.axis('off')
        
        # Print transform info
        print(f"Frame {t-1} → {t}:")
        print(f"  Position delta (world): {(curr_pos - prev_pos).squeeze()[:2].numpy()} m")
        print(f"  Position delta (robot): dx={dx.item()*100:.2f}cm, dy={dy.item()*100:.2f}cm")
        print(f"  Yaw delta: {np.degrees(dyaw.item()):.2f}°")
        print(f"  Valid pixels: {valid_mask.sum().item():.0f}/{MAP_SIZE*MAP_SIZE} ({100*valid_mask.mean().item():.1f}%)")
        print(f"  Mean diff (valid): {(diff * valid_mask).sum() / (valid_mask.sum() + 1e-6) * 100:.2f}cm")
        print()
    
    plt.tight_layout()
    save_path = "debug_frame_transform.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def visualize_sequence_coverage(h5_path, seq_len=16, stride=8, episode_idx=0):
    """Visualize how sequences cover an episode."""
    print(f"\n{'='*60}")
    print(f"SEQUENCE COVERAGE VISUALIZATION")
    print(f"{'='*60}\n")
    
    with h5py.File(h5_path, 'r') as f:
        episodes = list(f.keys())
        ep_name = episodes[episode_idx]
        ep_len = f[ep_name]["gt_height"].shape[0]
        
        pos_seq = f[ep_name]["robot_pos"][:]
    
    # Calculate sequences
    sequences = []
    for start in range(0, ep_len - seq_len + 1, stride):
        sequences.append((start, start + seq_len))
    
    print(f"Episode length: {ep_len} frames")
    print(f"Number of sequences: {len(sequences)}")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Sequence coverage
    ax1 = axes[0]
    for i, (start, end) in enumerate(sequences):
        color = plt.cm.tab10(i % 10)
        ax1.barh(0, end-start, left=start, height=0.8, color=color, alpha=0.5, edgecolor='black')
    ax1.set_xlim(0, ep_len)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel('Frame index')
    ax1.set_title(f'Sequence coverage (seq_len={seq_len}, stride={stride})')
    ax1.set_yticks([])
    
    # Add frame markers
    for i in range(0, ep_len, 100):
        ax1.axvline(i, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 2: Robot trajectory colored by sequence
    ax2 = axes[1]
    for i, (start, end) in enumerate(sequences):
        color = plt.cm.tab10(i % 10)
        ax2.plot(pos_seq[start:end, 0], pos_seq[start:end, 1], 
                color=color, linewidth=2, alpha=0.7)
        ax2.scatter(pos_seq[start, 0], pos_seq[start, 1], 
                   color=color, s=50, marker='o', zorder=5)
    
    ax2.set_xlabel('X position (m)')
    ax2.set_ylabel('Y position (m)')
    ax2.set_title('Robot trajectory colored by sequence')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "debug_sequence_coverage.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def test_transform_correctness():
    """Test coordinate transform with known values."""
    print(f"\n{'='*60}")
    print(f"TRANSFORM UNIT TESTS")
    print(f"{'='*60}\n")
    
    # Create test pattern: gradient in X direction
    test_map = torch.zeros(1, 1, MAP_SIZE, MAP_SIZE)
    for i in range(MAP_SIZE):
        test_map[0, 0, :, i] = (i - MAP_SIZE//2) / MAP_SIZE  # -0.5 to 0.5
    
    print("Test 1: No movement (should be identical)")
    prev_pos = torch.tensor([[0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[0.0, 0.0, 0.0]])
    prev_yaw = torch.tensor([0.0])
    curr_yaw = torch.tensor([0.0])
    
    transformed, dx, dy, dyaw = transform_height_map(test_map, prev_pos, prev_yaw, curr_pos, curr_yaw)
    diff = (transformed - test_map).abs().mean()
    print(f"  Mean difference: {diff.item():.6f} (should be ~0)")
    print(f"  PASS" if diff < 1e-5 else f"  FAIL")
    
    print("\nTest 2: Pure translation forward (robot moved +X in world)")
    prev_pos = torch.tensor([[0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[0.1, 0.0, 0.0]])  # 10cm forward
    prev_yaw = torch.tensor([0.0])
    curr_yaw = torch.tensor([0.0])
    
    transformed, dx, dy, dyaw = transform_height_map(test_map, prev_pos, prev_yaw, curr_pos, curr_yaw)
    print(f"  Delta in robot frame: dx={dx.item()*100:.1f}cm, dy={dy.item()*100:.1f}cm")
    print(f"  Map should shift: prev content that was 'ahead' is now 'closer'")
    # Check center column shifted
    center_before = test_map[0, 0, MAP_SIZE//2, :].mean()
    center_after = transformed[0, 0, MAP_SIZE//2, :].mean()
    print(f"  Center row mean: before={center_before:.3f}, after={center_after:.3f}")
    
    print("\nTest 3: Pure rotation (robot rotated 45° CCW)")
    prev_pos = torch.tensor([[0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[0.0, 0.0, 0.0]])
    prev_yaw = torch.tensor([0.0])
    curr_yaw = torch.tensor([np.pi/4])  # 45 degrees
    
    transformed, dx, dy, dyaw = transform_height_map(test_map, prev_pos, prev_yaw, curr_pos, curr_yaw)
    print(f"  Yaw delta: {np.degrees(dyaw.item()):.1f}°")
    print(f"  Map should be rotated")
    
    # Visualize tests
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].imshow(test_map.squeeze().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 0].set_title('Original (X gradient)')
    
    # Test 2 result
    prev_pos = torch.tensor([[0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[0.1, 0.0, 0.0]])
    transformed, _, _, _ = transform_height_map(test_map, prev_pos, torch.tensor([0.0]), curr_pos, torch.tensor([0.0]))
    axes[0, 1].imshow(transformed.squeeze().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('After +10cm X translation')
    
    # Test 3 result
    prev_pos = torch.tensor([[0.0, 0.0, 0.0]])
    curr_pos = torch.tensor([[0.0, 0.0, 0.0]])
    transformed, _, _, _ = transform_height_map(test_map, prev_pos, torch.tensor([0.0]), curr_pos, torch.tensor([np.pi/4]))
    axes[0, 2].imshow(transformed.squeeze().numpy(), cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('After 45° rotation')
    
    # Create checkerboard pattern for better rotation visualization
    checker = torch.zeros(1, 1, MAP_SIZE, MAP_SIZE)
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if (i // 5 + j // 5) % 2 == 0:
                checker[0, 0, i, j] = 1.0
    
    axes[1, 0].imshow(checker.squeeze().numpy(), cmap='gray')
    axes[1, 0].set_title('Checkerboard original')
    
    transformed, _, _, _ = transform_height_map(checker, torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([0.0]), 
                                                 torch.tensor([[0.05, 0.05, 0.0]]), torch.tensor([0.0]))
    axes[1, 1].imshow(transformed.squeeze().numpy(), cmap='gray')
    axes[1, 1].set_title('After +5cm X,Y translation')
    
    transformed, _, _, _ = transform_height_map(checker, torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([0.0]),
                                                 torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([np.pi/6]))
    axes[1, 2].imshow(transformed.squeeze().numpy(), cmap='gray')
    axes[1, 2].set_title('After 30° rotation')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    save_path = "debug_transform_tests.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def recommend_params(h5_path):
    """Recommend seq_len and stride based on dataset analysis."""
    print(f"\n{'='*60}")
    print(f"PARAMETER RECOMMENDATIONS")
    print(f"{'='*60}\n")
    
    with h5py.File(h5_path, 'r') as f:
        episodes = list(f.keys())
        ep = f[episodes[0]]
        
        # Analyze movement
        pos_seq = ep["robot_pos"][:]
        
        frame_deltas = np.linalg.norm(np.diff(pos_seq[:, :2], axis=0), axis=1)
        mean_speed = frame_deltas.mean() * 50  # m/s at 50Hz
        
        print(f"Robot movement analysis:")
        print(f"  Mean speed: {mean_speed:.2f} m/s")
        print(f"  Mean per-frame movement: {frame_deltas.mean()*100:.2f} cm")
    
    # Robot body is ~40cm, map is 2m x 2m
    robot_size = 0.4  # m
    map_size = MAP_SIZE * MAP_RES  # 2m
    
    # How many frames to cross robot body?
    frames_to_cross_robot = robot_size / (frame_deltas.mean() + 1e-6)
    
    # How many frames to cross entire map?
    frames_to_cross_map = map_size / (frame_deltas.mean() + 1e-6)
    
    print(f"\n  Frames to traverse robot body (~40cm): {frames_to_cross_robot:.0f}")
    print(f"  Frames to traverse map (2m): {frames_to_cross_map:.0f}")
    
    # Recommendations
    recommended_seq_len = int(min(frames_to_cross_robot * 2, 32))  # See terrain 2x robot body ahead/behind
    recommended_seq_len = max(8, recommended_seq_len)  # At least 8
    recommended_stride = recommended_seq_len // 2  # 50% overlap
    
    print(f"\n--- Recommendations ---")
    print(f"  seq_len: {recommended_seq_len} ({recommended_seq_len/50:.2f}s)")
    print(f"  stride: {recommended_stride} ({recommended_stride/50:.2f}s)")
    print(f"\nRationale:")
    print(f"  - seq_len covers ~2x robot body traversal")
    print(f"  - Long enough to see terrain approach and pass under robot")
    print(f"  - Short enough for stable gradient flow")
    print(f"  - 50% overlap provides good training coverage")
    
    return recommended_seq_len, recommended_stride


def main():
    parser = argparse.ArgumentParser(description="Debug temporal perception training")
    parser.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--frame_start", type=int, default=50, help="Starting frame for transform viz")
    parser.add_argument("--no_auto_find", action="store_true", help="Disable auto-finding interesting terrain")
    args = parser.parse_args()
    
    # Run all debug functions
    inspect_dataset(args.data, args.seq_len, args.stride)
    test_transform_correctness()
    
    # Find interesting frames first
    interesting = find_interesting_frames(args.data)
    
    # Visualize with auto-find unless disabled
    visualize_frame_transform(
        args.data, 
        args.episode, 
        args.frame_start, 
        n_frames=5,
        auto_find=not args.no_auto_find
    )
    visualize_sequence_coverage(args.data, args.seq_len, args.stride, args.episode)
    recommend_params(args.data)
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE - Check generated PNG files")
    print("="*60)


if __name__ == "__main__":
    main()
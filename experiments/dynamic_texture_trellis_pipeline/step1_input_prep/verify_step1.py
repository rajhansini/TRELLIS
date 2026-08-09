"""Verify Step 1: input prep. Run this to confirm paths and window logic are correct."""

from input_prep import prepare_input, t_to_frame_idx, get_window_indices

# Test t -> frame_idx mapping
assert t_to_frame_idx(0.0) == 1
assert t_to_frame_idx(1.0) == 150
assert t_to_frame_idx(0.5) == 75
print("t -> frame_idx: OK")

# Test clamping at boundaries
assert get_window_indices(1, 3)   == [1, 1, 1, 1, 2, 3, 4]
assert get_window_indices(150, 3) == [147, 148, 149, 150, 150, 150, 150]
assert get_window_indices(75, 3)  == [72, 73, 74, 75, 76, 77, 78]
print("window clamping: OK")

# Test full prepare_input for a mid-frame and a boundary frame
for t, k in [(0.5, 3), (0.0, 3), (1.0, 1)]:
    frame_idx, window_indices, window_frames, mesh_path = prepare_input(t, k)
    assert len(window_indices) == 2 * k + 1
    assert len(window_frames)  == 2 * k + 1
    assert mesh_path.exists(), f"Mesh not found: {mesh_path}"
    for img in window_frames:
        assert img.size[0] > 0
    print(f"t={t}, k={k}: frame_idx={frame_idx}, window={window_indices} -> OK")

print("\nStep 1 verified.")

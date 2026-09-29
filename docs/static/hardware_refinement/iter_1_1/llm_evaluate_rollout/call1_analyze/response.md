```python
def analyze(traj) -> tuple[str, list[int]]:
    import numpy as np
    import math

    step = np.asarray(traj["step"])
    obs = np.asarray(traj["observation"])
    ps = traj.get("policy_state", {})

    T = len(step)
    dt = 0.02

    root_xy = obs[:, 0:2]
    root_yaw = obs[:, 2]
    root_v_local = obs[:, 3:6]
    left_ee = obs[:, 6:9]
    right_ee = obs[:, 9:12]
    pelvis_h = obs[:, 12]
    button_pos = obs[:, 13:16]
    button_dir = obs[:, 16:18]
    last_action = obs[:, 18:28]

    def wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def world_to_local_xy(vec_world, yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = c * vec_world[:, 0] + s * vec_world[:, 1]
        y = -s * vec_world[:, 0] + c * vec_world[:, 1]
        return np.stack([x, y], axis=-1)

    def first_true(mask):
        idx = np.flatnonzero(mask)
        return None if len(idx) == 0 else int(idx[0])

    def fmt_idx(i):
        if i is None:
            return "never"
        return f"step {int(step[i])}"

    def fmt_vec(v):
        return f"({v[0]:.3f}, {v[1]:.3f})"

    # Policy-state fields if present
    phase = np.asarray(ps["phase"]) if "phase" in ps else None
    phase_timer = np.asarray(ps["phase_timer"]) if "phase_timer" in ps else None
    active_left = np.asarray(ps["active_left"]).astype(bool) if "active_left" in ps else None
    smoothed_vx = np.asarray(ps["smoothed_vx"]) if "smoothed_vx" in ps else None
    smoothed_vy = np.asarray(ps["smoothed_vy"]) if "smoothed_vy" in ps else None
    left_cmd = np.asarray(ps["left_cmd"]) if "left_cmd" in ps else None
    right_cmd = np.asarray(ps["right_cmd"]) if "right_cmd" in ps else None
    height_cmd = np.asarray(ps["height_cmd"]) if "height_cmd" in ps else None

    # Reconstruct key policy quantities
    if active_left is None:
        # Fallback from initial geometry using the policy's own init logic
        face_yaw0 = math.atan2(-button_dir[0, 1], -button_dir[0, 0])
        vec0 = button_pos[0, 0:2] - root_xy[0]
        c0, s0 = math.cos(face_yaw0), math.sin(face_yaw0)
        btn_local0_y = -s0 * vec0[0] + c0 * vec0[1]
        active_left = np.full(T, btn_local0_y >= 0.0, dtype=bool)

    side_sign = np.where(active_left, 1.0, -1.0)
    tangent_dir = np.stack([-button_dir[:, 1], button_dir[:, 0]], axis=-1)
    stand_offset = 0.40
    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.09 * side_sign)[:, None]
    to_stand_world = stand_xy - root_xy
    dist_to_stand = np.linalg.norm(to_stand_world, axis=-1)

    face_yaw = np.arctan2(-button_dir[:, 1], -button_dir[:, 0])
    yaw_err = wrap_to_pi(face_yaw - root_yaw)

    button_local = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
    hand_lateral_err = np.where(active_left, button_local[:, 1] - 0.14, button_local[:, 1] + 0.14)

    near_stand = dist_to_stand < 0.24
    enter_align_cond = (dist_to_stand < 0.30) & (np.abs(yaw_err) < 0.32)
    aligned = np.abs(yaw_err) < 0.32
    forward_ready = button_local[:, 0] < 0.40
    lateral_ready = np.abs(hand_lateral_err) < 0.07
    ready_to_press = near_stand & aligned & lateral_ready & forward_ready

    # Hand-to-button geometry in local frame
    press_world_xy = button_pos[:, 0:2] + button_dir * (0.035 - 0.02 + 0.005)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (0.035 + 0.035)
    press_local_xy = world_to_local_xy(press_world_xy - root_xy, root_yaw)
    prepress_local_xy = world_to_local_xy(prepress_world_xy - root_xy, root_yaw)

    # Distances from observed / commanded hands to targets
    left_xy_err_to_press = np.linalg.norm(left_ee[:, 0:2] - press_local_xy, axis=-1)
    right_xy_err_to_press = np.linalg.norm(right_ee[:, 0:2] - press_local_xy, axis=-1)
    left_xy_err_to_pre = np.linalg.norm(left_ee[:, 0:2] - prepress_local_xy, axis=-1)
    right_xy_err_to_pre = np.linalg.norm(right_ee[:, 0:2] - prepress_local_xy, axis=-1)

    active_press_xy_err = np.where(active_left, left_xy_err_to_press, right_xy_err_to_press)
    active_pre_xy_err = np.where(active_left, left_xy_err_to_pre, right_xy_err_to_pre)

    # Reconstruct commanded active-arm targets if policy state exists
    active_cmd_xy_err_to_press = None
    if left_cmd is not None and right_cmd is not None:
        cmd_xy = np.where(active_left[:, None], left_cmd[:, 0:2], right_cmd[:, 0:2])
        active_cmd_xy_err_to_press = np.linalg.norm(cmd_xy - press_local_xy, axis=-1)

    # Base motion/action summaries
    cmd_vx = last_action[:, 0]
    cmd_vy = last_action[:, 1]
    cmd_speed = np.sqrt(cmd_vx ** 2 + cmd_vy ** 2)
    obs_speed = np.sqrt(root_v_local[:, 0] ** 2 + root_v_local[:, 1] ** 2)

    moving_cmd = (np.abs(cmd_vx) >= 0.15) | (np.abs(cmd_vy) >= 0.15)
    standing_cmd = ~moving_cmd

    # Threshold crossings and extrema
    i_min_dist = int(np.argmin(dist_to_stand))
    i_min_btn_x = int(np.argmin(button_local[:, 0]))
    i_min_btn_lat_abs = int(np.argmin(np.abs(hand_lateral_err)))
    i_min_yaw_abs = int(np.argmin(np.abs(yaw_err)))
    i_max_cmd_speed = int(np.argmax(cmd_speed))
    i_max_obs_speed = int(np.argmax(obs_speed))
    i_min_active_press_obs = int(np.argmin(active_press_xy_err))
    i_min_active_pre_obs = int(np.argmin(active_pre_xy_err))

    cross_030 = first_true(dist_to_stand < 0.30)
    cross_024 = first_true(dist_to_stand < 0.24)
    cross_align = first_true(np.abs(yaw_err) < 0.32)
    cross_forward = first_true(forward_ready)
    cross_lateral = first_true(lateral_ready)
    cross_ready = first_true(ready_to_press)
    cross_stand_cmd = first_true(standing_cmd)

    # Phase summary
    phase_lines = []
    if phase is not None:
        change_idx = np.flatnonzero(np.r_[True, phase[1:] != phase[:-1]])
        for j in change_idx:
            phase_lines.append(f"{int(step[j])}:{int(phase[j])}")
        phase_counts = {int(k): int(np.sum(phase == k)) for k in np.unique(phase)}
        phase_summary = ", ".join([f"{k}={v} steps" for k, v in sorted(phase_counts.items())])
    else:
        phase_summary = "unavailable"
        phase_lines = []

    # Action evaluation cadence (policy runs every 5 env steps)
    eval_idx = np.arange(0, T, 5)
    if phase is not None:
        eval_phase = phase[eval_idx]
        eval_change_idx = eval_idx[np.flatnonzero(np.r_[True, eval_phase[1:] != eval_phase[:-1]])]
    else:
        eval_change_idx = np.array([], dtype=int)

    # Whether phase 2 / 3 were ever entered
    first_phase1 = first_true(phase == 1) if phase is not None else None
    first_phase2 = first_true(phase == 2) if phase is not None else None
    first_phase3 = first_true(phase == 3) if phase is not None else None

    # Keyframes: choose visually informative moments
    keyframes = []

    def add_k(i):
        if i is None:
            return
        i = int(i)
        if 0 <= i < T and i not in keyframes:
            keyframes.append(i)

    add_k(0)
    add_k(cross_030)
    add_k(first_phase1)
    add_k(i_min_btn_lat_abs)
    add_k(first_phase2)
    add_k(i_min_active_press_obs)
    add_k(first_phase3)
    add_k(i_min_dist)
    add_k(T - 1)

    # Fill to at least 5, keep at most 10
    extras = [cross_align, cross_forward, cross_lateral, i_min_btn_x, i_min_yaw_abs]
    for i in extras:
        if len(keyframes) >= 10:
            break
        add_k(i)

    keyframes = sorted(keyframes)
    if len(keyframes) > 10:
        keyframes = keyframes[:10]

    lines = []
    lines.append(f"Episode length: {T} steps = {T*dt:.2f}s. Policy eval every 5 steps ({5*dt:.2f}s).")
    lines.append(f"Button face center starts at ({button_pos[0,0]:.3f}, {button_pos[0,1]:.3f}, {button_pos[0,2]:.3f}); face dir=({button_dir[0,0]:.3f}, {button_dir[0,1]:.3f}).")
    lines.append(f"Chosen active hand: {'left' if bool(active_left[0]) else 'right'}.")

    lines.append(f"Root start xy={fmt_vec(root_xy[0])}, yaw={root_yaw[0]:.3f}; end xy={fmt_vec(root_xy[-1])}, yaw={root_yaw[-1]:.3f}.")
    lines.append(f"Pelvis height: start {pelvis_h[0]:.3f}, min {pelvis_h[np.argmin(pelvis_h)]:.3f} at step {int(step[np.argmin(pelvis_h)])}, end {pelvis_h[-1]:.3f}.")
    if height_cmd is not None:
        lines.append(f"Height cmd: start {height_cmd[0]:.3f}, min {height_cmd[np.argmin(height_cmd)]:.3f} at step {int(step[np.argmin(height_cmd)])}, end {height_cmd[-1]:.3f}.")

    lines.append(f"Stand target distance: start {dist_to_stand[0]:.3f} m, min {dist_to_stand[i_min_dist]:.3f} m at step {int(step[i_min_dist])}, end {dist_to_stand[-1]:.3f} m.")
    lines.append(f"First dist<0.30: {fmt_idx(cross_030)}; first dist<0.24: {fmt_idx(cross_024)}.")
    lines.append(f"Yaw error to face button: start {yaw_err[0]:.3f} rad, min |err| {abs(yaw_err[i_min_yaw_abs]):.3f} at step {int(step[i_min_yaw_abs])}, end {yaw_err[-1]:.3f} rad.")
    lines.append(f"First |yaw_err|<0.32: {fmt_idx(cross_align)}.")

    lines.append(f"Button in local frame: min forward x {button_local[i_min_btn_x,0]:.3f} m at step {int(step[i_min_btn_x])}, end x {button_local[-1,0]:.3f} m.")
    lines.append(f"Hand lateral error relative to chosen shoulder line: start {hand_lateral_err[0]:.3f} m, min |err| {abs(hand_lateral_err[i_min_btn_lat_abs]):.3f} m at step {int(step[i_min_btn_lat_abs])}, end {hand_lateral_err[-1]:.3f} m.")
    lines.append(f"First forward_ready (button local x<0.40): {fmt_idx(cross_forward)}; first lateral_ready (|lat err|<0.07): {fmt_idx(cross_lateral)}; first ready_to_press: {fmt_idx(cross_ready)}.")

    lines.append(f"Observed local root speed: max {obs_speed[i_max_obs_speed]:.3f} m/s at step {int(step[i_max_obs_speed])}; end {obs_speed[-1]:.3f} m/s.")
    lines.append(f"Commanded planar speed from last_action: max {cmd_speed[i_max_cmd_speed]:.3f} at step {int(step[i_max_cmd_speed])}; first standing command (both below deadzone thresholds): {fmt_idx(cross_stand_cmd)}.")
    if smoothed_vx is not None and smoothed_vy is not None:
        sv = np.sqrt(smoothed_vx**2 + smoothed_vy**2)
        im = int(np.argmax(sv))
        lines.append(f"Policy smoothed speed state: max {sv[im]:.3f} at step {int(step[im])}; end {sv[-1]:.3f}.")

    lines.append(f"Active hand observed XY error to prepress target: min {active_pre_xy_err[i_min_active_pre_obs]:.3f} m at step {int(step[i_min_active_pre_obs])}, end {active_pre_xy_err[-1]:.3f} m.")
    lines.append(f"Active hand observed XY error to press target: min {active_press_xy_err[i_min_active_press_obs]:.3f} m at step {int(step[i_min_active_press_obs])}, end {active_press_xy_err[-1]:.3f} m.")
    if active_cmd_xy_err_to_press is not None:
        ic = int(np.argmin(active_cmd_xy_err_to_press))
        lines.append(f"Active hand command XY error to press target: min {active_cmd_xy_err_to_press[ic]:.3f} m at step {int(step[ic])}, end {active_cmd_xy_err_to_press[-1]:.3f} m.")

    if phase is not None:
        lines.append(f"Phase occupancy: {phase_summary}.")
        lines.append(f"Phase changes at env steps: {'; '.join(phase_lines) if phase_lines else 'none'}.")
        lines.append(f"First phase 1: {fmt_idx(first_phase1)}; first phase 2: {fmt_idx(first_phase2)}; first phase 3: {fmt_idx(first_phase3)}.")
        if len(eval_change_idx) > 0:
            lines.append("Phase seen at policy-eval instants: " + "; ".join([f"{int(step[i])}:{int(phase[i])}" for i in eval_change_idx]) + ".")

    # Hand placement at end
    lines.append(
        f"End effector final positions: left=({left_ee[-1,0]:.3f}, {left_ee[-1,1]:.3f}, {left_ee[-1,2]:.3f}), "
        f"right=({right_ee[-1,0]:.3f}, {right_ee[-1,1]:.3f}, {right_ee[-1,2]:.3f})."
    )
    if left_cmd is not None and right_cmd is not None:
        lines.append(
            f"Final commanded hand targets from policy state: left=({left_cmd[-1,0]:.3f}, {left_cmd[-1,1]:.3f}, {left_cmd[-1,2]:.3f}), "
            f"right=({right_cmd[-1,0]:.3f}, {right_cmd[-1,1]:.3f}, {right_cmd[-1,2]:.3f})."
        )

    summary = "\n".join(lines[:150])
    return summary, keyframes
```

def analyze(traj) -> tuple[str, list[int]]:
    import numpy as np
    import math

    step = np.asarray(traj["step"])
    obs = np.asarray(traj["observation"])
    ps = traj.get("policy_state", {})

    T = len(step)
    dt = 0.02
    policy_dt = 0.1

    root_xy = obs[:, 0:2]
    root_yaw = obs[:, 2]
    root_vel_local = obs[:, 3:6]
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

    def fmt_step(i):
        return f"{int(step[i])} ({step[i]*dt:.2f}s)"

    def fmt_vec(v):
        return f"({v[0]:.3f}, {v[1]:.3f})"

    def fmt_vec3(v):
        return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"

    # Reconstruct key policy quantities
    active_left = np.asarray(ps.get("active_left", np.zeros(T, dtype=bool))).astype(bool)
    phase = np.asarray(ps.get("phase", np.full(T, -1, dtype=int))).astype(int)

    side_sign = np.where(active_left, 1.0, -1.0)
    tangent_dir = np.stack([-button_dir[:, 1], button_dir[:, 0]], axis=-1)
    stand_offset = 0.45
    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.12 * side_sign)[:, None]
    to_stand_world = stand_xy - root_xy
    dist_to_stand = np.linalg.norm(to_stand_world, axis=-1)
    to_stand_local = world_to_local_xy(to_stand_world, root_yaw)

    face_yaw = np.arctan2(-button_dir[:, 1], -button_dir[:, 0])
    yaw_err = wrap_to_pi(face_yaw - root_yaw)

    button_local = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
    hand_lateral_err = np.where(active_left, button_local[:, 1] - 0.14, button_local[:, 1] + 0.14)

    near_stand = dist_to_stand < 0.24
    aligned = np.abs(yaw_err) < 0.32
    lateral_ready = np.abs(hand_lateral_err) < 0.07
    enter_align_cond = (dist_to_stand < 0.30) & aligned
    good_enough_press = (dist_to_stand < 0.28) & (np.abs(yaw_err) < 0.38) & (np.abs(hand_lateral_err) < 0.10)
    ready_to_press = near_stand & aligned & lateral_ready

    # End-effector target geometry
    contact_outside = 0.035
    push_depth = 0.02
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.01)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.02)
    press_local_xy = world_to_local_xy(press_world_xy - root_xy, root_yaw)
    prepress_local_xy = world_to_local_xy(prepress_world_xy - root_xy, root_yaw)
    target_z = np.clip(button_pos[:, 2] - 0.72, -0.02, 0.20)

    left_cmd = last_action[:, 3:6]
    right_cmd = last_action[:, 6:9]
    active_cmd = np.where(active_left[:, None], left_cmd, right_cmd)
    active_ee = np.where(active_left[:, None], left_ee, right_ee)

    pre_target = np.stack([
        np.clip(prepress_local_xy[:, 0], 0.18, 0.43),
        np.where(active_left, np.clip(prepress_local_xy[:, 1], 0.08, 0.34), np.clip(prepress_local_xy[:, 1], -0.34, -0.08)),
        target_z
    ], axis=-1)
    press_target = np.stack([
        np.clip(press_local_xy[:, 0], 0.22, 0.45),
        np.where(active_left, np.clip(press_local_xy[:, 1], 0.08, 0.34), np.clip(press_local_xy[:, 1], -0.34, -0.08)),
        target_z
    ], axis=-1)

    cmd_to_pre = np.linalg.norm(active_cmd - pre_target, axis=-1)
    cmd_to_press = np.linalg.norm(active_cmd - press_target, axis=-1)
    ee_to_press = np.linalg.norm(active_ee - press_target, axis=-1)

    # Motion/action summaries
    cmd_speed = np.linalg.norm(last_action[:, 0:2], axis=-1)
    obs_speed_local = np.linalg.norm(root_vel_local[:, 0:2], axis=-1)
    moving_cmd = cmd_speed >= 0.15
    moving_obs = obs_speed_local >= 0.15

    # Button-side / arm choice
    lines = []
    lines.append(f"Episode length: {T} steps = {T*dt:.2f}s. Policy updates every 5 steps = {policy_dt:.2f}s.")
    if "active_left" in ps:
        left_frac = float(np.mean(active_left))
        chosen = "left" if left_frac > 0.5 else "right"
        lines.append(f"Active arm from policy_state: mostly {chosen} ({left_frac*100:.1f}% left). Initial active_left={bool(active_left[0])}.")
    else:
        lines.append("active_left missing from policy_state.")

    # Button geometry
    unique_dirs = np.unique(np.round(button_dir, 3), axis=0)
    lines.append(f"Button face dir starts {fmt_vec(button_dir[0])}; unique rounded dirs seen: {unique_dirs.tolist()[:4]}.")
    lines.append(f"Button center start {fmt_vec3(button_pos[0])}, end {fmt_vec3(button_pos[-1])}.")

    # Distances and alignment
    i_min_dist = int(np.argmin(dist_to_stand))
    i_max_dist = int(np.argmax(dist_to_stand))
    i_min_yaw = int(np.argmin(np.abs(yaw_err)))
    i_min_lat = int(np.argmin(np.abs(hand_lateral_err)))
    i_min_ee_press = int(np.argmin(ee_to_press))
    i_min_cmd_press = int(np.argmin(cmd_to_press))

    lines.append(
        f"Stand-point distance: start {dist_to_stand[0]:.3f} m, min {dist_to_stand[i_min_dist]:.3f} m at step {fmt_step(i_min_dist)}, "
        f"end {dist_to_stand[-1]:.3f} m."
    )
    lines.append(
        f"Yaw error to face pillar: start {yaw_err[0]:+.3f} rad, best |err| {abs(yaw_err[i_min_yaw]):.3f} at step {fmt_step(i_min_yaw)}, "
        f"end {yaw_err[-1]:+.3f} rad."
    )
    lines.append(
        f"Lateral alignment error for active hand: start {hand_lateral_err[0]:+.3f} m, best |err| {abs(hand_lateral_err[i_min_lat]):.3f} at step {fmt_step(i_min_lat)}, "
        f"end {hand_lateral_err[-1]:+.3f} m."
    )

    # Threshold crossings
    for name, mask in [
        ("enter_align condition (dist<0.30 and |yaw|<0.32)", enter_align_cond),
        ("near_stand (dist<0.24)", near_stand),
        ("aligned (|yaw|<0.32)", aligned),
        ("lateral_ready (|lat_err|<0.07)", lateral_ready),
        ("good_enough_press", good_enough_press),
        ("ready_to_press", ready_to_press),
    ]:
        i = first_true(mask)
        lines.append(f"First {name}: {'never' if i is None else 'step ' + fmt_step(i)}.")

    # Phase information
    if "phase" in ps:
        phases_seen = np.unique(phase).tolist()
        lines.append(f"Policy phases seen: {phases_seen}.")
        trans = np.flatnonzero(phase[1:] != phase[:-1]) + 1
        if len(trans) == 0:
            lines.append("No phase transitions recorded.")
        else:
            trans_str = ", ".join([f"{int(phase[i-1])}->{int(phase[i])} at step {fmt_step(i)}" for i in trans[:12]])
            if len(trans) > 12:
                trans_str += f", ... total {len(trans)} transitions"
            lines.append(f"Phase transitions: {trans_str}.")
        for ph in sorted(set(phases_seen)):
            idx = np.flatnonzero(phase == ph)
            if len(idx):
                lines.append(f"Phase {ph}: {len(idx)} steps ({len(idx)*dt:.2f}s), first at step {fmt_step(idx[0])}, last at step {fmt_step(idx[-1])}.")
    else:
        lines.append("phase missing from policy_state.")

    # Motion summary
    lines.append(
        f"Commanded base speed |(vx,vy)|: start {cmd_speed[0]:.3f}, max {cmd_speed[int(np.argmax(cmd_speed))]:.3f} at step {fmt_step(int(np.argmax(cmd_speed)))}, "
        f"end {cmd_speed[-1]:.3f}."
    )
    lines.append(
        f"Observed local base speed: start {obs_speed_local[0]:.3f}, max {obs_speed_local[int(np.argmax(obs_speed_local))]:.3f} at step {fmt_step(int(np.argmax(obs_speed_local)))}, "
        f"end {obs_speed_local[-1]:.3f}."
    )
    lines.append(
        f"Pelvis height: start {pelvis_h[0]:.3f}, min {pelvis_h[int(np.argmin(pelvis_h))]:.3f} at step {fmt_step(int(np.argmin(pelvis_h)))}, "
        f"max {pelvis_h[int(np.argmax(pelvis_h))]:.3f} at step {fmt_step(int(np.argmax(pelvis_h)))}, end {pelvis_h[-1]:.3f}."
    )

    # Arm command / actual hand placement summary
    arm_name = "left" if active_left[0] else "right"
    lines.append(
        f"Active {arm_name} command vs targets: min cmd-to-pre {cmd_to_pre[int(np.argmin(cmd_to_pre))]:.3f} m at step {fmt_step(int(np.argmin(cmd_to_pre)))}, "
        f"min cmd-to-press {cmd_to_press[i_min_cmd_press]:.3f} m at step {fmt_step(i_min_cmd_press)}."
    )
    lines.append(
        f"Active {arm_name} actual EE to press-target: start {ee_to_press[0]:.3f} m, min {ee_to_press[i_min_ee_press]:.3f} m at step {fmt_step(i_min_ee_press)}, "
        f"end {ee_to_press[-1]:.3f} m."
    )

    # Reach / frontage diagnostics from geometry only
    press_x = press_target[:, 0]
    press_y = press_target[:, 1]
    press_z = press_target[:, 2]
    lines.append(
        f"Press target in pelvis frame for active hand: x range [{press_x.min():.3f}, {press_x.max():.3f}], "
        f"y range [{press_y.min():.3f}, {press_y.max():.3f}], z range [{press_z.min():.3f}, {press_z.max():.3f}]."
    )

    # Image-worthy frame selection
    frames = []

    def add_idx(i):
        i = int(max(0, min(T - 1, i)))
        if i not in frames:
            frames.append(i)

    add_idx(0)

    if "phase" in ps and np.any(phase == 1):
        add_idx(np.flatnonzero(phase == 1)[0])
    else:
        i = first_true(enter_align_cond)
        if i is not None:
            add_idx(i)

    if "phase" in ps and np.any(phase == 2):
        add_idx(np.flatnonzero(phase == 2)[0])
        add_idx(int(np.flatnonzero(phase == 2)[len(np.flatnonzero(phase == 2)) // 2]))
    else:
        i = first_true(good_enough_press)
        if i is not None:
            add_idx(i)

    add_idx(i_min_dist)
    add_idx(i_min_ee_press)

    if "phase" in ps and np.any(phase == 3):
        add_idx(np.flatnonzero(phase == 3)[0])

    # Add final frame and a late manipulation frame if needed
    add_idx(T - 1)
    if len(frames) < 5:
        add_idx(i_min_yaw)
        add_idx(i_min_lat)
        add_idx(i_min_cmd_press)

    # Cap to 10 while preserving order
    frames = frames[:10]

    text = "\n".join(lines[:150])
    return text, frames

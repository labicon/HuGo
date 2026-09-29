def analyze(traj) -> tuple[str, list[int]]:
    import numpy as np
    import math

    step = np.asarray(traj["step"])
    obs = np.asarray(traj["observation"])
    ps = traj.get("policy_state", {})
    T = len(step)

    def wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def world_to_local_xy(v_world, yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        x = c * v_world[:, 0] + s * v_world[:, 1]
        y = -s * v_world[:, 0] + c * v_world[:, 1]
        return np.stack([x, y], axis=-1)

    def first_true(mask):
        idx = np.flatnonzero(mask)
        return None if len(idx) == 0 else int(idx[0])

    def last_true(mask):
        idx = np.flatnonzero(mask)
        return None if len(idx) == 0 else int(idx[-1])

    def fmt_step(i):
        return f"{int(step[i])}"

    # Observation parse
    root_xy = obs[:, 0:2]
    root_yaw = obs[:, 2]
    root_vel_local = obs[:, 3:6]
    left_ee = obs[:, 6:9]
    right_ee = obs[:, 9:12]
    pelvis_h = obs[:, 12]
    button_pos = obs[:, 13:16]
    button_dir = obs[:, 16:18]
    last_action = obs[:, 18:28]

    # Reconstruct key policy geometry
    face_yaw = np.arctan2(-button_dir[:, 1], -button_dir[:, 0])

    # init-style active arm choice from policy
    btn_local_in_face = world_to_local_xy(button_pos[:, 0:2] - root_xy, face_yaw)
    active_left_init = btn_local_in_face[0, 1] >= 0.0
    active_name = "left" if active_left_init else "right"

    side_sign = 1.0 if active_left_init else -1.0
    tangent_dir = np.stack([-button_dir[:, 1], button_dir[:, 0]], axis=-1)
    stand_xy = button_pos[:, 0:2] + button_dir * 0.42 + tangent_dir * (0.10 * side_sign)

    to_stand_world = stand_xy - root_xy
    dist_to_stand = np.linalg.norm(to_stand_world, axis=-1)
    to_stand_local = world_to_local_xy(to_stand_world, root_yaw)
    yaw_err = wrap_to_pi(face_yaw - root_yaw)

    button_local_now = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
    hand_lateral_err = button_local_now[:, 1] - (0.14 if active_left_init else -0.14)

    near_stand = dist_to_stand < 0.24
    aligned = np.abs(yaw_err) < 0.32
    lateral_ready = np.abs(hand_lateral_err) < 0.07
    enter_align = (dist_to_stand < 0.30) & aligned
    ready_to_press = near_stand & aligned & lateral_ready
    good_enough_press = (dist_to_stand < 0.20) & (np.abs(yaw_err) < 0.24) & (np.abs(hand_lateral_err) < 0.06)

    # Base motion / height commands from last action
    cmd_vx = last_action[:, 0]
    cmd_vy = last_action[:, 1]
    cmd_speed = np.linalg.norm(last_action[:, 0:2], axis=-1)
    root_speed_local = np.linalg.norm(root_vel_local[:, 0:2], axis=-1)
    cmd_yaw = last_action[:, 2]
    cmd_h = last_action[:, 9]

    # Policy state if present
    phase = np.asarray(ps["phase"]) if "phase" in ps else None
    phase_timer = np.asarray(ps["phase_timer"]) if "phase_timer" in ps else None
    left_cmd = np.asarray(ps["left_cmd"]) if "left_cmd" in ps else last_action[:, 3:6]
    right_cmd = np.asarray(ps["right_cmd"]) if "right_cmd" in ps else last_action[:, 6:9]

    active_cmd = left_cmd if active_left_init else right_cmd
    active_obs = left_ee if active_left_init else right_ee

    # Button contact geometry in local frame
    contact_outside = 0.035
    push_depth = 0.02
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.02)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.02)
    press_local_xy = world_to_local_xy(press_world_xy - root_xy, root_yaw)
    prepress_local_xy = world_to_local_xy(prepress_world_xy - root_xy, root_yaw)

    # Distances between commanded/observed active hand and intended targets
    cmd_pre_xy_err = np.linalg.norm(active_cmd[:, 0:2] - prepress_local_xy, axis=-1)
    cmd_press_xy_err = np.linalg.norm(active_cmd[:, 0:2] - press_local_xy, axis=-1)
    obs_pre_xyz_err = np.linalg.norm(active_obs - np.column_stack([prepress_local_xy, active_obs[:, 2]]), axis=-1)
    obs_cmd_xyz_err = np.linalg.norm(active_obs - active_cmd, axis=-1)

    # Extremes and events
    i_min_stand = int(np.argmin(dist_to_stand))
    i_min_yaw = int(np.argmin(np.abs(yaw_err)))
    i_min_lat = int(np.argmin(np.abs(hand_lateral_err)))
    i_max_active_x = int(np.argmax(active_obs[:, 0]))
    i_max_pelvis = int(np.argmax(pelvis_h))
    i_min_pelvis = int(np.argmin(pelvis_h))
    i_max_cmd_speed = int(np.argmax(cmd_speed))
    i_max_root_speed = int(np.argmax(root_speed_local))

    first_enter_align = first_true(enter_align)
    first_near_stand = first_true(near_stand)
    first_aligned = first_true(aligned)
    first_lateral_ready = first_true(lateral_ready)
    first_ready = first_true(ready_to_press)
    first_good_press = first_true(good_enough_press)

    # Phase summary
    phase_lines = []
    key_phase_steps = []
    if phase is not None:
        uniq = [int(phase[0])]
        trans = []
        for i in range(1, T):
            if int(phase[i]) != int(phase[i - 1]):
                trans.append((i, int(phase[i - 1]), int(phase[i])))
                uniq.append(int(phase[i]))
        counts = {p: int(np.sum(phase == p)) for p in sorted(set(phase.astype(int).tolist()))}
        phase_lines.append("phase counts: " + ", ".join([f"{p}:{counts[p]}" for p in sorted(counts)]))
        if trans:
            phase_lines.append("phase transitions: " + "; ".join([f"{a}->{b}@{fmt_step(i)}" for i, a, b in trans[:12]]))
        else:
            phase_lines.append("phase transitions: none")
        key_phase_steps = [i for i, _, _ in trans[:6]]

    # Oscillation / retries
    stop_start_idxs = np.flatnonzero((cmd_speed[:-1] < 1e-6) & (cmd_speed[1:] > 0.12)) + 1
    stop_resume_count = int(len(stop_start_idxs))

    lines = []
    lines.append(f"T={T} steps ({T*0.02:.2f}s). Policy ticks every 5 sim steps ({(T-1)//5 + 1} policy calls).")
    lines.append(
        f"Button face center starts at ({button_pos[0,0]:.3f}, {button_pos[0,1]:.3f}, {button_pos[0,2]:.3f}), "
        f"face dir=({button_dir[0,0]:.3f}, {button_dir[0,1]:.3f}), chosen active arm={active_name}."
    )
    lines.append(
        f"Root start=({root_xy[0,0]:.3f}, {root_xy[0,1]:.3f}, yaw {root_yaw[0]:.3f}), "
        f"end=({root_xy[-1,0]:.3f}, {root_xy[-1,1]:.3f}, yaw {root_yaw[-1]:.3f})."
    )

    lines.append(
        f"Stand target offset metric: dist_to_stand start {dist_to_stand[0]:.3f}, "
        f"min {dist_to_stand[i_min_stand]:.3f}@{fmt_step(i_min_stand)}, end {dist_to_stand[-1]:.3f}."
    )
    lines.append(
        f"Facing error |yaw_err| start {abs(yaw_err[0]):.3f}, "
        f"min {abs(yaw_err[i_min_yaw]):.3f}@{fmt_step(i_min_yaw)}, end {abs(yaw_err[-1]):.3f}."
    )
    lines.append(
        f"Active-hand lateral alignment error |button_local_y - shoulder_side| start {abs(hand_lateral_err[0]):.3f}, "
        f"min {abs(hand_lateral_err[i_min_lat]):.3f}@{fmt_step(i_min_lat)}, end {abs(hand_lateral_err[-1]):.3f}."
    )

    lines.append(
        f"Threshold crossings: aligned<{0.32:.2f} at "
        f"{'never' if first_aligned is None else fmt_step(first_aligned)}; "
        f"dist_to_stand<0.30 at {'never' if first_enter_align is None else fmt_step(first_enter_align)}; "
        f"near_stand<0.24 at {'never' if first_near_stand is None else fmt_step(first_near_stand)}; "
        f"lateral_ready<0.07 at {'never' if first_lateral_ready is None else fmt_step(first_lateral_ready)}."
    )
    lines.append(
        f"Ready conditions: ready_to_press first "
        f"{'never' if first_ready is None else fmt_step(first_ready)}, "
        f"good_enough_press first {'never' if first_good_press is None else fmt_step(first_good_press)}; "
        f"ready_to_press held for {int(np.sum(ready_to_press))} / {T} steps."
    )

    lines.append(
        f"Commanded base speed max {cmd_speed[i_max_cmd_speed]:.3f}@{fmt_step(i_max_cmd_speed)}; "
        f"actual local root speed max {root_speed_local[i_max_root_speed]:.3f}@{fmt_step(i_max_root_speed)}; "
        f"motion resumes from zero-speed command {stop_resume_count} times."
    )
    lines.append(
        f"Pelvis height cmd start/end {cmd_h[0]:.3f}/{cmd_h[-1]:.3f}; "
        f"observed pelvis height min {pelvis_h[i_min_pelvis]:.3f}@{fmt_step(i_min_pelvis)}, "
        f"max {pelvis_h[i_max_pelvis]:.3f}@{fmt_step(i_max_pelvis)}, end {pelvis_h[-1]:.3f}."
    )

    lines.append(
        f"Active {active_name} hand observed x reach: start {active_obs[0,0]:.3f}, "
        f"max {active_obs[i_max_active_x,0]:.3f}@{fmt_step(i_max_active_x)}, end {active_obs[-1,0]:.3f}."
    )
    i_min_cmd_press = int(np.argmin(cmd_press_xy_err))
    i_min_cmd_pre = int(np.argmin(cmd_pre_xy_err))
    i_min_obs_cmd = int(np.argmin(obs_cmd_xyz_err))
    lines.append(
        f"Active-hand command vs target: min cmd->prepress XY err {cmd_pre_xy_err[i_min_cmd_pre]:.3f}@{fmt_step(i_min_cmd_pre)}, "
        f"min cmd->press XY err {cmd_press_xy_err[i_min_cmd_press]:.3f}@{fmt_step(i_min_cmd_press)}."
    )
    lines.append(
        f"Active-hand tracking: min observed->command XYZ err {obs_cmd_xyz_err[i_min_obs_cmd]:.3f}@{fmt_step(i_min_obs_cmd)}, "
        f"end observed->command err {obs_cmd_xyz_err[-1]:.3f}."
    )

    # Reachability / local target geometry at best stance
    i_best = i_min_stand
    lines.append(
        f"At closest stand point @{fmt_step(i_best)}: button local=({button_local_now[i_best,0]:.3f}, {button_local_now[i_best,1]:.3f}), "
        f"prepress local XY=({prepress_local_xy[i_best,0]:.3f}, {prepress_local_xy[i_best,1]:.3f}), "
        f"press local XY=({press_local_xy[i_best,0]:.3f}, {press_local_xy[i_best,1]:.3f})."
    )

    if phase_lines:
        lines.extend(phase_lines)

    # Choose keyframes: start, first transition/approach milestones, best stance/alignment, max reach, late retry/end
    candidates = []
    for i in [
        0,
        first_enter_align,
        first_near_stand,
        first_lateral_ready,
        first_ready,
        i_min_stand,
        i_min_yaw,
        i_max_active_x,
        first_good_press,
        last_true(ready_to_press),
        T - 1,
    ] + key_phase_steps:
        if i is not None and 0 <= i < T:
            candidates.append(int(i))

    # dedupe while preserving order, and enforce spacing preference
    chosen = []
    for i in candidates:
        if all(abs(i - j) >= 8 for j in chosen):
            chosen.append(i)
        elif i in (0, T - 1):
            if i not in chosen:
                chosen.append(i)
    if 0 not in chosen:
        chosen = [0] + chosen
    if T - 1 not in chosen:
        chosen.append(T - 1)

    # cap to 10 with informative spread
    chosen = sorted(set(chosen), key=lambda x: x)
    if len(chosen) > 10:
        anchor = [0, T - 1, i_min_stand, i_max_active_x]
        anchor = [i for i in anchor if 0 <= i < T]
        keep = []
        for i in anchor + chosen:
            if i not in keep:
                keep.append(i)
        chosen = sorted(keep[:10])

    # ensure at least 5
    if len(chosen) < 5:
        extras = [T // 4, T // 2, (3 * T) // 4]
        for i in extras:
            if 0 <= i < T and i not in chosen:
                chosen.append(i)
        chosen = sorted(chosen)[:10]

    return "\n".join(lines[:150]), chosen

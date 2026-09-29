def analyze(traj) -> tuple[str, list[int]]:
    import numpy as np
    import math

    step = np.asarray(traj["step"])
    obs = np.asarray(traj["observation"])
    ps = traj.get("policy_state", {})
    T = len(step)

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
        return int(idx[0]) if idx.size else None

    def last_true(mask):
        idx = np.flatnonzero(mask)
        return int(idx[-1]) if idx.size else None

    def fmt_idx(i):
        return f"{step[i]}" if i is not None else "never"

    def fmt_f(x):
        return f"{float(x):.3f}"

    # Recover policy-side quantities
    if "active_left" in ps:
        active_left = np.asarray(ps["active_left"]).astype(bool)
    else:
        # fallback: infer from initial button side in face frame, same as policy init
        face_yaw0 = math.atan2(-button_dir[0, 1], -button_dir[0, 0])
        c = math.cos(face_yaw0)
        s = math.sin(face_yaw0)
        v = button_pos[0, 0:2] - root_xy[0]
        btn_local_y0 = -s * v[0] + c * v[1]
        active_left = np.full(T, btn_local_y0 >= 0.0, dtype=bool)

    side_sign = np.where(active_left, 1.0, -1.0)
    tangent_dir = np.stack([-button_dir[:, 1], button_dir[:, 0]], axis=-1)
    stand_xy = button_pos[:, 0:2] + button_dir * 0.40 + tangent_dir * (0.09 * side_sign)[:, None]
    to_stand_world = stand_xy - root_xy
    dist_to_stand = np.linalg.norm(to_stand_world, axis=-1)
    to_stand_local = world_to_local_xy(to_stand_world, root_yaw)

    face_yaw = np.arctan2(-button_dir[:, 1], -button_dir[:, 0])
    yaw_err = wrap_to_pi(face_yaw - root_yaw)

    button_local = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
    hand_lateral_err = np.where(active_left, button_local[:, 1] - 0.14, button_local[:, 1] + 0.14)

    near_stand = dist_to_stand < 0.24
    enter_align = (dist_to_stand < 0.30) & (np.abs(yaw_err) < 0.32)
    forward_ready = button_local[:, 0] < 0.40
    lateral_ready = np.abs(hand_lateral_err) < 0.07
    ready_to_press = near_stand & (np.abs(yaw_err) < 0.32) & lateral_ready & forward_ready
    good_enough_press = (
        (dist_to_stand < 0.22)
        & (np.abs(yaw_err) < 0.30)
        & (np.abs(hand_lateral_err) < 0.08)
        & (button_local[:, 0] < 0.395)
    )

    # Approximate contact/press geometry in pelvis frame
    contact_outside = 0.035
    push_depth = 0.02
    extra_press_margin = 0.030
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - extra_press_margin)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.045)
    press_local_xy = world_to_local_xy(press_world_xy - root_xy, root_yaw)
    prepress_local_xy = world_to_local_xy(prepress_world_xy - root_xy, root_yaw)

    # Actual active hand position and errors to policy targets
    active_hand = np.where(active_left[:, None], left_ee, right_ee)
    active_press_y = np.where(active_left, press_local_xy[:, 1], press_local_xy[:, 1])
    active_pre_y = np.where(active_left, prepress_local_xy[:, 1], prepress_local_xy[:, 1])

    err_press_xy = np.stack([
        active_hand[:, 0] - press_local_xy[:, 0],
        active_hand[:, 1] - active_press_y
    ], axis=-1)
    err_pre_xy = np.stack([
        active_hand[:, 0] - prepress_local_xy[:, 0],
        active_hand[:, 1] - active_pre_y
    ], axis=-1)
    press_xy_err_norm = np.linalg.norm(err_press_xy, axis=-1)
    pre_xy_err_norm = np.linalg.norm(err_pre_xy, axis=-1)

    # Policy state summaries
    phase = np.asarray(ps["phase"]).astype(int) if "phase" in ps else None
    phase_timer = np.asarray(ps["phase_timer"]) if "phase_timer" in ps else None

    # Action / movement summaries
    cmd_vx = last_action[:, 0]
    cmd_vy = last_action[:, 1]
    cmd_speed = np.linalg.norm(last_action[:, 0:2], axis=-1)
    obs_speed = np.linalg.norm(root_vel_local[:, 0:2], axis=-1)

    # Which arm was used
    left_frac = float(active_left.mean())
    arm_name = "left" if left_frac > 0.5 else "right"

    lines = []
    lines.append(f"episode: T={T} steps ({T*0.02:.2f}s), policy updates every 5 steps")
    lines.append(
        f"button face center start=({fmt_f(button_pos[0,0])}, {fmt_f(button_pos[0,1])}, {fmt_f(button_pos[0,2])}), "
        f"face dir=({fmt_f(button_dir[0,0])}, {fmt_f(button_dir[0,1])}), chosen arm={arm_name}"
    )

    # Root / navigation
    i_min_dist = int(np.argmin(dist_to_stand))
    i_max_dist = int(np.argmax(dist_to_stand))
    lines.append(
        f"stand target distance: start={fmt_f(dist_to_stand[0])}, min={fmt_f(dist_to_stand[i_min_dist])}@{step[i_min_dist]}, "
        f"max={fmt_f(dist_to_stand[i_max_dist])}@{step[i_max_dist]}, end={fmt_f(dist_to_stand[-1])}"
    )
    lines.append(
        f"first near stand (<0.30) at {fmt_idx(first_true(dist_to_stand < 0.30))}; "
        f"first near_stand (<0.24) at {fmt_idx(first_true(near_stand))}; "
        f"last near_stand at {fmt_idx(last_true(near_stand))}"
    )

    i_best_yaw = int(np.argmin(np.abs(yaw_err)))
    i_worst_yaw = int(np.argmax(np.abs(yaw_err)))
    lines.append(
        f"yaw-to-face error |rad|: start={fmt_f(abs(yaw_err[0]))}, best={fmt_f(abs(yaw_err[i_best_yaw]))}@{step[i_best_yaw]}, "
        f"worst={fmt_f(abs(yaw_err[i_worst_yaw]))}@{step[i_worst_yaw]}, end={fmt_f(abs(yaw_err[-1]))}"
    )
    lines.append(
        f"first aligned (|yaw_err|<0.32) at {fmt_idx(first_true(np.abs(yaw_err) < 0.32))}; "
        f"first enter_align at {fmt_idx(first_true(enter_align))}"
    )

    i_best_lat = int(np.argmin(np.abs(hand_lateral_err)))
    lines.append(
        f"button local x: start={fmt_f(button_local[0,0])}, min={fmt_f(button_local[:,0].min())}@{step[int(np.argmin(button_local[:,0]))]}, "
        f"end={fmt_f(button_local[-1,0])}; "
        f"lateral hand-line err: start={fmt_f(hand_lateral_err[0])}, best_abs={fmt_f(abs(hand_lateral_err[i_best_lat]))}@{step[i_best_lat]}, end={fmt_f(hand_lateral_err[-1])}"
    )
    lines.append(
        f"first forward_ready (x<0.40) at {fmt_idx(first_true(forward_ready))}; "
        f"first lateral_ready (|lat err|<0.07) at {fmt_idx(first_true(lateral_ready))}; "
        f"first ready_to_press at {fmt_idx(first_true(ready_to_press))}; "
        f"first good_enough_press at {fmt_idx(first_true(good_enough_press))}"
    )

    # Motion commands and actual motion
    moving_cmd = cmd_speed > 1e-6
    i_max_cmd = int(np.argmax(cmd_speed))
    i_max_obs = int(np.argmax(obs_speed))
    lines.append(
        f"commanded planar speed: max={fmt_f(cmd_speed[i_max_cmd])}@{step[i_max_cmd]}, "
        f"first zero-command at {fmt_idx(first_true(~moving_cmd))}, last nonzero-command at {fmt_idx(last_true(moving_cmd))}"
    )
    lines.append(
        f"observed planar speed: start={fmt_f(obs_speed[0])}, max={fmt_f(obs_speed[i_max_obs])}@{step[i_max_obs]}, end={fmt_f(obs_speed[-1])}"
    )

    # Pelvis / standing
    i_min_h = int(np.argmin(pelvis_h))
    i_max_h = int(np.argmax(pelvis_h))
    lines.append(
        f"pelvis height: start={fmt_f(pelvis_h[0])}, min={fmt_f(pelvis_h[i_min_h])}@{step[i_min_h]}, "
        f"max={fmt_f(pelvis_h[i_max_h])}@{step[i_max_h]}, end={fmt_f(pelvis_h[-1])}"
    )

    # Hand reach / placement
    active_x = active_hand[:, 0]
    active_y = active_hand[:, 1]
    active_z = active_hand[:, 2]
    lines.append(
        f"active hand pos in pelvis frame: x start/end={fmt_f(active_x[0])}/{fmt_f(active_x[-1])}, "
        f"x max={fmt_f(active_x.max())}@{step[int(np.argmax(active_x))]}; "
        f"y start/end={fmt_f(active_y[0])}/{fmt_f(active_y[-1])}; "
        f"z start/end={fmt_f(active_z[0])}/{fmt_f(active_z[-1])}, z max={fmt_f(active_z.max())}@{step[int(np.argmax(active_z))]}"
    )

    i_best_press = int(np.argmin(press_xy_err_norm))
    i_best_pre = int(np.argmin(pre_xy_err_norm))
    lines.append(
        f"active-hand XY error to prepress target: min={fmt_f(pre_xy_err_norm[i_best_pre])}@{step[i_best_pre]}, end={fmt_f(pre_xy_err_norm[-1])}; "
        f"to press target: min={fmt_f(press_xy_err_norm[i_best_press])}@{step[i_best_press]}, end={fmt_f(press_xy_err_norm[-1])}"
    )

    # Policy phase evolution
    if phase is not None:
        uniq = sorted(set(int(x) for x in phase.tolist()))
        phase_names = {0: "approach", 1: "align", 2: "press", 3: "retract_hold"}
        spans = []
        for p in uniq:
            idx = np.flatnonzero(phase == p)
            spans.append(f"{phase_names.get(p,p)}[{p}] {step[idx[0]]}->{step[idx[-1]]} ({len(idx)} steps)")
        lines.append("phase occupancy: " + "; ".join(spans))

        trans = []
        for i in range(1, T):
            if phase[i] != phase[i - 1]:
                trans.append(f"{step[i-1]}:{phase[i-1]}->{step[i]}:{phase[i]}")
        lines.append("phase transitions: " + (", ".join(trans) if trans else "none"))

        if phase_timer is not None:
            i_pt = int(np.argmax(phase_timer))
            lines.append(
                f"phase_timer max={fmt_f(phase_timer[i_pt])}@{step[i_pt]}; "
                f"first phase 1 at {fmt_idx(first_true(phase == 1))}; "
                f"first phase 2 at {fmt_idx(first_true(phase == 2))}; "
                f"first phase 3 at {fmt_idx(first_true(phase == 3))}"
            )

    # Keyframes: pick moments where image matters
    key_idxs = []

    def add_idx(i):
        if i is None:
            return
        i = int(i)
        if 0 <= i < T and i not in key_idxs:
            key_idxs.append(i)

    add_idx(0)
    add_idx(first_true(enter_align))
    add_idx(first_true(ready_to_press))
    if phase is not None:
        add_idx(first_true(phase == 2))
        add_idx(first_true(phase == 3))
    add_idx(i_best_press)
    add_idx(i_min_dist)
    add_idx(T - 1)

    # Fill to at least 5 with informative extras
    extras = [
        first_true(forward_ready),
        first_true(lateral_ready),
        i_best_lat,
        i_best_yaw,
    ]
    for i in extras:
        if len(key_idxs) >= 5:
            break
        add_idx(i)

    # Keep 5..10, ordered
    key_idxs = sorted(set(key_idxs))
    if len(key_idxs) > 10:
        # preserve endpoints and the most semantically useful earlier picks
        priority = []
        for cand in [0, first_true(enter_align), first_true(ready_to_press),
                     first_true(phase == 2) if phase is not None else None,
                     first_true(phase == 3) if phase is not None else None,
                     i_best_press, i_min_dist, T - 1]:
            if cand is not None and cand in key_idxs and cand not in priority:
                priority.append(cand)
        for i in key_idxs:
            if len(priority) >= 10:
                break
            if i not in priority:
                priority.append(i)
        key_idxs = sorted(priority[:10])

    return "\n".join(lines[:150]), key_idxs

# Task Description
The task is to control a humanoid to walk toward a button and push it with one of its end-effectors.
The button sits on one face of a pillar at a randomized position and height. It is mounted on either 
the +x or the -x face of the square pillar. The button should be pushed 0.02m inward. 
Pushing it sideways, or pushing on the pillar beside it, moves the button not at all.
Note that the button spawned area is reachable when standing. Squatting is not needed.

# Observation Description
Observation is a 2D tensor of shape (num_envs, obs_dim=28).
Observation for each environment consists of
- humanoid root position (x, y) in world frame. dim=2
- humanoid root orientation (yaw) in world frame, bounded in [-pi, pi]. (0 aligns to world +x, positive yaw rotates toward world +y, counterclockwise.) dim=1
- humanoid root velocity (linear velocity x, linear velocity y, angular velocity z) in local root frame. dim=3
- left end-effector position (x, y, z) in the pelvis-anchor frame. dim=3
- right end-effector position (x, y, z) in the pelvis-anchor frame. dim=3
- pelvis height in world frame. dim=1
- button face center position (x, y, z) in world frame: the center of the outward-facing surface
  to press, at its unpressed rest position. dim=3
- button face direction (dx, dy): 2D unit vector pointing from the pillar center toward the button,
  indicating which side of the pillar the button is on (approach from this direction). dim=2
- last action. dim=10
The x-axis of the local root frame is aligned with the humanoid's facing direction, the y-axis with its left, and the z-axis with the world frame's z-axis.

# Robot Description
The robot is a Unitree G1 humanoid: with two arms, two legs. 
Lengths below are in the pelvis frame (origin at the pelvis, x forward, y left, z up); 
the torso frame origin sits 0.05 m above it.
- Shoulders are 0.29 m above the pelvis and 0.14 m out to each side (0.28 m apart).
- Each arm is 0.185 m of upper arm (shoulder to elbow) plus 0.185 m of forearm (elbow
  to end-effector), so an end-effector never gets further than 0.37 m from its shoulder. 
- The humanoid is not a point mass. While travelling, treat it as a 0.5 m radius footprint
  and route around obstacles rather than through them. That footprint is a navigation rule only.
- The end-effector reaches ~0.37 m from the pelvis in x at maximum. Apart from 0.5m raidus when navigating,
  anything the robot wants to interact with the end-effector should be at least within 0.37m distance.
  If it is not, the humanoid can never reach the object with the end-effector.
- Reaching straight forward puts end-effector near (0.37, +-0.14, 0.29). Arms hanging at
  rest, the forearms point roughly down and the end-effectors sit near (0.0, +-0.22, -0.06).
- The commanded end-effector point is the center of the hand, and the hand is a capsule of
  roughly 0.035 m radius, not a point. To rest against a surface, command the hand center
  about 0.035 m outside it.

# Action Description
Action is a 2D torch tensor of shape (num_envs, action_dim=10).
Action for each environment consists of
- desired root linear velocity x in local root frame. min=-0.6, max=1.0. deadzone when |vx| < 0.15.
- desired root linear velocity y in local root frame. min=-0.5, max=0.5. deadzone when |vy| < 0.15.
- desired root orientation (yaw) in world frame, bounded in [-pi, pi]. min=-3.141592653589793, max=3.141592653589793. This is an ABSOLUTE target heading, not an angular velocity: hold it steady at the heading you want the robot to face, and it will turn toward it. (0 aligns to world +x, positive yaw rotates toward world +y, counterclockwise -- the same convention as the observed root orientation.) Wrap the shortest way around when the target crosses +-pi.
- left end-effector position x in the pelvis-anchor frame. min=-0.05, max=0.45
- left end-effector position y in the pelvis-anchor frame. min=0.05, max=0.45
- left end-effector position z in the pelvis-anchor frame. min=-0.15, max=0.25
- right end-effector position x in the pelvis-anchor frame. min=-0.05, max=0.45
- right end-effector position y in the pelvis-anchor frame. min=-0.45, max=-0.05
- right end-effector position z in the pelvis-anchor frame. min=-0.15, max=0.25
When walking (i.e. nonzero root velocity)
- pelvis height in world frame. min=0.75, max=0.75
When standing (i.e. zero root velocity)
- pelvis height in world frame. min=0.45, max=0.78
- when commanded in only-standing height, humanoid will freeze into standing regardless of what velocity is commanded.
Note:
The upper body always stay stright, when walking or standing.
Make sure the actions, especially end-effector and height, are smooth, not discrete.
This policy is called at 10 Hz: one call is 0.1 s, and each action is held until the next call.
Prefer deriving durations from seconds (e.g. HOLD_STEPS = int(1.5 / 0.1)) over writing raw frame counts.

Current policy code:
<policy_code>
```python
def policy_code(observation, policy_state):
    # observation layout
    # 0:2   root xy (world)
    # 2     root yaw
    # 3:6   root vel in local frame (vx, vy, wz)
    # 6:9   left ee pos in pelvis frame
    # 9:12  right ee pos in pelvis frame
    # 12    pelvis height
    # 13:16 button face center world xyz
    # 16:18 button face direction world xy
    # 18:28 last action

    dt = 0.1

    def wrap_to_pi(x):
        return (x + math.pi) % (2 * math.pi) - math.pi

    def clamp(x, lo, hi):
        return torch.clamp(x, lo, hi)

    def smooth_update(prev, target, rate):
        # first-order smoothing, rate in [0,1]
        return prev + rate * (target - prev)

    def world_to_local_xy(vec_world, yaw):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        x = c * vec_world[:, 0] + s * vec_world[:, 1]
        y = -s * vec_world[:, 0] + c * vec_world[:, 1]
        return torch.stack([x, y], dim=-1)

    if policy_state is None:
        num_envs = observation.shape[0]
        device = observation.device
        policy_state = {
            "needs_init": torch.ones(num_envs, dtype=torch.bool, device=device),
            "phase": torch.zeros(num_envs, dtype=torch.long, device=device),  # 0 approach, 1 align, 2 press, 3 hold/retract
            "phase_timer": torch.zeros(num_envs, device=device),
            "active_left": torch.ones(num_envs, dtype=torch.bool, device=device),
            "smoothed_vx": torch.zeros(num_envs, device=device),
            "smoothed_vy": torch.zeros(num_envs, device=device),
            "smoothed_yaw": torch.zeros(num_envs, device=device),
            "left_cmd": torch.zeros(num_envs, 3, device=device),
            "right_cmd": torch.zeros(num_envs, 3, device=device),
            "height_cmd": torch.full((num_envs,), 0.75, device=device),
        }

    reset_mask = policy_state["needs_init"]

    if torch.any(reset_mask):
        obs_r = observation[reset_mask]
        btn_dir_r = obs_r[:, 16:18]
        face_yaw_r = torch.atan2(-btn_dir_r[:, 1], -btn_dir_r[:, 0])
        btn_local_r = world_to_local_xy(obs_r[:, 13:15] - obs_r[:, 0:2], face_yaw_r)
        active_left_init = btn_local_r[:, 1] >= 0.0

        policy_state["phase"][reset_mask] = 0
        policy_state["phase_timer"][reset_mask] = 0.0
        policy_state["active_left"][reset_mask] = active_left_init
        policy_state["smoothed_vx"][reset_mask] = 0.0
        policy_state["smoothed_vy"][reset_mask] = 0.0
        policy_state["smoothed_yaw"][reset_mask] = obs_r[:, 2]
        policy_state["left_cmd"][reset_mask] = torch.tensor(
            [0.02, 0.22, -0.02], device=observation.device
        )
        policy_state["right_cmd"][reset_mask] = torch.tensor(
            [0.02, -0.22, -0.02], device=observation.device
        )
        policy_state["height_cmd"][reset_mask] = 0.75

    policy_state["needs_init"][reset_mask] = False

    num_envs = observation.shape[0]
    device = observation.device

    root_xy = observation[:, 0:2]
    root_yaw = observation[:, 2]
    left_ee = observation[:, 6:9]
    right_ee = observation[:, 9:12]
    button_pos = observation[:, 13:16]
    button_dir = observation[:, 16:18]
    active_left = policy_state["active_left"]
    side_sign = torch.where(active_left, torch.ones_like(root_yaw), -torch.ones_like(root_yaw))

    # Desired standing position: offset from button along outward normal,
    # with additional safety margin for navigation around pillar.
    # Need root center outside obstacle and wrist reachability.
    # Move closer so the wrist can actually reach the button while still staying outside the pillar.
    stand_offset = 0.40
    tangent_dir = torch.stack([-button_dir[:, 1], button_dir[:, 0]], dim=-1)
    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.09 * side_sign).unsqueeze(-1)

    to_stand_world = stand_xy - root_xy
    dist_to_stand = torch.norm(to_stand_world, dim=-1) + 1e-6
    to_stand_local = world_to_local_xy(to_stand_world, root_yaw)

    # Face toward the button surface (toward pillar), i.e. opposite the face outward direction
    face_yaw = torch.atan2(-button_dir[:, 1], -button_dir[:, 0])

    yaw_err = wrap_to_pi(face_yaw - root_yaw)

    # Hand target on the button surface:
    # hand center should be ~0.035 m outside the face at contact;
    # pressing 0.02 inward means center target moves 0.02 inward from there.
    contact_outside = 0.035
    push_depth = 0.02
    # Add extra inward margin because the realized hand position tends to undershoot
    # the commanded target by a few centimeters on hardware.
    extra_press_margin = 0.030
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - extra_press_margin)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.045)

    # Convert target points to local pelvis frame XY approximately using root yaw and root position.
    press_local_xy = world_to_local_xy(press_world_xy - root_xy, root_yaw)
    prepress_local_xy = world_to_local_xy(prepress_world_xy - root_xy, root_yaw)

    # Vertical target around button height minus pelvis height, with torso offset approximation.
    # Shoulder is at pelvis +0.29; hand best around button z relative to pelvis.
    raw_local_z = button_pos[:, 2] - 0.72
    target_z = clamp(raw_local_z, -0.02, 0.20)

    # Phase logic
    phase = policy_state["phase"]
    phase_timer = policy_state["phase_timer"] + dt

    # Conditions
    near_stand = dist_to_stand < 0.24
    aligned = torch.abs(yaw_err) < 0.32
    button_local_now = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
    forward_ready = button_local_now[:, 0] < 0.40
    hand_lateral_err = torch.where(active_left, button_local_now[:, 1] - 0.14, button_local_now[:, 1] + 0.14)
    lateral_ready = torch.abs(hand_lateral_err) < 0.07
    enter_align = (dist_to_stand < 0.30) & aligned
    ready_to_press = near_stand & aligned & lateral_ready & forward_ready

    # Phase transitions
    phase = torch.where((phase == 0) & enter_align, torch.ones_like(phase), phase)
    phase_timer = torch.where((policy_state["phase"] == 0) & enter_align, torch.zeros_like(phase_timer), phase_timer)

    # align -> press once stance is good enough, with a short minimum settle time
    good_enough_press = (dist_to_stand < 0.22) & (torch.abs(yaw_err) < 0.30) & (torch.abs(hand_lateral_err) < 0.08) & (button_local_now[:, 0] < 0.395)
    phase = torch.where((phase == 1) & (phase_timer > 0.2) & good_enough_press, torch.full_like(phase, 2), phase)
    phase_timer = torch.where((policy_state["phase"] == 1) & (phase_timer > 0.2) & good_enough_press, torch.zeros_like(phase_timer), phase_timer)

    press_steps = int(1.4 / dt)
    retract_steps = int(0.6 / dt)

    phase = torch.where((phase == 2) & (phase_timer > press_steps * dt), torch.full_like(phase, 3), phase)
    phase_timer = torch.where((policy_state["phase"] == 2) & (phase_timer > press_steps * dt), torch.zeros_like(phase_timer), phase_timer)

    # If after retract/hold, still not well placed, go back to approach; otherwise remain in hold posture
    go_back_approach = (phase == 3) & (phase_timer > retract_steps * dt) & (~ready_to_press)
    phase = torch.where(go_back_approach, torch.zeros_like(phase), phase)
    phase_timer = torch.where(go_back_approach, torch.zeros_like(phase_timer), phase_timer)

    policy_state["phase"] = phase
    policy_state["phase_timer"] = phase_timer

    # Base motion command
    target_vx = torch.zeros(num_envs, device=device)
    target_vy = torch.zeros(num_envs, device=device)

    moving_mask = phase == 0

    # Approach with local velocity toward stand point
    target_vx[moving_mask] = clamp(1.6 * to_stand_local[moving_mask, 0], -0.6, 1.0)
    target_vy[moving_mask] = clamp(2.4 * to_stand_local[moving_mask, 1], -0.5, 0.5)

    # During align, keep making small stance corrections instead of freezing too early.
    align_move_mask = phase == 1
    target_vx[align_move_mask] = clamp(1.0 * to_stand_local[align_move_mask, 0], -0.25, 0.25)
    target_vy[align_move_mask] = clamp(
        1.2 * to_stand_local[align_move_mask, 1] + 1.6 * hand_lateral_err[align_move_mask],
        -0.20, 0.20
    )

    # Slow down when yaw error is large
    yaw_scale = clamp(1.0 - torch.abs(yaw_err) / 1.8, 0.5, 1.0)
    target_vx = target_vx * yaw_scale
    target_vy = target_vy * yaw_scale

    # Stop only when actually ready to press, or while executing press/retract.
    stop_mask = ready_to_press | (phase >= 2)
    target_vx[stop_mask] = 0.0
    target_vy[stop_mask] = 0.0

    # Smooth base commands
    smoothed_vx = smooth_update(policy_state["smoothed_vx"], target_vx, 0.6)
    smoothed_vy = smooth_update(policy_state["smoothed_vy"], target_vy, 0.6)

    # Apply deadzone intentionally: if small, set to zero
    smoothed_vx = torch.where(torch.abs(smoothed_vx) < 0.08, torch.zeros_like(smoothed_vx), smoothed_vx)
    smoothed_vy = torch.where(torch.abs(smoothed_vy) < 0.08, torch.zeros_like(smoothed_vy), smoothed_vy)

    smoothed_yaw = wrap_to_pi(smooth_update(policy_state["smoothed_yaw"], face_yaw, 0.35))

    # Height: walking fixed 0.75, standing a bit lower for stable press
    moving_now = (torch.abs(smoothed_vx) >= 0.15) | (torch.abs(smoothed_vy) >= 0.15)
    target_height = torch.where(moving_now, torch.full_like(root_yaw, 0.75), torch.full_like(root_yaw, 0.72))
    target_height = torch.where(phase == 2, torch.full_like(root_yaw, 0.70), target_height)
    target_height = torch.where(phase == 3, torch.full_like(root_yaw, 0.72), target_height)
    height_cmd = smooth_update(policy_state["height_cmd"], target_height, 0.25)
    height_cmd = torch.where(moving_now, torch.full_like(height_cmd, 0.75), clamp(height_cmd, 0.45, 0.78))

    # Arm posture defaults
    left_target = torch.stack([
        torch.full((num_envs,), 0.02, device=device),
        torch.full((num_envs,), 0.22, device=device),
        torch.full((num_envs,), -0.02, device=device),
    ], dim=-1)
    right_target = torch.stack([
        torch.full((num_envs,), 0.02, device=device),
        torch.full((num_envs,), -0.22, device=device),
        torch.full((num_envs,), -0.02, device=device),
    ], dim=-1)

    # Support arm slightly out of the way
    support_left = torch.stack([
        torch.full((num_envs,), 0.05, device=device),
        torch.full((num_envs,), 0.18, device=device),
        torch.full((num_envs,), 0.02, device=device),
    ], dim=-1)
    support_right = torch.stack([
        torch.full((num_envs,), 0.05, device=device),
        torch.full((num_envs,), -0.18, device=device),
        torch.full((num_envs,), 0.02, device=device),
    ], dim=-1)

    # Active hand targets
    active_pre_x = clamp(prepress_local_xy[:, 0], 0.18, 0.42)
    active_pre_y_left = clamp(prepress_local_xy[:, 1], 0.08, 0.34)
    active_pre_y_right = clamp(prepress_local_xy[:, 1], -0.34, -0.08)

    active_press_x = clamp(press_local_xy[:, 0], 0.22, 0.45)
    active_press_y_left = clamp(press_local_xy[:, 1], 0.08, 0.34)
    active_press_y_right = clamp(press_local_xy[:, 1], -0.34, -0.08)

    pre_left = torch.stack([active_pre_x, active_pre_y_left, target_z], dim=-1)
    pre_right = torch.stack([active_pre_x, active_pre_y_right, target_z], dim=-1)
    press_left = torch.stack([active_press_x, active_press_y_left, target_z], dim=-1)
    press_right = torch.stack([active_press_x, active_press_y_right, target_z], dim=-1)

    # By phase:
    # 0 walking: keep both arms relaxed
    # 1 align: move active arm to prepress
    # 2 press: move active arm to press target
    # 3 retract/hold: back to prepress
    align_mask = phase == 1
    press_mask = phase == 2
    retract_mask = phase == 3

    left_active = policy_state["active_left"]
    right_active = ~left_active

    # Assign support arm always during manipulation phases
    manip_mask = align_mask | press_mask | retract_mask
    left_target = torch.where(manip_mask.unsqueeze(-1), support_left, left_target)
    right_target = torch.where(manip_mask.unsqueeze(-1), support_right, right_target)

    # Active arm commands
    left_target = torch.where((align_mask & left_active).unsqueeze(-1), pre_left, left_target)
    left_target = torch.where((press_mask & left_active).unsqueeze(-1), press_left, left_target)
    left_target = torch.where((retract_mask & left_active).unsqueeze(-1), pre_left, left_target)

    right_target = torch.where((align_mask & right_active).unsqueeze(-1), pre_right, right_target)
    right_target = torch.where((press_mask & right_active).unsqueeze(-1), press_right, right_target)
    right_target = torch.where((retract_mask & right_active).unsqueeze(-1), pre_right, right_target)

    # Smooth arm commands
    left_cmd = smooth_update(policy_state["left_cmd"], left_target, 0.5)
    right_cmd = smooth_update(policy_state["right_cmd"], right_target, 0.5)

    # Clamp to action limits
    left_cmd[:, 0] = clamp(left_cmd[:, 0], -0.05, 0.45)
    left_cmd[:, 1] = clamp(left_cmd[:, 1], 0.05, 0.45)
    left_cmd[:, 2] = clamp(left_cmd[:, 2], -0.15, 0.25)

    right_cmd[:, 0] = clamp(right_cmd[:, 0], -0.05, 0.45)
    right_cmd[:, 1] = clamp(right_cmd[:, 1], -0.45, -0.05)
    right_cmd[:, 2] = clamp(right_cmd[:, 2], -0.15, 0.25)

    action = torch.zeros(num_envs, 10, device=device)
    action[:, 0] = clamp(smoothed_vx, -0.6, 1.0)
    action[:, 1] = clamp(smoothed_vy, -0.5, 0.5)
    action[:, 2] = wrap_to_pi(smoothed_yaw)
    action[:, 3:6] = left_cmd
    action[:, 6:9] = right_cmd
    action[:, 9] = height_cmd

    policy_state["smoothed_vx"] = smoothed_vx
    policy_state["smoothed_vy"] = smoothed_vy
    policy_state["smoothed_yaw"] = smoothed_yaw
    policy_state["left_cmd"] = left_cmd
    policy_state["right_cmd"] = right_cmd
    policy_state["height_cmd"] = height_cmd

    return action, policy_state
```
</policy_code>

# Rollout analysis
Computed over the full episode by a function written for this policy code.
<analysis>
episode: T=540 steps (10.80s), policy updates every 5 steps
button face center start=(-0.594, 0.633, 0.956), face dir=(1.000, 0.000), chosen arm=right
stand target distance: start=1.205, min=0.016@412, max=1.205@0, end=0.049
first near stand (<0.30) at 107; first near_stand (<0.24) at 113; last near_stand at 539
yaw-to-face error |rad|: start=0.147, best=0.000@202, worst=1.586@191, end=0.177
first aligned (|yaw_err|<0.32) at 0; first enter_align at 107
button local x: start=0.550, min=0.275@21, end=0.343; lateral hand-line err: start=-1.201, best_abs=0.000@439, end=0.015
first forward_ready (x<0.40) at 16; first lateral_ready (|lat err|<0.07) at 118; first ready_to_press at 277; first good_enough_press at 278
commanded planar speed: max=0.568@2, first zero-command at 133, last nonzero-command at 277
observed planar speed: start=0.804, max=0.940@3, end=0.015
pelvis height: start=0.758, min=0.728@47, max=0.804@303, end=0.759
active hand pos in pelvis frame: x start/end=0.042/0.264, x max=0.345@220; y start/end=-0.237/-0.118; z start/end=-0.034/0.218, z max=0.333@223
active-hand XY error to prepress target: min=0.000@522, end=0.006; to press target: min=0.023@302, end=0.094
phase occupancy: approach[0] 0->110 (111 steps); align[1] 111->280 (170 steps); press[2] 281->345 (65 steps); retract_hold[3] 346->539 (194 steps)
phase transitions: 110:0->111:1, 280:1->281:2, 345:2->346:3
phase_timer max=4.200@106; first phase 1 at 111; first phase 2 at 281; first phase 3 at 346
</analysis>

# Frames
The moments the analysis selected, in order.

Frame 0 (step 0)

[Frame 0 image: frame_0.png]

Trajectory info:
step=0
observation: (Look at observation prompt for details)
  robot observation = [0.1464, -0.6138, -2.9945, 0.2931, -0.7486, -0.0576, 0.0237, 0.2216, -0.0137, 0.0417, -0.2366, -0.0345, 0.7576]
  task-specific observation = [-0.5939, 0.6326, 0.9563, 1.0000, 0.0000]
  last action = [0.3227, -0.4661, -3.1415, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
policy_state:
  active_left = False
  height_cmd = 0.7500
  left_cmd = [0.0200, 0.2200, -0.0200]
  phase = 0
  phase_timer = 2.0000
  right_cmd = [0.0200, -0.2200, -0.0200]
  smoothed_vx = 0.3231
  smoothed_vy = -0.4666
  smoothed_yaw = -3.1415

Frame 1 (step 107)

[Frame 1 image: frame_1.png]

Trajectory info:
step=107
observation: (Look at observation prompt for details)
  robot observation = [-0.0615, 0.2688, -3.0263, 0.1012, -0.5400, -0.1693, 0.0283, 0.2259, -0.0346, 0.0350, -0.2300, -0.0057, 0.7631]
  task-specific observation = [-0.6010, 0.6232, 0.9398, 1.0000, 0.0000]
  last action = [0.0926, -0.4586, -3.1416, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
policy_state:
  active_left = False
  height_cmd = 0.7500
  left_cmd = [0.0200, 0.2200, -0.0200]
  phase = 0
  phase_timer = 4.2000
  right_cmd = [0.0200, -0.2200, -0.0200]
  smoothed_vx = 0.1349
  smoothed_vy = -0.4636
  smoothed_yaw = -3.1416

Frame 2 (step 192)

[Frame 2 image: frame_2.png]

Trajectory info:
step=192
observation: (Look at observation prompt for details)
  robot observation = [-0.0565, 0.7207, 3.1232, -0.1225, 0.2475, -0.5236, 0.0372, 0.1631, 0.0422, 0.3287, -0.0703, 0.3103, 0.7979]
  task-specific observation = [-0.6112, 0.6721, 0.9334, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -2.5918, 0.0500, 0.1800, 0.0200, 0.4194, -0.0801, 0.2000, 0.7425]
policy_state:
  active_left = False
  height_cmd = 0.7369
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 1
  phase_timer = 1.6000
  right_cmd = [0.4197, -0.0800, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -2.2345

Frame 3 (step 277)

[Frame 3 image: frame_3.png]

Trajectory info:
step=277
observation: (Look at observation prompt for details)
  robot observation = [-0.2042, 0.6506, -3.0021, 0.4601, -0.3570, -0.2827, 0.0548, 0.1762, 0.0511, 0.3222, -0.2340, 0.3077, 0.7855]
  task-specific observation = [-0.6139, 0.6928, 0.9473, 1.0000, 0.0000]
  last action = [0.0809, -0.1058, -3.1359, 0.0500, 0.1800, 0.0200, 0.3822, -0.2095, 0.2000, 0.7425]
policy_state:
  active_left = False
  height_cmd = 0.7369
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 1
  phase_timer = 3.3000
  right_cmd = [0.3551, -0.1534, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1379

Frame 4 (step 281)

[Frame 4 image: frame_4.png]

Trajectory info:
step=281
observation: (Look at observation prompt for details)
  robot observation = [-0.2361, 0.6703, -3.0249, 0.3353, -0.2438, 0.0175, 0.0604, 0.1799, 0.0481, 0.3105, -0.1976, 0.3038, 0.7909]
  task-specific observation = [-0.6151, 0.6952, 0.9463, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1379, 0.0500, 0.1800, 0.0200, 0.3551, -0.1534, 0.2000, 0.7369]
policy_state:
  active_left = False
  height_cmd = 0.7277
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 2
  phase_timer = 0.0000
  right_cmd = [0.3718, -0.1167, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1392

Frame 5 (step 302)

[Frame 5 image: frame_5.png]

Trajectory info:
step=302
observation: (Look at observation prompt for details)
  robot observation = [-0.3136, 0.6548, -2.9911, 0.1269, 0.1080, -0.1885, 0.0591, 0.1882, 0.0369, 0.2870, -0.0921, 0.3113, 0.8040]
  task-specific observation = [-0.6142, 0.6877, 0.9714, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1409, 0.0500, 0.1800, 0.0200, 0.3390, -0.0846, 0.2000, 0.7117]
policy_state:
  active_left = False
  height_cmd = 0.7088
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 2
  phase_timer = 0.4000
  right_cmd = [0.3241, -0.0823, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1412

Frame 6 (step 346)

[Frame 6 image: frame_6.png]

Trajectory info:
step=346
observation: (Look at observation prompt for details)
  robot observation = [-0.2383, 0.6136, -2.9929, -0.0953, 0.0030, -0.0471, 0.0654, 0.1793, 0.0300, 0.2989, -0.1068, 0.2752, 0.7746]
  task-specific observation = [-0.6164, 0.6801, 0.9430, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3566, -0.1153, 0.2000, 0.7009]
policy_state:
  active_left = False
  height_cmd = 0.7057
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 3
  phase_timer = 0.0000
  right_cmd = [0.3208, -0.1126, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1416

Frame 7 (step 412)

[Frame 7 image: frame_7.png]

Trajectory info:
step=412
observation: (Look at observation prompt for details)
  robot observation = [-0.2198, 0.6072, -2.9963, -0.0431, 0.0159, 0.0349, 0.0556, 0.1778, 0.0227, 0.2741, -0.1062, 0.2471, 0.7570]
  task-specific observation = [-0.6180, 0.6810, 0.9437, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3010, -0.1122, 0.2000, 0.7195]
policy_state:
  active_left = False
  height_cmd = 0.7197
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 3
  phase_timer = 1.3000
  right_cmd = [0.3022, -0.1156, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1416

Frame 8 (step 475)

[Frame 8 image: frame_8.png]

Trajectory info:
step=475
observation: (Look at observation prompt for details)
  robot observation = [-0.2347, 0.6074, -2.9469, 0.0230, -0.0164, -0.0192, 0.0594, 0.1762, 0.0190, 0.2741, -0.1174, 0.2429, 0.7562]
  task-specific observation = [-0.6107, 0.6775, 0.9640, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2800, -0.1269, 0.2000, 0.7200]
policy_state:
  active_left = False
  height_cmd = 0.7200
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 3
  phase_timer = 2.5000
  right_cmd = [0.2800, -0.1269, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1416

Frame 9 (step 539)

[Frame 9 image: frame_9.png]

Trajectory info:
step=539
observation: (Look at observation prompt for details)
  robot observation = [-0.2548, 0.6184, -2.9650, 0.0127, 0.0088, -0.0105, 0.0597, 0.1738, 0.0176, 0.2637, -0.1178, 0.2182, 0.7592]
  task-specific observation = [-0.6141, 0.6817, 0.9432, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2643, -0.1128, 0.2000, 0.7200]
policy_state:
  active_left = False
  height_cmd = 0.7200
  left_cmd = [0.0500, 0.1800, 0.0200]
  phase = 3
  phase_timer = 3.8000
  right_cmd = [0.2643, -0.1128, 0.2000]
  smoothed_vx = 0.0000
  smoothed_vy = 0.0000
  smoothed_yaw = -3.1416

# Instructions
Analyse this rollout in two parts:
1. Write a detailed step-by-step analysis in <reasoning></reasoning> tags. The rollout analysis
above is the primary evidence and the frames are there to confirm or contradict it; do not assert
anything about the rollout that is not supported by a value in the analysis or visible in a frame.
Explain what the robot is doing at each stage and why it succeeds or fails.
2. Write a concise summary (~150-200 words) in <summary></summary> tags covering: (a) the intended
strategy of the policy, (b) what actually happened in this rollout, and (c) the root cause of failure.
Do not generate new policy code.
The given rollout consists of one episode, no need to consider if the reset has been done for next episode or not.

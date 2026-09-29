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

# Episode overview
The episode is 540 steps long, recorded at 50 Hz. Below are 20 evenly spaced samples of the observation, for a coarse sense of how it went.
<overview>
step=0
  robot observation = [0.1464, -0.6138, -2.9945, 0.2931, -0.7486, -0.0576, 0.0237, 0.2216, -0.0137, 0.0417, -0.2366, -0.0345, 0.7576]
  task-specific observation = [-0.5939, 0.6326, 0.9563, 1.0000, 0.0000]
  last action = [0.3227, -0.4661, -3.1415, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=28
  robot observation = [-0.0887, -0.3300, -3.0217, 0.1639, -0.3011, -1.8692, 0.0132, 0.2185, -0.0085, 0.0544, -0.2332, -0.0253, 0.7763]
  task-specific observation = [-0.5970, 0.6195, 0.9545, 1.0000, 0.0000]
  last action = [0.0000, -0.4428, -3.1416, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=57
  robot observation = [-0.1986, 0.0249, -3.0349, -0.0273, -0.3878, 0.2042, 0.0195, 0.2360, -0.0199, 0.0129, -0.2263, -0.0367, 0.7477]
  task-specific observation = [-0.6020, 0.6188, 0.9278, 1.0000, 0.0000]
  last action = [0.0000, -0.4919, -3.1416, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=85
  robot observation = [-0.1199, 0.1437, -2.9673, -0.2530, -0.2946, -0.5463, 0.0220, 0.2269, -0.0343, 0.0070, -0.2327, -0.0195, 0.7354]
  task-specific observation = [-0.6012, 0.6180, 0.9414, 1.0000, 0.0000]
  last action = [0.0000, -0.4596, -3.1416, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=113
  robot observation = [-0.0854, 0.3396, -3.0134, 0.1961, -0.6840, -0.4520, 0.0404, 0.2211, -0.0347, 0.0411, -0.2345, 0.0141, 0.7674]
  task-specific observation = [-0.6267, 0.6165, 0.8933, 1.0000, 0.0000]
  last action = [0.1075, -0.2968, -3.1416, 0.0350, 0.2000, 0.0000, 0.2109, -0.2800, 0.0900, 0.7500]
step=142
  robot observation = [-0.1044, 0.6300, -2.9194, -0.2240, -0.1720, 0.7592, 0.0573, 0.1850, 0.0179, 0.2975, -0.1296, 0.2970, 0.7724]
  task-specific observation = [-0.5994, 0.6456, 0.9400, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0495, 0.1806, 0.0194, 0.3873, -0.1285, 0.1958, 0.7327]
step=170
  robot observation = [-0.1136, 0.7888, -3.0719, -0.1149, 0.0329, -0.1344, 0.0491, 0.1747, 0.0431, 0.3143, -0.1036, 0.3209, 0.7977]
  task-specific observation = [-0.6048, 0.6767, 0.9327, 1.0000, 0.0000]
  last action = [0.0000, 0.1861, -3.1416, 0.0500, 0.1800, 0.0200, 0.4106, -0.0812, 0.1999, 0.7500]
step=199
  robot observation = [-0.0434, 0.6887, 3.1135, -0.1341, 0.1942, -0.0960, 0.0315, 0.1632, 0.0378, 0.3275, -0.0778, 0.3051, 0.7912]
  task-specific observation = [-0.6095, 0.6713, 0.9368, 1.0000, 0.0000]
  last action = [0.0932, 0.1185, -2.5520, 0.0500, 0.1800, 0.0200, 0.4199, -0.0800, 0.2000, 0.7327]
step=227
  robot observation = [0.0181, 0.5347, -2.6895, -0.0024, 0.1399, -1.0524, 0.0644, 0.1767, 0.0226, 0.3179, -0.2227, 0.3238, 0.7798]
  task-specific observation = [-0.6057, 0.6613, 0.9564, 1.0000, 0.0000]
  last action = [0.1395, -0.0868, -2.7158, 0.0500, 0.1800, 0.0200, 0.4200, -0.2899, 0.2000, 0.7369]
step=255
  robot observation = [0.0067, 0.5314, -2.8240, 0.1653, -0.2953, 0.1100, 0.0569, 0.1832, 0.0374, 0.2936, -0.3091, 0.3239, 0.7886]
  task-specific observation = [-0.6069, 0.6659, 0.9425, 1.0000, 0.0000]
  last action = [0.1563, -0.1603, -3.1095, 0.0500, 0.1800, 0.0200, 0.4200, -0.3283, 0.2000, 0.7500]
step=284
  robot observation = [-0.2557, 0.6805, -3.0170, 0.2651, -0.1714, -0.0471, 0.0608, 0.1864, 0.0429, 0.3046, -0.1693, 0.2971, 0.7936]
  task-specific observation = [-0.6155, 0.6963, 0.9466, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1392, 0.0500, 0.1800, 0.0200, 0.3718, -0.1167, 0.2000, 0.7277]
step=312
  robot observation = [-0.3130, 0.6384, -3.0068, -0.0451, 0.0935, 0.1274, 0.0682, 0.1870, 0.0316, 0.2841, -0.0836, 0.2942, 0.8014]
  task-specific observation = [-0.6159, 0.6796, 0.9611, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1413, 0.0500, 0.1800, 0.0200, 0.3156, -0.0811, 0.2000, 0.7066]
step=340
  robot observation = [-0.2521, 0.6102, -3.0078, -0.0789, 0.0594, 0.1606, 0.0654, 0.1804, 0.0320, 0.2922, -0.0988, 0.2768, 0.7788]
  task-specific observation = [-0.6155, 0.6803, 0.9441, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3429, -0.1088, 0.2000, 0.7012]
step=369
  robot observation = [-0.2239, 0.6176, -3.0007, -0.0131, -0.0086, -0.0017, 0.0645, 0.1778, 0.0242, 0.2741, -0.1047, 0.2515, 0.7636]
  task-specific observation = [-0.6191, 0.6813, 0.9437, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3012, -0.1084, 0.2000, 0.7155]
step=397
  robot observation = [-0.2245, 0.6106, -2.9999, -0.0164, 0.0248, -0.0541, 0.0589, 0.1775, 0.0211, 0.2726, -0.1069, 0.2539, 0.7612]
  task-specific observation = [-0.6163, 0.6782, 0.9678, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3011, -0.1113, 0.2000, 0.7189]
step=426
  robot observation = [-0.2140, 0.6105, -2.9855, 0.0106, 0.0092, 0.0297, 0.0533, 0.1767, 0.0228, 0.2767, -0.1090, 0.2455, 0.7571]
  task-specific observation = [-0.6155, 0.6804, 0.9428, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.3061, -0.1169, 0.2000, 0.7198]
step=454
  robot observation = [-0.2308, 0.6038, -2.9480, 0.0139, 0.0075, -0.0209, 0.0571, 0.1776, 0.0194, 0.2840, -0.1146, 0.2428, 0.7556]
  task-specific observation = [-0.6148, 0.6782, 0.9405, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2888, -0.1310, 0.2000, 0.7200]
step=482
  robot observation = [-0.2354, 0.6074, -2.9467, 0.0015, -0.0034, -0.0401, 0.0599, 0.1762, 0.0188, 0.2723, -0.1181, 0.2430, 0.7565]
  task-specific observation = [-0.6106, 0.6790, 0.9649, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2782, -0.1266, 0.2000, 0.7200]
step=511
  robot observation = [-0.2542, 0.6107, -2.9595, -0.1957, 0.0146, -0.0087, 0.0601, 0.1755, 0.0167, 0.2633, -0.1163, 0.2335, 0.7582]
  task-specific observation = [-0.6124, 0.6805, 0.9659, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2622, -0.1235, 0.2000, 0.7200]
step=539
  robot observation = [-0.2548, 0.6184, -2.9650, 0.0127, 0.0088, -0.0105, 0.0597, 0.1738, 0.0176, 0.2637, -0.1178, 0.2182, 0.7592]
  task-specific observation = [-0.6141, 0.6817, 0.9432, 1.0000, 0.0000]
  last action = [0.0000, 0.0000, -3.1416, 0.0500, 0.1800, 0.0200, 0.2643, -0.1128, 0.2000, 0.7200]
</overview>

# Instructions
The goal is to analyze where and how the policy code failed or succeeded to finish the task.
There is two steps in this evaluation pipeline.
First step: With a brief information about the rollout, writes a function that computes the number for the rollout to understand in detail how the episode went.
Second step: Reads that output together with the video frames at the keyframes your function picked, and writes the evaluation of the policy code.

Your job is the first step.
Write a function that computes numbers from this recorded rollout, so that what happened in
the episode can be understood from its output alone. Do not diagnose the episode here.

Return exactly one fenced python block defining:

    def analyze(traj) -> tuple[str, list[int]]

`traj` is the whole episode for a single environment, as numpy arrays:
  - traj["step"]: (T,) int
  - traj["observation"]: (T, obs_dim) float, laid out exactly as the observation prompt describes
  - traj["policy_state"][key]: (T,) or (T, d), the state the policy code carries between steps
numpy and math are available. No file or network access.
One recorded step is 0.02 s; the policy is evaluated once every 5 steps.

The returned string is what the second step reads in place of the trajectory itself, so it has
to stand on its own. Keep it under 150 lines. Never print per-step rows, but every extremum and
every threshold crossing must carry the step it happened at.

The returned indices are positions into the arrays (0 <= i < T), at minimum 5, at most 10 of them. 
The second step is shown the video frame at each one, so choose the moments where an image tells it
something the numbers cannot -- contact, posture, how the limbs are placed relative to the
objects -- rather than the moments whose values you already reported.

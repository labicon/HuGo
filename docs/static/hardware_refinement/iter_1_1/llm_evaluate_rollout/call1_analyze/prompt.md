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
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth + 0.005)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.035)

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

    press_steps = int(1.0 / dt)
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
    active_pre_x = clamp(prepress_local_xy[:, 0], 0.18, 0.41)
    active_pre_y_left = clamp(prepress_local_xy[:, 1], 0.08, 0.34)
    active_pre_y_right = clamp(prepress_local_xy[:, 1], -0.34, -0.08)

    active_press_x = clamp(press_local_xy[:, 0], 0.20, 0.42)
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
The episode is 148 steps long, recorded at 50 Hz. Below are 20 evenly spaced samples of the observation, for a coarse sense of how it went.
<overview>
step=0
  robot observation = [0.3804, 0.6948, -0.0721, 0.4509, -0.4555, 0.5969, 0.0559, 0.2248, 0.0282, 0.0249, -0.2175, -0.0771, 0.7913]
  task-specific observation = [0.8026, 0.1398, 0.9568, -1.0000, 0.0000]
  last action = [0.2558, -0.4658, 0.0000, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=8
  robot observation = [0.4725, 0.5748, 0.0201, 0.4347, -0.7100, 0.5428, 0.0231, 0.2418, 0.0144, 0.0581, -0.2204, -0.0526, 0.7691]
  task-specific observation = [0.8078, 0.1418, 0.9575, -1.0000, 0.0000]
  last action = [0.0000, -0.4884, 0.0000, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=15
  robot observation = [0.5308, 0.4949, 0.1054, 0.3132, -0.4830, 0.3368, 0.0268, 0.2366, 0.0011, 0.0524, -0.2266, -0.0362, 0.7656]
  task-specific observation = [0.8122, 0.1410, 0.9578, -1.0000, 0.0000]
  last action = [-0.1049, -0.4809, 0.0000, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=23
  robot observation = [0.5841, 0.4317, 0.0626, 0.3220, -0.4223, -0.3299, 0.0612, 0.2195, 0.0373, 0.0853, -0.2959, 0.1025, 0.7822]
  task-specific observation = [0.8167, 0.1399, 0.9581, -1.0000, 0.0000]
  last action = [-0.1471, -0.2385, 0.0000, 0.0425, 0.1900, 0.0100, 0.1400, -0.2978, 0.1450, 0.7500]
step=31
  robot observation = [0.5962, 0.3737, -0.0791, 0.0802, -0.4821, -1.2584, 0.0652, 0.1830, 0.0362, 0.1144, -0.3012, 0.1772, 0.7624]
  task-specific observation = [0.8204, 0.1374, 0.9588, -1.0000, 0.0000]
  last action = [-0.1642, -0.2140, 0.0000, 0.0463, 0.1850, 0.0150, 0.1600, -0.2815, 0.1725, 0.7500]
step=39
  robot observation = [0.5785, 0.3348, -0.0501, -0.2139, -0.1941, 0.1117, 0.0631, 0.1901, 0.0267, 0.1390, -0.2704, 0.1983, 0.7521]
  task-specific observation = [0.8242, 0.1245, 0.9551, -1.0000, 0.0000]
  last action = [0.0000, -0.0801, 0.0000, 0.0491, 0.1813, 0.0188, 0.2003, -0.2262, 0.1931, 0.7375]
step=46
  robot observation = [0.5240, 0.3026, -0.0486, -0.4755, -0.1087, 0.8692, 0.0524, 0.1969, 0.0261, 0.1643, -0.2330, 0.2302, 0.7697]
  task-specific observation = [0.8257, 0.1144, 0.9541, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0495, 0.1806, 0.0194, 0.2265, -0.2076, 0.1966, 0.7281]
step=54
  robot observation = [0.4899, 0.2735, 0.0645, -0.1640, -0.2476, 0.4346, 0.0390, 0.1856, 0.0373, 0.2021, -0.2243, 0.2389, 0.7809]
  task-specific observation = [0.8237, 0.1069, 0.9575, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0499, 0.1802, 0.0198, 0.2785, -0.1904, 0.1991, 0.7158]
step=62
  robot observation = [0.4771, 0.2215, 0.0731, -0.1365, -0.3463, -0.0192, 0.0334, 0.1849, 0.0430, 0.2343, -0.2141, 0.2620, 0.7863]
  task-specific observation = [0.8205, 0.0982, 0.9603, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0499, 0.1801, 0.0199, 0.2910, -0.1842, 0.1996, 0.7119]
step=70
  robot observation = [0.4601, 0.2045, 0.1276, -0.1119, 0.0062, 0.2758, 0.0425, 0.1827, 0.0462, 0.2539, -0.1999, 0.2727, 0.7849]
  task-specific observation = [0.8188, 0.0883, 0.9610, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3097, -0.1558, 0.1999, 0.7067]
step=77
  robot observation = [0.4473, 0.2083, 0.1446, -0.1318, 0.0778, -0.0175, 0.0442, 0.1808, 0.0441, 0.2722, -0.1806, 0.2654, 0.7798]
  task-specific observation = [0.8190, 0.0812, 0.9615, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3158, -0.1585, 0.1999, 0.7050]
step=85
  robot observation = [0.4298, 0.2197, 0.1403, -0.0897, 0.1485, -0.1204, 0.0532, 0.1777, 0.0379, 0.2995, -0.1783, 0.2476, 0.7663]
  task-specific observation = [0.8201, 0.0771, 0.9628, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3288, -0.1765, 0.2000, 0.7028]
step=93
  robot observation = [0.4176, 0.2257, 0.1336, -0.0647, 0.0149, -0.1920, 0.0547, 0.1760, 0.0327, 0.3145, -0.1789, 0.2320, 0.7509]
  task-specific observation = [0.8179, 0.0782, 0.9653, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3090, -0.1870, 0.2000, 0.7103]
step=101
  robot observation = [0.4108, 0.2284, 0.1416, -0.1003, 0.0016, -0.0035, 0.0580, 0.1766, 0.0310, 0.3195, -0.1846, 0.2225, 0.7435]
  task-specific observation = [0.8135, 0.0779, 0.9669, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3078, -0.1900, 0.2000, 0.7128]
step=108
  robot observation = [0.4081, 0.2298, 0.1447, -0.0165, -0.0016, 0.0559, 0.0592, 0.1766, 0.0316, 0.3203, -0.1866, 0.2229, 0.7405]
  task-specific observation = [0.8114, 0.0764, 0.9656, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3075, -0.1966, 0.2000, 0.7159]
step=116
  robot observation = [0.4099, 0.2259, 0.1428, 0.0172, -0.0313, -0.1082, 0.0625, 0.1753, 0.0304, 0.3166, -0.1920, 0.2261, 0.7402]
  task-specific observation = [0.8118, 0.0750, 0.9616, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3075, -0.1979, 0.2000, 0.7169]
step=124
  robot observation = [0.4130, 0.2205, 0.1361, 0.0032, -0.0401, -0.0698, 0.0640, 0.1756, 0.0295, 0.3131, -0.1922, 0.2302, 0.7401]
  task-specific observation = [0.8127, 0.0741, 0.9590, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3070, -0.1945, 0.2000, 0.7183]
step=132
  robot observation = [0.4132, 0.2165, 0.1422, 0.0011, -0.0342, 0.0105, 0.0634, 0.1755, 0.0298, 0.3116, -0.1919, 0.2317, 0.7403]
  task-specific observation = [0.8130, 0.0738, 0.9577, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3072, -0.1920, 0.2000, 0.7187]
step=139
  robot observation = [0.4113, 0.2154, 0.1449, -0.0073, 0.0099, -0.0052, 0.0633, 0.1754, 0.0294, 0.3109, -0.1916, 0.2323, 0.7400]
  task-specific observation = [0.8129, 0.0737, 0.9571, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3068, -0.1889, 0.2000, 0.7193]
step=147
  robot observation = [0.4097, 0.2155, 0.1476, -0.0074, -0.0052, 0.0000, 0.0630, 0.1757, 0.0289, 0.3102, -0.1910, 0.2328, 0.7397]
  task-specific observation = [0.8127, 0.0733, 0.9569, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, 0.0000, 0.0500, 0.1800, 0.0200, 0.3072, -0.1889, 0.2000, 0.7195]
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

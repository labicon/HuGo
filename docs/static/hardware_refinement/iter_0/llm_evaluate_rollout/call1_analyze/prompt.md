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
    stand_offset = 0.45
    tangent_dir = torch.stack([-button_dir[:, 1], button_dir[:, 0]], dim=-1)
    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.12 * side_sign).unsqueeze(-1)

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
    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.01)
    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.02)

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
    hand_lateral_err = torch.where(active_left, button_local_now[:, 1] - 0.14, button_local_now[:, 1] + 0.14)
    lateral_ready = torch.abs(hand_lateral_err) < 0.07
    enter_align = (dist_to_stand < 0.30) & aligned
    ready_to_press = near_stand & aligned & lateral_ready

    # Phase transitions
    phase = torch.where((phase == 0) & enter_align, torch.ones_like(phase), phase)
    phase_timer = torch.where((policy_state["phase"] == 0) & enter_align, torch.zeros_like(phase_timer), phase_timer)

    # align -> press once stance is good enough, with a short minimum settle time
    good_enough_press = (dist_to_stand < 0.28) & (torch.abs(yaw_err) < 0.38) & (torch.abs(hand_lateral_err) < 0.10)
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
    active_pre_x = clamp(prepress_local_xy[:, 0], 0.18, 0.43)
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
The episode is 372 steps long, recorded at 50 Hz. Below are 20 evenly spaced samples of the observation, for a coarse sense of how it went.
<overview>
step=0
  robot observation = [-0.1966, 0.1915, -0.2395, 0.3550, 0.1382, 1.0891, 0.0485, 0.2310, -0.0453, 0.0377, -0.2294, -0.0289, 0.7665]
  task-specific observation = [0.8046, 0.8920, 0.9604, -1.0000, 0.0000]
  last action = [0.5140, 0.4166, -0.0006, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=20
  robot observation = [-0.0299, 0.3461, 0.1209, 0.6636, 0.8047, 0.3595, 0.0281, 0.2323, -0.0376, 0.0248, -0.2255, -0.0151, 0.7552]
  task-specific observation = [0.8063, 0.8866, 0.9515, -1.0000, 0.0000]
  last action = [0.7155, 0.4721, -0.0001, 0.0200, 0.2200, -0.0200, 0.0200, -0.2200, -0.0200, 0.7500]
step=39
  robot observation = [0.1730, 0.5892, 0.0547, 0.4975, 0.3486, 0.8029, 0.1117, 0.2222, 0.0484, 0.0139, -0.2273, -0.0432, 0.7616]
  task-specific observation = [0.8085, 0.8935, 0.9505, -1.0000, 0.0000]
  last action = [0.3275, 0.3058, -0.0000, 0.2250, 0.2577, 0.0900, 0.0350, -0.2000, 0.0000, 0.7500]
step=59
  robot observation = [0.4537, 0.6894, 0.3295, 0.4566, 0.0926, 0.7016, 0.3293, 0.1635, 0.2643, 0.0416, -0.1987, 0.0070, 0.7639]
  task-specific observation = [0.8167, 0.8979, 0.9437, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.4234, 0.1376, 0.1931, 0.0491, -0.1813, 0.0188, 0.7239]
step=78
  robot observation = [0.5321, 0.8462, 0.1402, 0.2998, 0.0783, -0.7383, 0.3318, 0.1238, 0.2761, 0.0470, -0.1892, 0.0270, 0.7968]
  task-specific observation = [0.8236, 0.9119, 0.9377, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3318, 0.0836, 0.1996, 0.0499, -0.1801, 0.0199, 0.7076]
step=98
  robot observation = [0.5415, 0.9887, 0.0818, -0.1008, 0.3308, -0.1728, 0.2882, 0.0958, 0.2322, 0.0519, -0.1901, 0.0497, 0.7916]
  task-specific observation = [0.8260, 0.9228, 0.9365, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.2822, 0.0802, 0.2000, 0.0500, -0.1800, 0.0200, 0.7024]
step=117
  robot observation = [0.5139, 1.0516, 0.0562, -0.0915, 0.0878, -0.0489, 0.2628, 0.0967, 0.2095, 0.0635, -0.1897, 0.0495, 0.7487]
  task-specific observation = [0.8214, 0.9267, 0.9422, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.2406, 0.0800, 0.2000, 0.0500, -0.1800, 0.0200, 0.7126]
step=137
  robot observation = [0.5137, 1.0487, -0.0029, 0.0313, -0.1318, 0.3037, 0.1924, 0.1363, 0.0859, 0.0473, -0.2128, 0.0308, 0.7425]
  task-specific observation = [0.8254, 0.9286, 0.9371, -1.0000, 0.0000]
  last action = [-0.1283, -0.2925, -0.0000, 0.1349, 0.1500, 0.0900, 0.0350, -0.2000, -0.0000, 0.7500]
step=156
  robot observation = [0.5055, 1.0010, 0.0552, -0.0494, -0.1034, -0.2286, 0.2551, 0.1027, 0.1472, 0.0557, -0.1880, 0.0358, 0.7246]
  task-specific observation = [0.8349, 0.9327, 0.9300, -1.0000, 0.0000]
  last action = [-0.1397, -0.1963, -0.0000, 0.2478, 0.0844, 0.1931, 0.0491, -0.1813, 0.0187, 0.7500]
step=176
  robot observation = [0.5196, 0.9737, 0.0506, 0.0801, -0.0452, -0.2670, 0.2837, 0.0935, 0.1633, 0.0521, -0.1821, 0.0255, 0.7229]
  task-specific observation = [0.8351, 0.9402, 0.9396, -1.0000, 0.0000]
  last action = [-0.1304, -0.1931, -0.0000, 0.2661, 0.0803, 0.1996, 0.0499, -0.1801, 0.0199, 0.7500]
step=195
  robot observation = [0.5243, 0.9557, 0.0055, -0.3478, 0.0834, -0.0838, 0.2531, 0.0856, 0.2051, 0.0507, -0.1812, 0.0282, 0.7419]
  task-specific observation = [0.8405, 0.9404, 0.9302, -1.0000, 0.0000]
  last action = [-0.1348, -0.1972, -0.0000, 0.2561, 0.0800, 0.1961, 0.0500, -0.1800, 0.0200, 0.7500]
step=215
  robot observation = [0.4655, 0.8870, 0.0204, -0.1719, -0.4518, -0.0663, 0.2522, 0.0854, 0.2289, 0.0633, -0.1791, 0.0418, 0.7460]
  task-specific observation = [0.8258, 0.9305, 0.9434, -1.0000, 0.0000]
  last action = [-0.1049, -0.1983, -0.0000, 0.2881, 0.0800, 0.1998, 0.0500, -0.1800, 0.0200, 0.7500]
step=234
  robot observation = [0.4524, 0.8082, -0.0374, 0.1320, 0.0191, -0.2374, 0.3058, 0.0950, 0.2743, 0.0665, -0.1794, 0.0349, 0.7703]
  task-specific observation = [0.8286, 0.9218, 0.9415, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3684, 0.1164, 0.2000, 0.0500, -0.1800, 0.0200, 0.7158]
step=254
  robot observation = [0.4790, 0.8112, 0.0024, -0.0109, -0.1658, 0.0489, 0.3329, 0.1129, 0.2591, 0.0652, -0.1817, 0.0332, 0.7736]
  task-specific observation = [0.8311, 0.9274, 0.9440, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3500, 0.1125, 0.2000, 0.0500, -0.1800, 0.0200, 0.7050]
step=273
  robot observation = [0.4328, 0.8136, -0.0087, -0.0656, -0.1135, 0.0506, 0.3465, 0.1143, 0.2297, 0.0797, -0.1812, 0.0263, 0.7393]
  task-specific observation = [0.8368, 0.9307, 0.9359, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3361, 0.1166, 0.2000, 0.0500, -0.1800, 0.0200, 0.7103]
step=293
  robot observation = [0.3951, 0.8348, 0.0023, -0.0162, 0.0502, 0.0977, 0.3531, 0.1068, 0.2176, 0.0709, -0.1842, 0.0325, 0.7210]
  task-specific observation = [0.8307, 0.9310, 0.9409, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3724, 0.1031, 0.2000, 0.0500, -0.1800, 0.0200, 0.7169]
step=312
  robot observation = [0.3901, 0.8493, 0.0244, -0.0169, 0.0070, 0.0244, 0.3656, 0.0953, 0.2236, 0.0681, -0.1863, 0.0289, 0.7192]
  task-specific observation = [0.8314, 0.9320, 0.9410, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3841, 0.0849, 0.2000, 0.0500, -0.1800, 0.0200, 0.7187]
step=332
  robot observation = [0.4124, 0.8484, -0.0501, 0.1057, -0.0277, 0.1100, 0.2637, 0.1430, 0.0849, 0.0557, -0.2076, -0.0045, 0.7316]
  task-specific observation = [0.8314, 0.9311, 0.9445, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.2032, 0.1503, 0.0900, 0.0350, -0.2000, 0.0000, 0.7196]
step=351
  robot observation = [0.3985, 0.8549, 0.0674, -0.0066, 0.1459, -0.0995, 0.3596, 0.0819, 0.2035, 0.0559, -0.1917, 0.0193, 0.7282]
  task-specific observation = [0.8544, 0.9420, 0.9171, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.3982, 0.0858, 0.1917, 0.0491, -0.1813, 0.0188, 0.7149]
step=371
  robot observation = [0.3843, 0.8583, 0.0637, 0.0016, -0.0716, -0.0785, 0.3999, 0.0802, 0.2248, 0.0414, -0.1890, 0.0273, 0.7309]
  task-specific observation = [0.8326, 0.9311, 0.9460, -1.0000, 0.0000]
  last action = [0.0000, 0.0000, -0.0000, 0.4429, 0.0804, 0.1993, 0.0499, -0.1801, 0.0199, 0.7047]
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

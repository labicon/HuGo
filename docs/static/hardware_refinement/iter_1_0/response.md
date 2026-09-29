The rollout suggests the policy is already getting the robot to the right general place and orientation; the failure is in the last few centimeters of contact. So I’d make the smallest plausible fix aimed at improving press margin rather than changing the whole strategy.

Key improvements:
1. **Stand slightly closer to the button** so the press target is not at the edge of reach.
2. **Delay press until stance is tighter** instead of entering press from a merely “good enough” pose.
3. **Bias the commanded press slightly more inward** to overcome small residual pose/contact errors.
4. **Hold press a bit longer** so once contact is made, the robot sustains it long enough to register the 0.02 m inward motion.

I would avoid larger logic changes because the navigation/alignment behavior is already mostly working, and the rollout specifically points to marginal geometry/reachability as the issue.

<diff>
@@
     # Desired standing position: offset from button along outward normal,
     # with additional safety margin for navigation around pillar.
     # Need root center outside obstacle and wrist reachability.
     # Move closer so the wrist can actually reach the button while still staying outside the pillar.
-    stand_offset = 0.45
+    stand_offset = 0.42
     tangent_dir = torch.stack([-button_dir[:, 1], button_dir[:, 0]], dim=-1)
-    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.12 * side_sign).unsqueeze(-1)
+    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.10 * side_sign).unsqueeze(-1)
@@
     # Hand target on the button surface:
     # hand center should be ~0.035 m outside the face at contact;
     # pressing 0.02 inward means center target moves 0.02 inward from there.
     contact_outside = 0.035
     push_depth = 0.02
-    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.01)
+    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.02)
     prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.02)
@@
     near_stand = dist_to_stand < 0.24
     aligned = torch.abs(yaw_err) < 0.32
     button_local_now = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
     hand_lateral_err = torch.where(active_left, button_local_now[:, 1] - 0.14, button_local_now[:, 1] + 0.14)
     lateral_ready = torch.abs(hand_lateral_err) < 0.07
     enter_align = (dist_to_stand < 0.30) & aligned
     ready_to_press = near_stand & aligned & lateral_ready
@@
     # align -> press once stance is good enough, with a short minimum settle time
-    good_enough_press = (dist_to_stand < 0.28) & (torch.abs(yaw_err) < 0.38) & (torch.abs(hand_lateral_err) < 0.10)
-    phase = torch.where((phase == 1) & (phase_timer > 0.2) & good_enough_press, torch.full_like(phase, 2), phase)
-    phase_timer = torch.where((policy_state["phase"] == 1) & (phase_timer > 0.2) & good_enough_press, torch.zeros_like(phase_timer), phase_timer)
+    good_enough_press = (dist_to_stand < 0.20) & (torch.abs(yaw_err) < 0.24) & (torch.abs(hand_lateral_err) < 0.06)
+    phase = torch.where((phase == 1) & (phase_timer > 0.4) & good_enough_press, torch.full_like(phase, 2), phase)
+    phase_timer = torch.where((policy_state["phase"] == 1) & (phase_timer > 0.4) & good_enough_press, torch.zeros_like(phase_timer), phase_timer)
@@
-    press_steps = int(1.0 / dt)
+    press_steps = int(1.3 / dt)
     retract_steps = int(0.6 / dt)
@@
     # Stop only when actually ready to press, or while executing press/retract.
     stop_mask = ready_to_press | (phase >= 2)
     target_vx[stop_mask] = 0.0
     target_vy[stop_mask] = 0.0
</diff>

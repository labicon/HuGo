The anchor already gets the robot to the right side and orientation; the failure is in the last bit of arm reach. Attempt #1 pushed in the right direction, but the rollout showed the real bottleneck more clearly: even at a very good stance, the planned press target was still too far forward in pelvis coordinates (~0.44 m) relative to what the arm actually realized (~0.409 m). So just asking for a slightly tighter stance and slightly deeper press was not enough.

Smallest plausible robust fix:
1. **Bring the root stance a bit closer to the button face** so the button lies further inside practical arm reach.
2. **Reduce lateral tangent offset slightly** to avoid wasting reach on sideways placement.
3. **Move both pre-press and press targets slightly back outward** in world space so the commanded local x is less extreme, while relying on the hand capsule radius and sustained contact rather than overdriving deep into the action limit.
4. **Tighten the local-x gating before press** so the policy only enters press when the button is actually within a more reachable forward range in the current pelvis frame.
5. **Keep the rest of the logic intact** since navigation/alignment already work.

This is still a modest geometric retuning, but it directly targets the measured reach shortfall rather than only timing/thresholds.

<diff>
@@
     # Desired standing position: offset from button along outward normal,
     # with additional safety margin for navigation around pillar.
     # Need root center outside obstacle and wrist reachability.
     # Move closer so the wrist can actually reach the button while still staying outside the pillar.
-    stand_offset = 0.45
+    stand_offset = 0.40
     tangent_dir = torch.stack([-button_dir[:, 1], button_dir[:, 0]], dim=-1)
-    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.12 * side_sign).unsqueeze(-1)
+    stand_xy = button_pos[:, 0:2] + button_dir * stand_offset + tangent_dir * (0.09 * side_sign).unsqueeze(-1)
@@
     # Hand target on the button surface:
     # hand center should be ~0.035 m outside the face at contact;
     # pressing 0.02 inward means center target moves 0.02 inward from there.
     contact_outside = 0.035
     push_depth = 0.02
-    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth - 0.01)
-    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.02)
+    press_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside - push_depth + 0.005)
+    prepress_world_xy = button_pos[:, 0:2] + button_dir * (contact_outside + 0.035)
@@
     near_stand = dist_to_stand < 0.24
     aligned = torch.abs(yaw_err) < 0.32
     button_local_now = world_to_local_xy(button_pos[:, 0:2] - root_xy, root_yaw)
+    forward_ready = button_local_now[:, 0] < 0.40
     hand_lateral_err = torch.where(active_left, button_local_now[:, 1] - 0.14, button_local_now[:, 1] + 0.14)
     lateral_ready = torch.abs(hand_lateral_err) < 0.07
     enter_align = (dist_to_stand < 0.30) & aligned
-    ready_to_press = near_stand & aligned & lateral_ready
+    ready_to_press = near_stand & aligned & lateral_ready & forward_ready
@@
     phase_timer = torch.where((policy_state["phase"] == 0) & enter_align, torch.zeros_like(phase_timer), phase_timer)
 
     # align -> press once stance is good enough, with a short minimum settle time
-    good_enough_press = (dist_to_stand < 0.28) & (torch.abs(yaw_err) < 0.38) & (torch.abs(hand_lateral_err) < 0.10)
+    good_enough_press = (dist_to_stand < 0.22) & (torch.abs(yaw_err) < 0.30) & (torch.abs(hand_lateral_err) < 0.08) & (button_local_now[:, 0] < 0.395)
     phase = torch.where((phase == 1) & (phase_timer > 0.2) & good_enough_press, torch.full_like(phase, 2), phase)
     phase_timer = torch.where((policy_state["phase"] == 1) & (phase_timer > 0.2) & good_enough_press, torch.zeros_like(phase_timer), phase_timer)
@@
     # Active hand targets
-    active_pre_x = clamp(prepress_local_xy[:, 0], 0.18, 0.43)
+    active_pre_x = clamp(prepress_local_xy[:, 0], 0.18, 0.41)
     active_pre_y_left = clamp(prepress_local_xy[:, 1], 0.08, 0.34)
     active_pre_y_right = clamp(prepress_local_xy[:, 1], -0.34, -0.08)
 
-    active_press_x = clamp(press_local_xy[:, 0], 0.22, 0.45)
+    active_press_x = clamp(press_local_xy[:, 0], 0.20, 0.42)
     active_press_y_left = clamp(press_local_xy[:, 1], 0.08, 0.34)
     active_press_y_right = clamp(press_local_xy[:, 1], -0.34, -0.08)
</diff>

The rollout suggests the policy is already getting the robot to the right general place and orientation; the failure is in the last few centimeters of contact. So I’d make the smallest plausible fix aimed at improving press margin rather than changing the whole strategy.

Key improvements:
1. **Stand slightly closer to the button** so the press target is not at the edge of reach.
2. **Delay press until stance is tighter** instead of entering press from a merely “good enough” pose.
3. **Bias the commanded press slightly more inward** to overcome small residual pose/contact errors.
4. **Hold press a bit longer** so once contact is made, the robot sustains it long enough to register the 0.02 m inward motion.

I would avoid larger logic changes because the navigation/alignment behavior is already mostly working, and the rollout specifically points to marginal geometry/reachability as the issue.

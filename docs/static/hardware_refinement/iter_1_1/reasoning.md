The anchor already gets the robot to the right side and orientation; the failure is in the last bit of arm reach. Attempt #1 pushed in the right direction, but the rollout showed the real bottleneck more clearly: even at a very good stance, the planned press target was still too far forward in pelvis coordinates (~0.44 m) relative to what the arm actually realized (~0.409 m). So just asking for a slightly tighter stance and slightly deeper press was not enough.

Smallest plausible robust fix:
1. **Bring the root stance a bit closer to the button face** so the button lies further inside practical arm reach.
2. **Reduce lateral tangent offset slightly** to avoid wasting reach on sideways placement.
3. **Move both pre-press and press targets slightly back outward** in world space so the commanded local x is less extreme, while relying on the hand capsule radius and sustained contact rather than overdriving deep into the action limit.
4. **Tighten the local-x gating before press** so the policy only enters press when the button is actually within a more reachable forward range in the current pelvis frame.
5. **Keep the rest of the logic intact** since navigation/alignment already work.

This is still a modest geometric retuning, but it directly targets the measured reach shortfall rather than only timing/thresholds.

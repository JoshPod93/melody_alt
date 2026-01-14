#!/usr/bin/env python
"""
Analyze P300 flash timings and color distribution.

Computes:
- Total flash events in 10 seconds
- Number of red (target) flashes
- Color balance (should be in blocks)
- Command operations per 10 seconds
"""

# Current timings
flash_duration_ms = 62
isi_ms = 125
duration_seconds = 10.0
target_probability = 0.2  # 20%

# Non-target colors
NON_TARGET_COLORS = ['blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
TARGET_COLOR = 'red'
ALL_COLORS = [TARGET_COLOR] + NON_TARGET_COLORS

# Timing calculations
cycle_time_ms = flash_duration_ms + isi_ms  # 187ms
flashes_per_second_per_target = 1000.0 / cycle_time_ms  # ~5.35 flashes/sec

# Target offset: bottom is offset by half ISI (62.5ms)
# This means they alternate, so we get approximately:
# - Top flashes at: 0ms, 187ms, 374ms, 561ms, ...
# - Bottom flashes at: 62.5ms, 249.5ms, 436.5ms, 623.5ms, ...
# Total flash rate: ~10.7 flashes/sec (both targets combined)

total_flashes_per_second = 2 * flashes_per_second_per_target
total_flashes_10s = total_flashes_per_second * duration_seconds

# Expected red flashes (20% probability, but red never on both simultaneously)
# If both would be red, one is randomly chosen, so effective red probability is slightly less
# For independent 20% on each: P(red on top) = 0.2, P(red on bottom) = 0.2
# P(both red) = 0.2 * 0.2 = 0.04, which gets resolved to one red
# So: P(red on top only) = 0.2 * 0.8 = 0.16
#     P(red on bottom only) = 0.2 * 0.8 = 0.16  
#     P(red on one, resolved) = 0.04
# Total P(red appears) = 0.16 + 0.16 + 0.04 = 0.36 per flash pair
# But since we have alternating flashes, each flash has ~18% chance of being red

# More accurate: each flash independently has 20% chance, but if both would be red in same cycle,
# one is chosen. Since they're offset, this rarely happens.
# Expected red flashes ≈ 0.2 * total_flashes_10s

expected_red_flashes = target_probability * total_flashes_10s
expected_commands_per_10s = expected_red_flashes

print("=" * 60)
print("P300 Flash Timing Analysis")
print("=" * 60)
print(f"\nTimings:")
print(f"  Flash duration: {flash_duration_ms}ms")
print(f"  ISI: {isi_ms}ms")
print(f"  Cycle time: {cycle_time_ms}ms")
print(f"  Flashes/sec per target: {flashes_per_second_per_target:.2f}")
print(f"  Total flashes/sec (both targets): {total_flashes_per_second:.2f}")
print(f"\nFor {duration_seconds}s duration:")
print(f"  Total flash events: {total_flashes_10s:.0f}")
print(f"  Expected red (target) flashes: {expected_red_flashes:.1f}")
print(f"  Expected command operations: {expected_commands_per_10s:.1f}")
print(f"\nColor distribution:")
print(f"  Target color (red): {TARGET_COLOR}")
print(f"  Non-target colors: {len(NON_TARGET_COLORS)} colors")
print(f"  Total colors: {len(ALL_COLORS)}")
print(f"\nCurrent Issues:")
print(f"  ❌ Colors are randomly selected (not in blocks)")
print(f"  ❌ Color balance not guaranteed")
print(f"  ✅ Red never on both targets simultaneously (enforced)")
print(f"  ✅ Targets alternate (offset by {isi_ms/2}ms)")
print("\n" + "=" * 60)
print("Recommended Block-Based Color Scheme:")
print("=" * 60)
print(f"\nFor balanced presentation, colors should cycle in blocks:")
print(f"  - Each block contains all {len(ALL_COLORS)} colors")
print(f"  - Red appears once per block (1/{len(ALL_COLORS)} = {1/len(ALL_COLORS)*100:.1f}%)")
print(f"  - Blocks repeat throughout the session")
print(f"  - Top and bottom use DIFFERENT block offsets to prevent sync")
print(f"\nWith blocks:")
print(f"  - Red frequency: 1/{len(ALL_COLORS)} = {1/len(ALL_COLORS)*100:.1f}% (vs target {target_probability*100:.0f}%)")
print(f"  - Commands per 10s: {total_flashes_10s / len(ALL_COLORS):.1f}")
print(f"\nNote: Block-based approach gives {1/len(ALL_COLORS)*100:.1f}% red, not {target_probability*100:.0f}%")
print(f"     To get {target_probability*100:.0f}% red, need to adjust block composition")

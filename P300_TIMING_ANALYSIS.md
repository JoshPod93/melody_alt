# P300 Flash Timing and Color Sequence Analysis

## Current Timings
- **Flash duration**: 62ms
- **ISI (Inter-Stimulus Interval)**: 125ms
- **Cycle time**: 187ms (62ms + 125ms)
- **Flashes per second per target**: ~5.35 flashes/sec
- **Total flashes/sec (both targets)**: ~10.70 flashes/sec

## For 10 Second Duration
- **Total flash events**: ~107 flashes
  - Top target: ~53.5 flashes
  - Bottom target: ~53.5 flashes
- **Expected red (target) flashes**: ~21.4 flashes (20% probability)
- **Expected command operations**: ~21.4 commands per 10 seconds

## Color Distribution
- **Target color**: red (1 color)
- **Non-target colors**: blue, green, yellow, cyan, magenta, orange, purple (7 colors)
- **Total colors**: 8 colors

## Implementation: Block-Based Randomized Sequences

### Pre-Generation and Verification
1. **Sequence Generation** (before composition starts):
   - Generate randomized block-based sequences for top and bottom
   - Each block contains red + 4 non-target colors (5 colors per block)
   - Red appears once per block = 20% frequency
   - Blocks are shuffled randomly
   - Top and bottom use different block compositions and offsets

2. **Verification** (before composition starts):
   - ✅ No simultaneous red flashes (0 occurrences)
   - ✅ Red frequency ~20% on both targets (within 5% tolerance)
   - ✅ Top and bottom sequences are different
   - ✅ All colors appear in sequences
   - ✅ Sequences match expected length

3. **Runtime**:
   - Colors are selected from pre-generated sequences
   - No random selection during runtime (deterministic)
   - Sequences cycle if duration exceeds sequence length

### Key Features
- **Block-based**: Colors appear in balanced blocks
- **Randomized**: Blocks are shuffled, sequences are different
- **Verified**: All constraints checked before starting
- **No double red**: Guaranteed no simultaneous red flashes
- **Different schemes**: Top and bottom use different sequences with offset

### Command Operations
- **Commands per 10 seconds**: ~21-22 commands
- **Command rate**: ~2.1-2.2 commands/second
- Each red flash on top = UP command
- Each red flash on bottom = DOWN command

## Statistics Output
When sequences are generated, the system prints:
```
[P300] Color sequences generated and verified:
  Top: 54 flashes, 11 red (20.4%)
  Bottom: 54 flashes, 11 red (20.4%)
  Total commands per 10s: 22
  Simultaneous red flashes: 0 (verified)
```

# Starter 2 Checkpoint

This checkpoint saves the game state at the RIVAL_BEDROOM milestone, which occurs after visiting May's bedroom in her house during the tutorial.

## Files Included:
- `02_starter_2.state` - The emulator state at this checkpoint
- `02_starter_2_milestones.json` - Milestone progress data
- `02_starter_2_map_stitcher.json` - Map data and stitching information

## Milestones Reached:
- GAME_RUNNING: ✓
- PLAYER_NAME_SET: ✓
- INTRO_CUTSCENE_COMPLETE: ✓
- LITTLEROOT_TOWN: ✓
- PLAYER_HOUSE_ENTERED: ✓
- PLAYER_BEDROOM: ✓
- CLOCK_SET: ✓
- RIVAL_HOUSE: ✓
- RIVAL_BEDROOM: ✓

## How to Load This Checkpoint:

```bash
# Using the load-state flag
python run.py --load-state Emerald-GBAdvance/splits/02_starter_2/02_starter_2.state

# Or using the checkpoint loading system
python run.py --load-checkpoint --load-state Emerald-GBAdvance/splits/02_starter_2/02_starter_2.state
```

## Note:
This checkpoint was automatically saved by the CLI agent after each step, ensuring we can resume from this point if needed.


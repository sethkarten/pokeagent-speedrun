# Tutorial 3 Checkpoint

This checkpoint saves the game state at the RIVAL_HOUSE milestone, which occurs after visiting the rival's house and before proceeding further in the tutorial.

## Files Included:
- `01_tutorial_3.state` - The emulator state at this checkpoint
- `01_tutorial_3_milestones.json` - Milestone progress data
- `01_tutorial_3_map_stitcher.json` - Map data and stitching information

## Milestones Reached:
- GAME_RUNNING: ✓
- PLAYER_NAME_SET: ✓
- INTRO_CUTSCENE_COMPLETE: ✓
- LITTLEROOT_TOWN: ✓
- PLAYER_HOUSE_ENTERED: ✓
- PLAYER_BEDROOM: ✓
- CLOCK_SET: ✓
- RIVAL_HOUSE: ✓

## How to Load This Checkpoint:

```bash
# Using the load-state flag
python run.py --load-state Emerald-GBAdvance/splits/01_tutorial_3/01_tutorial_3.state

# Or using the checkpoint loading system
python run.py --load-checkpoint --load-state Emerald-GBAdvance/splits/01_tutorial_3/01_tutorial_3.state
```

## Note:
This checkpoint was automatically saved by the CLI agent after each step, ensuring we can resume from this point if needed.


# Tutorial Split 2 - Resume Checkpoint

This split contains a checkpoint state from the middle of the tutorial sequence.

## State Files
- `01_tutorial_2.state` - Game state checkpoint
- `01_tutorial_2_map_stitcher.json` - Map exploration data
- `01_tutorial_2_milestones.json` - Milestone progress

## Milestones Completed
From `01_tutorial_2_milestones.json`:
- GAME_RUNNING ✓
- PLAYER_NAME_SET ✓
- INTRO_CUTSCENE_COMPLETE ✓
- LITTLEROOT_TOWN ✓
- PLAYER_HOUSE_ENTERED ✓
- PLAYER_BEDROOM ✓
- CLOCK_SET ✓

## Next Steps
The agent should continue from objective index 3 (tutorial_04_exit_player_house), which means:
- Exit the player's house by walking down the stairs (S) and through the door (D)
- Continue with the rest of the tutorial sequence

## Running This Split
```bash
python run.py --load-state Emerald-GBAdvance/splits/01_tutorial_2/01_tutorial_2.state --direct-objectives tutorial_to_starter --direct-objectives-start 3
```

## Direct Objective Sequence
Starting at index 3 out of 15 total objectives:
- [x] tutorial_01_exit_truck
- [x] tutorial_02_go_to_bedroom
- [x] tutorial_03_interact_with_clock
- [ ] tutorial_04_exit_player_house (CURRENT)
- [ ] tutorial_05_enter_rival_house
- [ ] tutorial_06_go_to_rival_bedroom
- [ ] tutorial_07_talk_to_rival
- [ ] tutorial_08_exit_rival_house
- [ ] tutorial_09_north_to_route101
- [ ] tutorial_10_find_prof_birch
- [ ] tutorial_11_approach_birch
- [ ] tutorial_12_battle_zigzagoon
- [ ] tutorial_13_interact_with_professor_birch
- [ ] tutorial_14_talk_to_professor_birch_in_lab
- [ ] tutorial_15_exit_professor_birch_lab











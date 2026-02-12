"""
Categorized Game Objectives

269 objectives split into 2 categories:
- STORY_OBJECTIVES (214): Narrative progression
- BATTLING_OBJECTIVES (55): Team building and training
"""

from agent.objective_types import DirectObjective


# Story objectives (214 total)
STORY_OBJECTIVES = [
    DirectObjective(
        id="tutorial_000",
        description="Exit the moving truck and enter Littleroot Town",
        action_type="navigate",
        category="story",
        target_location="Littleroot Town",
        navigation_hint="Walk right to the door (D) to exit truck and enter Littleroot Town",
        completion_condition="location_contains_littleroot",
        priority=1
    ),
    DirectObjective(
        id="tutorial_001",
        description="Follow Mom into your house and go upstairs to your bedroom",
        action_type="navigate",
        category="story",
        target_location="Player's Bedroom",
        navigation_hint="Press A through dialogue. Walk north to stairs (S) at (8,2). Mom redirects you until you set the clock.",
        completion_condition="player_bedroom_reached",
        priority=1
    ),
    DirectObjective(
        id="tutorial_002",
        description="Interact with the clock on the wall to set the time",
        action_type="interact",
        category="story",
        target_location="Player's Bedroom",
        navigation_hint="Clock (K) at position (5,1). Navigate to (5,2), press UP then A. Use D-pad to set time, confirm with A.",
        completion_condition="clock_set",
        priority=1
    ),
    DirectObjective(
        id="tutorial_003",
        description="Withdraw Potion from PC",
        action_type="interact",
        category="story",
        target_location="Player's Bedroom",
        navigation_hint="PC on east wall. Interact, select WITHDRAW ITEM, take Potion. Useful for early game.",
        completion_condition="potion_withdrawn",
        priority=1
    ),
    DirectObjective(
        id="tutorial_004",
        description="Exit your house to Littleroot Town",
        action_type="navigate",
        category="story",
        target_location="Littleroot Town",
        navigation_hint="Go downstairs. Mom shows TV event. Exit through door (D) south-east. Walk through doors, don't press A.",
        completion_condition="exited_player_house",
        priority=1
    ),
    DirectObjective(
        id="tutorial_005",
        description="Visit rival's house and go upstairs",
        action_type="navigate",
        category="story",
        target_location="Rival's House 2F",
        navigation_hint="House west of yours. Enter, go upstairs to trigger rival intro event. Rival not home yet.",
        completion_condition="rival_house_visited",
        priority=1
    ),
    DirectObjective(
        id="tutorial_006",
        description="Go to Route 101 to find Professor Birch being attacked",
        action_type="navigate",
        category="story",
        target_location="Route 101",
        navigation_hint="Exit Littleroot north. Birch is being chased by wild Poochyena. Approach his bag.",
        completion_condition="birch_rescue_triggered",
        priority=1
    ),
    DirectObjective(
        id="tutorial_007",
        description="Choose starter Pokemon from Birch's bag",
        action_type="menu",
        category="story",
        target_location="Route 101",
        navigation_hint="Interact with bag. Choose: Treecko (Grass), Torchic (Fire), or Mudkip (Water). Mudkip has best type coverage overall. Treecko good for speedruns.",
        completion_condition="starter_pokemon_obtained",
        priority=1
    ),
    DirectObjective(
        id="tutorial_008",
        description="Defeat the wild Poochyena attacking Professor Birch",
        action_type="battle",
        category="story",
        target_location="Route 101",
        navigation_hint="Use FIGHT, select attacking move (Pound/Scratch/Tackle). Lv 2 Poochyena, easy fight.",
        completion_condition="birch_rescued",
        priority=1
    ),
    DirectObjective(
        id="tutorial_009",
        description="Follow Birch back to his lab",
        action_type="navigate",
        category="story",
        target_location="Birch's Lab",
        navigation_hint="Birch walks south. Follow to lab (large building south Littleroot). Enter and talk to him.",
        completion_condition="in_birch_lab",
        priority=1
    ),
    DirectObjective(
        id="tutorial_010",
        description="Receive starter Pokemon officially from Birch",
        action_type="dialogue",
        category="story",
        target_location="Birch's Lab",
        navigation_hint="Talk to Birch. He officially gives you the Pokemon you chose.",
        completion_condition="starter_officially_received",
        priority=1
    ),
    DirectObjective(
        id="early_011",
        description="Travel north through Route 101 to Oldale Town",
        action_type="navigate",
        category="story",
        target_location="Oldale Town",
        navigation_hint="Head north from Littleroot. Can battle wild Pokemon in grass for exp.",
        completion_condition="oldale_town_reached",
        priority=1
    ),
    DirectObjective(
        id="early_012",
        description="Heal at Oldale Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Oldale Pokemon Center",
        navigation_hint="Pokemon Center near town entrance. Talk to Nurse Joy to heal.",
        completion_condition="healed_oldale",
        priority=1
    ),
    DirectObjective(
        id="early_013",
        description="Travel west to Route 103 to find rival",
        action_type="navigate",
        category="story",
        target_location="Route 103",
        navigation_hint="Exit Oldale west. Rival is at north end of Route 103.",
        completion_condition="route103_reached",
        priority=1
    ),
    DirectObjective(
        id="early_014",
        description="Battle rival (May/Brendan) - first rival battle",
        action_type="battle",
        category="story",
        target_location="Route 103",
        navigation_hint="Rival has starter with type advantage (Lv 5). At Lv 7+ you should win easily.",
        completion_condition="rival_battle_1_won",
        priority=1,
        recommended_battling_objectives=["battle_000", "battle_002"]
    ),
    DirectObjective(
        id="early_015",
        description="Return to Birch's Lab",
        action_type="navigate",
        category="story",
        target_location="Birch's Lab",
        navigation_hint="Route 103 → Oldale → Route 101 → Littleroot → Birch's Lab",
        completion_condition="returned_to_lab",
        priority=1
    ),
    DirectObjective(
        id="early_016",
        description="Receive Pokedex from Professor Birch",
        action_type="dialogue",
        category="story",
        target_location="Birch's Lab",
        navigation_hint="Talk to Birch. He gives you the Pokedex.",
        completion_condition="pokedex_received",
        priority=1
    ),
    DirectObjective(
        id="early_017",
        description="Receive Poke Balls from rival",
        action_type="dialogue",
        category="story",
        target_location="Birch's Lab",
        navigation_hint="Rival gives you 5 Poke Balls. Now you can catch Pokemon.",
        completion_condition="pokeballs_received",
        priority=1
    ),
    DirectObjective(
        id="early_018",
        description="Visit Mom to receive Running Shoes",
        action_type="dialogue",
        category="story",
        target_location="Player's House",
        navigation_hint="Go home, talk to Mom. She gives Running Shoes. Hold B to run - essential!",
        completion_condition="running_shoes_received",
        priority=1
    ),
    DirectObjective(
        id="petalburg_019",
        description="Travel west through Route 102 to Petalburg City",
        action_type="navigate",
        category="story",
        target_location="Petalburg City",
        navigation_hint="Battle trainers for exp. Wild Pokemon: Zigzagoon, Wurmple, Lotad/Seedot.",
        completion_condition="petalburg_reached",
        priority=1
    ),
    DirectObjective(
        id="petalburg_020",
        description="Heal at Petalburg Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Petalburg Pokemon Center",
        navigation_hint="Pokemon Center in city. Heal before meeting Dad.",
        completion_condition="healed_petalburg",
        priority=1
    ),
    DirectObjective(
        id="petalburg_021",
        description="Visit Petalburg Gym to meet your father Norman",
        action_type="dialogue",
        category="story",
        target_location="Petalburg Gym",
        navigation_hint="Gym is large building north-center. Meet Norman. Cannot battle him until 4 badges.",
        completion_condition="norman_met",
        priority=1
    ),
    DirectObjective(
        id="petalburg_022",
        description="Help Wally catch his first Pokemon",
        action_type="dialogue",
        category="story",
        target_location="Route 102",
        navigation_hint="Wally enters gym. Norman asks you to help. Automatic trip to Route 102. Watch Wally catch Ralts.",
        completion_condition="wally_event_complete",
        priority=1
    ),
    DirectObjective(
        id="petalburg_023",
        description="Travel west to Route 104 South",
        action_type="navigate",
        category="story",
        target_location="Route 104 South",
        navigation_hint="Exit Petalburg west. Route 104 has beach and trainers.",
        completion_condition="route104_south_reached",
        priority=1
    ),
    DirectObjective(
        id="petalburg_024",
        description="Enter Petalburg Woods",
        action_type="navigate",
        category="story",
        target_location="Petalburg Woods",
        navigation_hint="Forest entrance at north of Route 104 South. Dark winding paths.",
        completion_condition="petalburg_woods_entered",
        priority=1
    ),
    DirectObjective(
        id="petalburg_025",
        description="Find Devon Researcher being threatened by Team Aqua",
        action_type="dialogue",
        category="story",
        target_location="Petalburg Woods",
        navigation_hint="Near woods exit, researcher threatened by Aqua Grunt.",
        completion_condition="devon_researcher_found",
        priority=1
    ),
    DirectObjective(
        id="petalburg_026",
        description="Battle Team Aqua Grunt to save researcher",
        action_type="battle",
        category="story",
        target_location="Petalburg Woods",
        navigation_hint="Grunt has Poochyena Lv 9. First Team Aqua encounter.",
        completion_condition="aqua_grunt_woods_defeated",
        priority=1
    ),
    DirectObjective(
        id="petalburg_027",
        description="Receive Great Ball from Devon Researcher",
        action_type="dialogue",
        category="story",
        target_location="Petalburg Woods",
        navigation_hint="Researcher thanks you with a Great Ball.",
        completion_condition="great_ball_received",
        priority=1
    ),
    DirectObjective(
        id="petalburg_028",
        description="Exit Petalburg Woods to Route 104 North",
        action_type="navigate",
        category="story",
        target_location="Route 104 North",
        navigation_hint="Continue north through woods. Exit leads to bridge and Rustboro.",
        completion_condition="route104_north_reached",
        priority=1
    ),
    DirectObjective(
        id="petalburg_029",
        description="Cross bridge and continue to Rustboro City",
        action_type="navigate",
        category="story",
        target_location="Rustboro City",
        navigation_hint="Cross bridge, battle trainers. Flower Shop on left (optional). Continue north.",
        completion_condition="rustboro_reached",
        priority=1
    ),
    DirectObjective(
        id="rustboro_030",
        description="Heal at Rustboro Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Rustboro Pokemon Center",
        navigation_hint="Pokemon Center near city entrance.",
        completion_condition="healed_rustboro",
        priority=1
    ),
    DirectObjective(
        id="rustboro_031",
        description="Enter Rustboro Gym",
        action_type="navigate",
        category="story",
        target_location="Rustboro Gym",
        navigation_hint="Gym in northwest Rustboro. Rock-type gym with school theme.",
        completion_condition="rustboro_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="rustboro_032",
        description="Defeat Gym trainers for experience",
        action_type="battle",
        category="story",
        target_location="Rustboro Gym",
        navigation_hint="Two trainers with Geodude. Good exp. Can skip by walking around.",
        completion_condition="rustboro_gym_trainers_defeated",
        priority=2,
        optional=True
    ),
    DirectObjective(
        id="rustboro_033",
        description="Battle Gym Leader Roxanne for Stone Badge",
        action_type="battle",
        category="story",
        target_location="Rustboro Gym",
        navigation_hint="Roxanne: Geodude Lv12, Geodude Lv12, Nosepass Lv15. Nosepass has Rock Tomb. Water/Grass moves 4x effective on Geodude, 2x on Nosepass.",
        completion_condition="stone_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_008", "battle_009", "battle_007", "battle_010"]
    ),
    DirectObjective(
        id="rustboro_034",
        description="Receive TM39 Rock Tomb from Roxanne",
        action_type="dialogue",
        category="story",
        target_location="Rustboro Gym",
        navigation_hint="Badge allows Cut outside battle, Pokemon up to Lv 20 obey.",
        completion_condition="tm39_received",
        priority=1
    ),
    DirectObjective(
        id="rustboro_035",
        description="Exit Gym - Devon Researcher reports theft",
        action_type="dialogue",
        category="story",
        target_location="Rustboro City",
        navigation_hint="Researcher says goods stolen again. Chase thief to Route 116!",
        completion_condition="theft_event_triggered",
        priority=1
    ),
    DirectObjective(
        id="rustboro_036",
        description="Chase Team Aqua Grunt east to Route 116",
        action_type="navigate",
        category="story",
        target_location="Route 116",
        navigation_hint="Go east from Rustboro. Grunt fled to Rusturf Tunnel.",
        completion_condition="route116_reached",
        priority=1
    ),
    DirectObjective(
        id="rustboro_037",
        description="Enter Rusturf Tunnel",
        action_type="navigate",
        category="story",
        target_location="Rusturf Tunnel",
        navigation_hint="Tunnel at east end of Route 116. Grunt inside with hostage Peeko (Wingull).",
        completion_condition="rusturf_tunnel_entered",
        priority=1
    ),
    DirectObjective(
        id="rustboro_038",
        description="Battle Team Aqua Grunt to rescue Peeko",
        action_type="battle",
        category="story",
        target_location="Rusturf Tunnel",
        navigation_hint="Grunt has Poochyena Lv 11. After win, he drops Devon Goods.",
        completion_condition="aqua_grunt_tunnel_defeated",
        priority=1
    ),
    DirectObjective(
        id="rustboro_039",
        description="Retrieve Devon Goods and meet Mr. Briney",
        action_type="dialogue",
        category="story",
        target_location="Rusturf Tunnel",
        navigation_hint="Pick up Devon Goods. Mr. Briney thanks you for saving Peeko.",
        completion_condition="devon_goods_obtained",
        priority=1
    ),
    DirectObjective(
        id="rustboro_040",
        description="Return to Rustboro and visit Devon Corporation",
        action_type="navigate",
        category="story",
        target_location="Devon Corporation",
        navigation_hint="Large building in center Rustboro. Researcher directs you inside.",
        completion_condition="devon_corp_entered",
        priority=1
    ),
    DirectObjective(
        id="rustboro_041",
        description="Meet Mr. Stone on top floor",
        action_type="dialogue",
        category="story",
        target_location="Devon Corporation 3F",
        navigation_hint="Take elevator/stairs up. Mr. Stone asks you to deliver Letter to Steven (Dewford) and Goods to Capt. Stern (Slateport).",
        completion_condition="mr_stone_met",
        priority=1
    ),
    DirectObjective(
        id="rustboro_042",
        description="Receive PokeNav, Letter, and Devon Goods",
        action_type="dialogue",
        category="story",
        target_location="Devon Corporation 3F",
        navigation_hint="Mr. Stone gives PokeNav (map/trainer eyes). Letter for Steven, Goods for Stern.",
        completion_condition="pokenav_received",
        priority=1
    ),
    DirectObjective(
        id="dewford_043",
        description="Travel to Mr. Briney's cottage on Route 104 South",
        action_type="navigate",
        category="story",
        target_location="Mr. Briney's Cottage",
        navigation_hint="Go back through Petalburg Woods to Route 104 beach. Cottage on shore.",
        completion_condition="briney_cottage_reached",
        priority=1
    ),
    DirectObjective(
        id="dewford_044",
        description="Sail to Dewford Town with Mr. Briney",
        action_type="dialogue",
        category="story",
        target_location="Mr. Briney's Cottage",
        navigation_hint="Talk to Briney. He offers boat rides. Choose Dewford Town.",
        completion_condition="sailed_to_dewford",
        priority=1
    ),
    DirectObjective(
        id="dewford_045",
        description="Arrive at Dewford Town",
        action_type="navigate",
        category="story",
        target_location="Dewford Town",
        navigation_hint="Small island town. Gym and Granite Cave access.",
        completion_condition="dewford_reached",
        priority=1
    ),
    DirectObjective(
        id="dewford_046",
        description="Enter Granite Cave to find Steven",
        action_type="navigate",
        category="story",
        target_location="Granite Cave",
        navigation_hint="Cave north of Dewford. Dark inside but Steven accessible without Flash.",
        completion_condition="granite_cave_entered",
        priority=1
    ),
    DirectObjective(
        id="dewford_047",
        description="Receive HM05 Flash from hiker at entrance",
        action_type="dialogue",
        category="story",
        target_location="Granite Cave Entrance",
        navigation_hint="Hiker gives HM05 Flash. Lights up dark caves. Needs Stone Badge.",
        completion_condition="hm05_received",
        priority=1
    ),
    DirectObjective(
        id="dewford_048",
        description="Navigate to Granite Cave depths to find Steven",
        action_type="navigate",
        category="story",
        target_location="Granite Cave B2F",
        navigation_hint="Go down ladders. Without Flash, move carefully along walls. Steven studying stones in back.",
        completion_condition="steven_found_granite",
        priority=1
    ),
    DirectObjective(
        id="dewford_049",
        description="Deliver Letter to Steven, receive TM47 Steel Wing",
        action_type="dialogue",
        category="story",
        target_location="Granite Cave B2F",
        navigation_hint="Talk to Steven. He reads letter, gives TM47 Steel Wing as thanks.",
        completion_condition="letter_delivered",
        priority=1
    ),
    DirectObjective(
        id="dewford_050",
        description="Return to Dewford Town",
        action_type="navigate",
        category="story",
        target_location="Dewford Town",
        navigation_hint="Exit Granite Cave, return to town.",
        completion_condition="returned_dewford",
        priority=1
    ),
    DirectObjective(
        id="dewford_051",
        description="Heal at Dewford Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Dewford Pokemon Center",
        navigation_hint="Heal before gym challenge.",
        completion_condition="healed_dewford",
        priority=1
    ),
    DirectObjective(
        id="dewford_052",
        description="Enter Dewford Gym",
        action_type="navigate",
        category="story",
        target_location="Dewford Gym",
        navigation_hint="Fighting-type gym. Dark inside - visibility expands as you beat trainers.",
        completion_condition="dewford_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="dewford_053",
        description="Navigate dark gym, defeat trainers to expand visibility",
        action_type="battle",
        category="story",
        target_location="Dewford Gym",
        navigation_hint="Each trainer defeated expands light radius. Fighting types weak to Flying/Psychic.",
        completion_condition="dewford_gym_trainers_defeated",
        priority=1
    ),
    DirectObjective(
        id="dewford_054",
        description="Battle Gym Leader Brawly for Knuckle Badge",
        action_type="battle",
        category="story",
        target_location="Dewford Gym",
        navigation_hint="Brawly: Machop Lv16, Meditite Lv16, Makuhita Lv19. Makuhita has Bulk Up. Flying types (Taillow/Wingull) or Psychic (Ralts) excellent.",
        completion_condition="knuckle_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_013", "battle_011", "battle_005", "battle_014"]
    ),
    DirectObjective(
        id="dewford_055",
        description="Receive TM08 Bulk Up from Brawly",
        action_type="dialogue",
        category="story",
        target_location="Dewford Gym",
        navigation_hint="Badge allows Flash outside, Pokemon up to Lv 30 obey.",
        completion_condition="tm08_received",
        priority=1
    ),
    DirectObjective(
        id="dewford_056",
        description="Return to Briney and sail to Slateport City",
        action_type="dialogue",
        category="story",
        target_location="Dewford Dock",
        navigation_hint="Find Briney at dock. Ask to sail to Slateport.",
        completion_condition="sailed_to_slateport",
        priority=1
    ),
    DirectObjective(
        id="slateport_057",
        description="Arrive at Slateport City",
        action_type="navigate",
        category="story",
        target_location="Slateport City",
        navigation_hint="Large coastal city. Beach, Market, Museum, Shipyard.",
        completion_condition="slateport_reached",
        priority=1
    ),
    DirectObjective(
        id="slateport_058",
        description="Heal at Slateport Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Slateport Pokemon Center",
        navigation_hint="Pokemon Center in city center.",
        completion_condition="healed_slateport",
        priority=1
    ),
    DirectObjective(
        id="slateport_059",
        description="Check Shipyard for Captain Stern",
        action_type="navigate",
        category="story",
        target_location="Slateport Shipyard",
        navigation_hint="South part of city. Dock Worker says Stern is at Oceanic Museum.",
        completion_condition="shipyard_checked",
        priority=1
    ),
    DirectObjective(
        id="slateport_060",
        description="Go to Oceanic Museum (50 Pokedollars entry)",
        action_type="navigate",
        category="story",
        target_location="Oceanic Museum",
        navigation_hint="North part of city. Pay 50 to enter. Team Aqua grunts inside.",
        completion_condition="museum_entered",
        priority=1
    ),
    DirectObjective(
        id="slateport_061",
        description="Battle Team Aqua Grunts in Museum 2F",
        action_type="battle",
        category="story",
        target_location="Oceanic Museum 2F",
        navigation_hint="Go upstairs. Two grunts block Stern. Battle both (Lv 15ish Poochyena/Carvanha).",
        completion_condition="museum_grunts_defeated",
        priority=1
    ),
    DirectObjective(
        id="slateport_062",
        description="Encounter Team Aqua Leader Archie",
        action_type="dialogue",
        category="story",
        target_location="Oceanic Museum 2F",
        navigation_hint="After grunts, Archie appears. Warns you not to interfere. Leaves.",
        completion_condition="archie_first_encounter",
        priority=1
    ),
    DirectObjective(
        id="slateport_063",
        description="Deliver Devon Goods to Captain Stern",
        action_type="dialogue",
        category="story",
        target_location="Oceanic Museum 2F",
        navigation_hint="Talk to Stern. Give Devon Goods (submarine parts). Delivery complete.",
        completion_condition="goods_delivered",
        priority=1
    ),
    DirectObjective(
        id="slateport_064",
        description="Exit Museum and head north to Route 110",
        action_type="navigate",
        category="story",
        target_location="Route 110",
        navigation_hint="Leave museum. Scott may appear. Head north from city to Route 110.",
        completion_condition="route110_entered",
        priority=1
    ),
    DirectObjective(
        id="slateport_065",
        description="Battle Rival on Route 110",
        action_type="battle",
        category="story",
        target_location="Route 110",
        navigation_hint="Rival appears shortly after entering East from the Trick House. Then continue North until you see your rival. Team: Starter Lv18, Wingull Lv18, plus one more Lv18.",
        completion_condition="rival_battle_2_won",
        priority=1
    ),
    DirectObjective(
        id="slateport_066",
        description="Receive Itemfinder from Rival",
        action_type="dialogue",
        category="story",
        target_location="Route 110",
        navigation_hint="Rival gives Itemfinder. Finds hidden items. Register to key items.",
        completion_condition="itemfinder_received",
        priority=1
    ),
    DirectObjective(
        id="slateport_067",
        description="Continue north through Route 110 to Mauville City",
        action_type="navigate",
        category="story",
        target_location="Mauville City",
        navigation_hint="Long route. Cycling Road above (need bike), ground path below. Trainers throughout.",
        completion_condition="mauville_reached",
        priority=1
    ),
    DirectObjective(
        id="mauville_068",
        description="Heal at Mauville Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Mauville Pokemon Center",
        navigation_hint="Pokemon Center centrally located.",
        completion_condition="healed_mauville",
        priority=1
    ),
    DirectObjective(
        id="mauville_069",
        description="Get free Bicycle from Rydel's Cycles",
        action_type="dialogue",
        category="story",
        target_location="Rydel's Cycles",
        navigation_hint="Bike shop east of Pokemon Center. Choose Mach Bike (speed) or Acro Bike (tricks). Mach better for speedruns.",
        completion_condition="bicycle_received",
        priority=1
    ),
    DirectObjective(
        id="mauville_070",
        description="Register Bicycle to Select button",
        action_type="menu",
        category="story",
        target_location="Mauville City",
        navigation_hint="Bag > Key Items > Bicycle > Register. Press Select to mount/dismount quickly.",
        completion_condition="bike_registered",
        priority=1
    ),
    DirectObjective(
        id="mauville_071",
        description="Battle Wally at Mauville Gym entrance",
        action_type="battle",
        category="story",
        target_location="Mauville Gym Entrance",
        navigation_hint="Approaching gym triggers Wally. He has Ralts Lv 16. Easy fight.",
        completion_condition="wally_mauville_defeated",
        priority=1
    ),
    DirectObjective(
        id="mauville_072",
        description="Enter Mauville Gym",
        action_type="navigate",
        category="story",
        target_location="Mauville Gym",
        navigation_hint="Electric-type gym. Switch puzzle - step on switches to change barriers.",
        completion_condition="mauville_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="mauville_073",
        description="Solve switch puzzle and navigate to Wattson",
        action_type="navigate",
        category="story",
        target_location="Mauville Gym",
        navigation_hint="Step on switches to toggle barriers. Path to Wattson in back.",
        completion_condition="gym_puzzle_solved",
        priority=1
    ),
    DirectObjective(
        id="mauville_074",
        description="Battle Gym Leader Wattson for Dynamo Badge",
        action_type="battle",
        category="story",
        target_location="Mauville Gym",
        navigation_hint="Wattson: Voltorb Lv20, Electrike Lv20, Magneton Lv22, Manectric Lv24. Ground types immune to all Electric moves. Marshtomp dominates.",
        completion_condition="dynamo_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_017", "battle_016", "battle_018", "battle_019"]
    ),
    DirectObjective(
        id="mauville_075",
        description="Receive TM34 Shock Wave from Wattson",
        action_type="dialogue",
        category="story",
        target_location="Mauville Gym",
        navigation_hint="Badge allows Rock Smash outside, Pokemon up to Lv 40 obey.",
        completion_condition="tm34_received",
        priority=1
    ),
    DirectObjective(
        id="mauville_076",
        description="Get HM06 Rock Smash from house next to Gym",
        action_type="dialogue",
        category="story",
        target_location="Rock Smash House",
        navigation_hint="House right of Gym entrance. NPC gives HM06 free.",
        completion_condition="hm06_received",
        priority=1
    ),
    DirectObjective(
        id="mauville_077",
        description="Teach Rock Smash to a Pokemon",
        action_type="menu",
        category="story",
        target_location="Mauville City",
        navigation_hint="Bag > TMs/HMs > HM06 > Teach. Many Pokemon can learn it. Required for progression.",
        completion_condition="rock_smash_taught",
        priority=1
    ),
    DirectObjective(
        id="chimney_078",
        description="Head north from Mauville to Route 111",
        action_type="navigate",
        category="story",
        target_location="Route 111 South",
        navigation_hint="North exit of Mauville. Route 111 has desert (need Go-Goggles later).",
        completion_condition="route111_south_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_079",
        description="Continue to Route 112 towards Mt. Chimney",
        action_type="navigate",
        category="story",
        target_location="Route 112",
        navigation_hint="From Route 111, head northwest. Route 112 has Fiery Path and Cable Car.",
        completion_condition="route112_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_080",
        description="Go through Fiery Path (south cave entrance)",
        action_type="navigate",
        category="story",
        target_location="Fiery Path",
        navigation_hint="Short cave. Fire type Pokemon inside. Exit to Route 111 north area.",
        completion_condition="fiery_path_traversed",
        priority=1
    ),
    DirectObjective(
        id="chimney_081",
        description="Travel through Route 113 (ash-covered route)",
        action_type="navigate",
        category="story",
        target_location="Route 113",
        navigation_hint="Volcanic ash everywhere. Glass Workshop gives Soot Sack (optional). Continue west.",
        completion_condition="route113_traversed",
        priority=1
    ),
    DirectObjective(
        id="chimney_082",
        description="Arrive at Fallarbor Town",
        action_type="navigate",
        category="story",
        target_location="Fallarbor Town",
        navigation_hint="Small town. Contest Hall, Move Tutor. Prof. Cozmo's house.",
        completion_condition="fallarbor_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_083",
        description="Heal at Fallarbor Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Fallarbor Pokemon Center",
        navigation_hint="Heal before continuing.",
        completion_condition="healed_fallarbor",
        priority=1
    ),
    DirectObjective(
        id="chimney_084",
        description="Head west through Route 114 to Meteor Falls",
        action_type="navigate",
        category="story",
        target_location="Meteor Falls",
        navigation_hint="Route 114 leads to Meteor Falls. Lanette's house along the way.",
        completion_condition="meteor_falls_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_085",
        description="Witness Team Magma/Aqua confrontation in Meteor Falls",
        action_type="dialogue",
        category="story",
        target_location="Meteor Falls",
        navigation_hint="Inside cave, Team Magma threatens Prof. Cozmo. Team Aqua interferes. Both flee to Mt. Chimney.",
        completion_condition="meteor_falls_event_witnessed",
        priority=1
    ),
    DirectObjective(
        id="chimney_086",
        description="Return to Route 112 Cable Car Station",
        action_type="navigate",
        category="story",
        target_location="Route 112 Cable Car",
        navigation_hint="Go back: Route 114 → Fallarbor → Route 113 → Route 111 → Route 112 Cable Car.",
        completion_condition="cable_car_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_087",
        description="Take Cable Car to Mt. Chimney summit",
        action_type="interact",
        category="story",
        target_location="Mt. Chimney Cable Car",
        navigation_hint="Enter station, ride car to summit. Teams fighting at top.",
        completion_condition="cable_car_used",
        priority=1
    ),
    DirectObjective(
        id="chimney_088",
        description="Battle through Team Magma Grunts at summit",
        action_type="battle",
        category="story",
        target_location="Mt. Chimney Summit",
        navigation_hint="Multiple Magma grunts. Water and Dark types. Clear path to leaders.",
        completion_condition="summit_grunts_defeated",
        priority=1
    ),
    DirectObjective(
        id="chimney_089",
        description="Battle Team Magma Admin Maxie",
        action_type="battle",
        category="story",
        target_location="Mt. Chimney Summit",
        navigation_hint="Maxie guards Archie. Has Carvanha, Mightyena. Admin-level difficulty.",
        completion_condition="maxie_chimney_defeated",
        priority=1
    ),
    DirectObjective(
        id="chimney_090",
        description="Watch Archie's plan fail and Team Magma retreat",
        action_type="dialogue",
        category="story",
        target_location="Mt. Chimney Summit",
        navigation_hint="After defeating Maxie, Archie's Meteorite plan fails. Team Magma retreats.",
        completion_condition="archie_chimney_fled",
        priority=1
    ),
    DirectObjective(
        id="chimney_091",
        description="Collect Meteorite from machine",
        action_type="interact",
        category="story",
        target_location="Mt. Chimney Summit",
        navigation_hint="Interact with machine to get Meteorite. Can trade later.",
        completion_condition="meteorite_obtained",
        priority=1
    ),
    DirectObjective(
        id="chimney_092",
        description="Descend Jagged Pass to Lavaridge Town",
        action_type="navigate",
        category="story",
        target_location="Lavaridge Town",
        navigation_hint="South path from summit = Jagged Pass. Steep descent with ledges. Leads to Lavaridge.",
        completion_condition="lavaridge_reached",
        priority=1
    ),
    DirectObjective(
        id="chimney_093",
        description="Heal at Lavaridge Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Lavaridge Pokemon Center",
        navigation_hint="Hot springs town. Heal before Gym.",
        completion_condition="healed_lavaridge",
        priority=1
    ),
    DirectObjective(
        id="chimney_094",
        description="Receive Wynaut Egg from old lady at hot springs",
        action_type="dialogue",
        category="story",
        target_location="Lavaridge Hot Springs",
        navigation_hint="Old lady near sand baths gives Pokemon Egg (Wynaut). Optional but free Pokemon.",
        completion_condition="wynaut_egg_received",
        priority=2,
        optional=True
    ),
    DirectObjective(
        id="chimney_095",
        description="Enter Lavaridge Gym",
        action_type="navigate",
        category="story",
        target_location="Lavaridge Gym",
        navigation_hint="Fire-type gym. Hot spring trapdoor puzzle - some holes drop you, some are geysers.",
        completion_condition="lavaridge_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="chimney_096",
        description="Navigate hot spring puzzle to reach Flannery",
        action_type="navigate",
        category="story",
        target_location="Lavaridge Gym",
        navigation_hint="Fall through correct holes, use geysers to ascend. Trial and error.",
        completion_condition="lavaridge_puzzle_solved",
        priority=1
    ),
    DirectObjective(
        id="chimney_097",
        description="Battle Gym Leader Flannery for Heat Badge",
        action_type="battle",
        category="story",
        target_location="Lavaridge Gym",
        navigation_hint="Flannery: Numel Lv24, Slugma Lv24, Camerupt Lv26, Torkoal Lv29. Torkoal has Overheat (strong but lowers SpAtk). Water sweeps easily.",
        completion_condition="heat_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_023", "battle_022", "battle_021", "battle_024"]
    ),
    DirectObjective(
        id="chimney_098",
        description="Receive TM50 Overheat from Flannery",
        action_type="dialogue",
        category="story",
        target_location="Lavaridge Gym",
        navigation_hint="Badge allows Strength outside, Pokemon up to Lv 50 obey.",
        completion_condition="tm50_received",
        priority=1
    ),
    DirectObjective(
        id="chimney_099",
        description="Exit Gym and receive Go-Goggles from Rival",
        action_type="dialogue",
        category="story",
        target_location="Lavaridge Town",
        navigation_hint="Rival appears outside. Gives Go-Goggles for Route 111 desert.",
        completion_condition="go_goggles_received",
        priority=1
    ),
    DirectObjective(
        id="norman_100",
        description="Travel back to Petalburg City (you now have 4 badges)",
        action_type="navigate",
        category="story",
        target_location="Petalburg City",
        navigation_hint="Fly if you taught it, or: Lavaridge → Mauville → Slateport → Route 104/102 → Petalburg.",
        completion_condition="returned_to_petalburg",
        priority=1
    ),
    DirectObjective(
        id="norman_101",
        description="Heal at Petalburg Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Petalburg Pokemon Center",
        navigation_hint="Full heal before Dad battle. This is a tough one.",
        completion_condition="healed_petalburg_pre_norman",
        priority=1
    ),
    DirectObjective(
        id="norman_102",
        description="Enter Petalburg Gym",
        action_type="navigate",
        category="story",
        target_location="Petalburg Gym",
        navigation_hint="Normal-type gym. Room system - each room has specialist trainer (Speed, Defense, etc.).",
        completion_condition="petalburg_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="norman_103",
        description="Navigate through Gym rooms and defeat trainers",
        action_type="battle",
        category="story",
        target_location="Petalburg Gym",
        navigation_hint="Choose door paths. Each trainer uses specific strategy. Must defeat trainers to progress.",
        completion_condition="petalburg_gym_trainers_defeated",
        priority=1
    ),
    DirectObjective(
        id="norman_104",
        description="Battle Gym Leader Norman (Dad) for Balance Badge",
        action_type="battle",
        category="story",
        target_location="Petalburg Gym",
        navigation_hint="Norman: Spinda Lv27, Vigoroth Lv27, Linoone Lv29, Slaking Lv31. SLAKING IS DANGEROUS - massive Attack. Has Truant ability (loafs every other turn). Use Truant turns to heal/set up. Fighting moves are key!",
        completion_condition="balance_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_029", "battle_028", "battle_011", "battle_027", "battle_026", "battle_030"]
    ),
    DirectObjective(
        id="norman_105",
        description="Receive TM42 Facade from Norman",
        action_type="dialogue",
        category="story",
        target_location="Petalburg Gym",
        navigation_hint="Badge allows Surf outside. All Pokemon up to Lv 60 obey.",
        completion_condition="tm42_received",
        priority=1
    ),
    DirectObjective(
        id="norman_106",
        description="Exit Gym and receive HM03 Surf from Wally's father",
        action_type="dialogue",
        category="story",
        target_location="Petalburg City",
        navigation_hint="Wally's dad finds you outside. Gives HM03 Surf as thanks. Essential HM!",
        completion_condition="hm03_received",
        priority=1
    ),
    DirectObjective(
        id="norman_107",
        description="Teach Surf to a Water-type Pokemon",
        action_type="menu",
        category="story",
        target_location="Petalburg City",
        navigation_hint="Bag > TMs/HMs > HM03 > Teach to Water type. Enables water travel. Strong move too.",
        completion_condition="surf_taught",
        priority=1
    ),
    DirectObjective(
        id="fortree_108",
        description="Travel to Route 118 (east of Mauville)",
        action_type="navigate",
        category="story",
        target_location="Route 118",
        navigation_hint="Go to Mauville, exit east. Use Surf to cross water to eastern side.",
        completion_condition="route118_reached",
        priority=1
    ),
    DirectObjective(
        id="fortree_109",
        description="Surf east across Route 118 water",
        action_type="navigate",
        category="story",
        target_location="Route 118 East",
        navigation_hint="Use Surf on water. Cross to eastern side. New areas unlock!",
        completion_condition="route118_surfed",
        priority=1
    ),
    DirectObjective(
        id="fortree_110",
        description="Travel north through Route 119 (long rainy route)",
        action_type="navigate",
        category="story",
        target_location="Route 119",
        navigation_hint="Long route with constant rain. Many trainers. Weather Institute at north.",
        completion_condition="route119_traversed",
        priority=1
    ),
    DirectObjective(
        id="fortree_111",
        description="Enter Weather Institute (Team Aqua takeover)",
        action_type="navigate",
        category="story",
        target_location="Weather Institute",
        navigation_hint="Building in north Route 119. Team Aqua has taken over.",
        completion_condition="weather_institute_entered",
        priority=1
    ),
    DirectObjective(
        id="fortree_112",
        description="Battle Team Aqua Grunts inside Weather Institute",
        action_type="battle",
        category="story",
        target_location="Weather Institute",
        navigation_hint="Clear grunts on both floors. Standard Aqua teams.",
        completion_condition="weather_grunts_defeated",
        priority=1
    ),
    DirectObjective(
        id="fortree_113",
        description="Battle Team Aqua Admin Shelly",
        action_type="battle",
        category="story",
        target_location="Weather Institute 2F",
        navigation_hint="Shelly: Carvanha Lv28, Mightyena Lv28. Admin battle.",
        completion_condition="shelly_defeated",
        priority=1
    ),
    DirectObjective(
        id="fortree_114",
        description="Receive Castform from Weather Institute scientist",
        action_type="dialogue",
        category="story",
        target_location="Weather Institute 2F",
        navigation_hint="Scientist gives Castform (changes type with weather). Unique Pokemon.",
        completion_condition="castform_received",
        priority=1
    ),
    DirectObjective(
        id="fortree_115",
        description="Exit Institute and battle Rival",
        action_type="battle",
        category="story",
        target_location="Route 119 North",
        navigation_hint="Rival appears as you leave. Evolved team: Starter ~Lv29, Pelipper, plus one ~Lv27.",
        completion_condition="rival_battle_3_won",
        priority=1
    ),
    DirectObjective(
        id="fortree_116",
        description="Receive HM02 Fly from Rival",
        action_type="dialogue",
        category="story",
        target_location="Route 119 North",
        navigation_hint="Rival gives HM02 Fly. Fast travel to visited Pokemon Centers!",
        completion_condition="hm02_received",
        priority=1
    ),
    DirectObjective(
        id="fortree_117",
        description="Teach Fly to a Flying-type Pokemon",
        action_type="menu",
        category="story",
        target_location="Route 119",
        navigation_hint="Teach to Swellow, Pelipper, etc. Register for quick travel. Game changer!",
        completion_condition="fly_taught",
        priority=1
    ),
    DirectObjective(
        id="fortree_118",
        description="Continue to Fortree City",
        action_type="navigate",
        category="story",
        target_location="Fortree City",
        navigation_hint="Continue north from Weather Institute. Cross bridges to treehouse city.",
        completion_condition="fortree_reached",
        priority=1
    ),
    DirectObjective(
        id="fortree_119",
        description="Heal at Fortree Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Fortree Pokemon Center",
        navigation_hint="Pokemon Center on one of the treehouses.",
        completion_condition="healed_fortree",
        priority=1
    ),
    DirectObjective(
        id="fortree_120",
        description="Try to enter Gym - blocked by invisible obstacle",
        action_type="navigate",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Something invisible blocks gym entrance. Need Devon Scope.",
        completion_condition="gym_blocked_discovered",
        priority=1
    ),
    DirectObjective(
        id="fortree_121",
        description="Go east to Route 120 to find Steven",
        action_type="navigate",
        category="story",
        target_location="Route 120",
        navigation_hint="Exit Fortree east. Steven is on a bridge with same invisible obstacle.",
        completion_condition="steven_route120_found",
        priority=1
    ),
    DirectObjective(
        id="fortree_122",
        description="Receive Devon Scope from Steven",
        action_type="dialogue",
        category="story",
        target_location="Route 120 Bridge",
        navigation_hint="Steven reveals obstacle is Kecleon. Gives Devon Scope to reveal invisible Kecleon.",
        completion_condition="devon_scope_received",
        priority=1
    ),
    DirectObjective(
        id="fortree_123",
        description="Battle the invisible Kecleon on Route 120",
        action_type="battle",
        category="story",
        target_location="Route 120 Bridge",
        navigation_hint="Use Devon Scope, then battle/catch Kecleon Lv30. Color Change ability.",
        completion_condition="kecleon_route120_battled",
        priority=1
    ),
    DirectObjective(
        id="fortree_124",
        description="Return to Fortree Gym entrance",
        action_type="navigate",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Go back to Fortree. Use Devon Scope on gym obstacle.",
        completion_condition="returned_fortree_gym",
        priority=1
    ),
    DirectObjective(
        id="fortree_125",
        description="Use Devon Scope and battle Kecleon blocking Gym",
        action_type="battle",
        category="story",
        target_location="Fortree Gym Entrance",
        navigation_hint="Use Devon Scope from Key Items. Battle Kecleon to clear path.",
        completion_condition="gym_kecleon_cleared",
        priority=1
    ),
    DirectObjective(
        id="fortree_126",
        description="Enter Fortree Gym",
        action_type="navigate",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Flying-type gym. Rotating door puzzle - step on switches to rotate doors.",
        completion_condition="fortree_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="fortree_127",
        description="Solve rotating door puzzle to reach Winona",
        action_type="navigate",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Step on switches to rotate doors. Create path through.",
        completion_condition="fortree_puzzle_solved",
        priority=1
    ),
    DirectObjective(
        id="fortree_128",
        description="Battle Gym Leader Winona for Feather Badge",
        action_type="battle",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Winona: Swablu Lv29, Tropius Lv29, Pelipper Lv30, Skarmory Lv31, Altaria Lv33. ALTARIA KNOWS DRAGON DANCE - KO fast before it sets up! Electric types excel.",
        completion_condition="feather_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_034", "battle_032", "battle_033", "battle_035"]
    ),
    DirectObjective(
        id="fortree_129",
        description="Receive TM40 Aerial Ace from Winona",
        action_type="dialogue",
        category="story",
        target_location="Fortree Gym",
        navigation_hint="Badge allows Fly outside. Pokemon up to Lv 70 obey.",
        completion_condition="tm40_received",
        priority=1
    ),
    DirectObjective(
        id="pyre_130",
        description="Continue south on Route 120",
        action_type="navigate",
        category="story",
        target_location="Route 120 South",
        navigation_hint="From Fortree, go east then south. Safari Zone entrance nearby.",
        completion_condition="route120_south_traversed",
        priority=1
    ),
    DirectObjective(
        id="pyre_131",
        description="Travel through Route 121 to Lilycove City",
        action_type="navigate",
        category="story",
        target_location="Lilycove City",
        navigation_hint="Continue east. Large coastal city ahead.",
        completion_condition="lilycove_reached",
        priority=1
    ),
    DirectObjective(
        id="pyre_132",
        description="Heal at Lilycove Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Lilycove Pokemon Center",
        navigation_hint="Pokemon Center near entrance.",
        completion_condition="healed_lilycove",
        priority=1
    ),
    DirectObjective(
        id="pyre_133",
        description="Check Team Aqua Hideout (blocked for now)",
        action_type="navigate",
        category="story",
        target_location="Team Aqua Hideout Entrance",
        navigation_hint="Northeast of Lilycove. Grunts block entry for now.",
        completion_condition="aqua_hideout_blocked",
        priority=1
    ),
    DirectObjective(
        id="pyre_134",
        description="Surf south from Route 121 to Route 122 and Mt. Pyre",
        action_type="navigate",
        category="story",
        target_location="Mt. Pyre",
        navigation_hint="From Route 121 water, Surf south to Route 122. Mt. Pyre is island in center.",
        completion_condition="mt_pyre_reached",
        priority=1
    ),
    DirectObjective(
        id="pyre_135",
        description="Enter Mt. Pyre interior",
        action_type="navigate",
        category="story",
        target_location="Mt. Pyre Interior",
        navigation_hint="Enter building. Multiple grave floors. Ghost types common.",
        completion_condition="mt_pyre_entered",
        priority=1
    ),
    DirectObjective(
        id="pyre_136",
        description="Climb through Mt. Pyre interior to summit",
        action_type="navigate",
        category="story",
        target_location="Mt. Pyre Summit",
        navigation_hint="Navigate up floors. Exit to outdoor summit area. Continue up with graves.",
        completion_condition="mt_pyre_summit_reached",
        priority=1
    ),
    DirectObjective(
        id="pyre_137",
        description="Battle Team Aqua at summit",
        action_type="battle",
        category="story",
        target_location="Mt. Pyre Summit",
        navigation_hint="Aqua grunts guard summit. Defeat them to reach old couple.",
        completion_condition="summit_aqua_defeated",
        priority=1
    ),
    DirectObjective(
        id="pyre_138",
        description="Learn Blue Orb was stolen by Team Aqua",
        action_type="dialogue",
        category="story",
        target_location="Mt. Pyre Summit",
        navigation_hint="Old couple explains: Aqua stole Blue Orb (controls Kyogre). Red Orb still here.",
        completion_condition="blue_orb_stolen_learned",
        priority=1
    ),
    DirectObjective(
        id="pyre_139",
        description="Receive Magma Emblem from old couple",
        action_type="dialogue",
        category="story",
        target_location="Mt. Pyre Summit",
        navigation_hint="Receive Magma Emblem. Opens Team Magma's hideout in Jagged Pass.",
        completion_condition="magma_emblem_received",
        priority=1
    ),
    DirectObjective(
        id="pyre_140",
        description="Return to Lilycove City",
        action_type="navigate",
        category="story",
        target_location="Lilycove City",
        navigation_hint="Surf back north. Team Aqua Hideout now accessible.",
        completion_condition="returned_lilycove",
        priority=1
    ),
    DirectObjective(
        id="pyre_141",
        description="Enter Team Aqua Hideout (now open)",
        action_type="navigate",
        category="story",
        target_location="Team Aqua Hideout",
        navigation_hint="Northeast of Lilycove. Guards gone. Enter cave system.",
        completion_condition="aqua_hideout_entered",
        priority=1
    ),
    DirectObjective(
        id="pyre_142",
        description="Navigate through Team Aqua Hideout (warp puzzle)",
        action_type="navigate",
        category="story",
        target_location="Team Aqua Hideout Interior",
        navigation_hint="Large cave with warp pads. Multiple grunts. Find submarine bay.",
        completion_condition="aqua_hideout_navigated",
        priority=1
    ),
    DirectObjective(
        id="pyre_143",
        description="Battle Team Aqua Admin Matt in hideout",
        action_type="battle",
        category="story",
        target_location="Team Aqua Hideout",
        navigation_hint="Matt guards key area. Evolved team with Mightyena, Sharpedo.",
        completion_condition="matt_hideout_defeated",
        priority=1
    ),
    DirectObjective(
        id="pyre_144",
        description="Reach submarine bay - Aqua escapes to Seafloor Cavern",
        action_type="dialogue",
        category="story",
        target_location="Team Aqua Submarine Bay",
        navigation_hint="Find submarine bay too late. Aqua left in submarine to awaken Kyogre.",
        completion_condition="submarine_escape_witnessed",
        priority=1
    ),
    DirectObjective(
        id="pyre_145",
        description="Exit Team Aqua Hideout",
        action_type="navigate",
        category="story",
        target_location="Lilycove City",
        navigation_hint="Leave hideout. Next: Mossdeep City via Route 124.",
        completion_condition="aqua_hideout_exited",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_146",
        description="Surf east from Lilycove through Route 124",
        action_type="navigate",
        category="story",
        target_location="Route 124",
        navigation_hint="Surf east. Ocean route with swimmers and divers.",
        completion_condition="route124_traversed",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_147",
        description="Arrive at Mossdeep City",
        action_type="navigate",
        category="story",
        target_location="Mossdeep City",
        navigation_hint="Island city with Space Center and Gym. Psychic double battle Gym.",
        completion_condition="mossdeep_reached",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_148",
        description="Heal at Mossdeep Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Mossdeep Pokemon Center",
        navigation_hint="Pokemon Center near Surf landing.",
        completion_condition="healed_mossdeep",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_149",
        description="Visit Steven's house to receive HM08 Dive",
        action_type="dialogue",
        category="story",
        target_location="Steven's House",
        navigation_hint="Northwest Mossdeep. Steven gives HM08 Dive. Underwater exploration!",
        completion_condition="hm08_received",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_150",
        description="Teach Dive to a Water-type Pokemon",
        action_type="menu",
        category="story",
        target_location="Mossdeep City",
        navigation_hint="Teach HM08 Dive. Enables diving into dark water spots.",
        completion_condition="dive_taught",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_151",
        description="Enter Mossdeep Gym",
        action_type="navigate",
        category="story",
        target_location="Mossdeep Gym",
        navigation_hint="Psychic-type gym. DOUBLE BATTLE format! Arrow tile puzzle.",
        completion_condition="mossdeep_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_152",
        description="Navigate arrow tile puzzle to Tate & Liza",
        action_type="navigate",
        category="story",
        target_location="Mossdeep Gym",
        navigation_hint="Step on arrows to move automatically. Find path to twins.",
        completion_condition="mossdeep_puzzle_solved",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_153",
        description="Battle Gym Leaders Tate & Liza for Mind Badge (DOUBLE BATTLE)",
        action_type="battle",
        category="story",
        target_location="Mossdeep Gym",
        navigation_hint="DOUBLE BATTLE! Tate & Liza: Claydol Lv41, Xatu Lv41, Lunatone Lv42, Solrock Lv42. Solrock+Lunatone use Sunny Day + Solar Beam combo. Dark types resist Psychic. Surf hits both enemies!",
        completion_condition="mind_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_040", "battle_039", "battle_032", "battle_038", "battle_041"]
    ),
    DirectObjective(
        id="mossdeep_154",
        description="Receive TM04 Calm Mind from Tate & Liza",
        action_type="dialogue",
        category="story",
        target_location="Mossdeep Gym",
        navigation_hint="Badge allows Dive outside. Pokemon up to Lv 80 obey.",
        completion_condition="tm04_received",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_155",
        description="Team Magma attacks Space Center - go there",
        action_type="navigate",
        category="story",
        target_location="Mossdeep Space Center",
        navigation_hint="Event triggers after Gym. Team Magma attacking. Go to Space Center!",
        completion_condition="space_center_attack_triggered",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_156",
        description="Battle Team Magma inside Space Center",
        action_type="battle",
        category="story",
        target_location="Mossdeep Space Center",
        navigation_hint="Magma grunts inside. Steven also fighting them.",
        completion_condition="space_center_magma_defeated",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_157",
        description="Double battle with Steven vs Maxie and Tabitha",
        action_type="battle",
        category="story",
        target_location="Mossdeep Space Center",
        navigation_hint="Tag team with Steven! Face Magma Leader Maxie + Admin Tabitha. Steven helps with strong Pokemon.",
        completion_condition="maxie_tabitha_defeated",
        priority=1
    ),
    DirectObjective(
        id="mossdeep_158",
        description="Team Magma retreats, mentions Seafloor Cavern",
        action_type="dialogue",
        category="story",
        target_location="Mossdeep Space Center",
        navigation_hint="After defeat, Magma flees. Both teams heading to Seafloor Cavern.",
        completion_condition="magma_space_retreat",
        priority=1
    ),
    DirectObjective(
        id="seafloor_159",
        description="Surf south from Mossdeep to Route 127",
        action_type="navigate",
        category="story",
        target_location="Route 127",
        navigation_hint="Surf south and slightly west. Ocean route.",
        completion_condition="route127_reached",
        priority=1
    ),
    DirectObjective(
        id="seafloor_160",
        description="Continue to Route 128",
        action_type="navigate",
        category="story",
        target_location="Route 128",
        navigation_hint="Continue south. Seafloor Cavern entrance is underwater here.",
        completion_condition="route128_reached",
        priority=1
    ),
    DirectObjective(
        id="seafloor_161",
        description="Find dark water dive spot and use Dive",
        action_type="navigate",
        category="story",
        target_location="Route 128 Underwater",
        navigation_hint="Look for dark water patches. Use Dive to go underwater.",
        completion_condition="dive_spot_found",
        priority=1
    ),
    DirectObjective(
        id="seafloor_162",
        description="Navigate underwater to Seafloor Cavern entrance",
        action_type="navigate",
        category="story",
        target_location="Underwater Route 128",
        navigation_hint="Underwater maze. Find cave entrance marked by rocks. Surface inside.",
        completion_condition="seafloor_entrance_found",
        priority=1
    ),
    DirectObjective(
        id="seafloor_163",
        description="Surface inside Seafloor Cavern",
        action_type="navigate",
        category="story",
        target_location="Seafloor Cavern",
        navigation_hint="Use Dive again to surface. Team Aqua's submarine visible.",
        completion_condition="seafloor_cavern_entered",
        priority=1
    ),
    DirectObjective(
        id="seafloor_164",
        description="Navigate through Seafloor Cavern (Strength puzzles)",
        action_type="navigate",
        category="story",
        target_location="Seafloor Cavern Interior",
        navigation_hint="Requires Strength for boulders, Rock Smash for rocks. Complex puzzle rooms.",
        completion_condition="seafloor_puzzles_solved",
        priority=1
    ),
    DirectObjective(
        id="seafloor_165",
        description="Battle Team Aqua Grunts in cavern",
        action_type="battle",
        category="story",
        target_location="Seafloor Cavern",
        navigation_hint="Grunts throughout. Standard Aqua teams.",
        completion_condition="seafloor_grunts_defeated",
        priority=1
    ),
    DirectObjective(
        id="seafloor_166",
        description="Find Archie at deepest part of cavern",
        action_type="navigate",
        category="story",
        target_location="Seafloor Cavern Depths",
        navigation_hint="Navigate to back. Archie before Kyogre's chamber.",
        completion_condition="archie_found_seafloor",
        priority=1
    ),
    DirectObjective(
        id="seafloor_167",
        description="Battle Team Aqua Leader Archie",
        action_type="battle",
        category="story",
        target_location="Seafloor Cavern Depths",
        navigation_hint="Archie: Mightyena Lv41, Crobat Lv41, Sharpedo Lv43. Important battle!",
        completion_condition="archie_defeated_seafloor",
        priority=1
    ),
    DirectObjective(
        id="seafloor_168",
        description="Watch Archie awaken Kyogre with Blue Orb",
        action_type="dialogue",
        category="story",
        target_location="Seafloor Cavern Chamber",
        navigation_hint="Archie uses Blue Orb. Kyogre awakens! Massive rainfall begins. Kyogre escapes.",
        completion_condition="kyogre_awakened",
        priority=1
    ),
    DirectObjective(
        id="seafloor_169",
        description="Archie realizes mistake - go to Sootopolis",
        action_type="dialogue",
        category="story",
        target_location="Seafloor Cavern Chamber",
        navigation_hint="Archie regrets awakening Kyogre. Tells you to go to Sootopolis.",
        completion_condition="archie_regret_witnessed",
        priority=1
    ),
    DirectObjective(
        id="seafloor_170",
        description="Exit Seafloor Cavern",
        action_type="navigate",
        category="story",
        target_location="Route 128",
        navigation_hint="Navigate back out. Heavy rain everywhere.",
        completion_condition="seafloor_exited",
        priority=1
    ),
    DirectObjective(
        id="seafloor_171",
        description="Navigate to Route 126 around Sootopolis",
        action_type="navigate",
        category="story",
        target_location="Route 126",
        navigation_hint="From Route 127, go west. Route 126 surrounds Sootopolis crater.",
        completion_condition="route126_reached",
        priority=1
    ),
    DirectObjective(
        id="seafloor_172",
        description="Dive at Route 126 to enter Sootopolis City",
        action_type="navigate",
        category="story",
        target_location="Route 126 Underwater",
        navigation_hint="Find dive spot near central crater. Underwater path leads into city.",
        completion_condition="sootopolis_underwater_path",
        priority=1
    ),
    DirectObjective(
        id="seafloor_173",
        description="Surface inside Sootopolis City",
        action_type="navigate",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Surface inside volcanic crater. Kyogre and Groudon fighting in center!",
        completion_condition="sootopolis_entered",
        priority=1
    ),
    DirectObjective(
        id="seafloor_174",
        description="Witness Kyogre vs Groudon battle",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis City Center",
        navigation_hint="Legendary battle in progress! Weather cycling between rain and sun.",
        completion_condition="legendary_battle_witnessed",
        priority=1
    ),
    DirectObjective(
        id="seafloor_175",
        description="Find Steven and Wallace near Cave of Origin",
        action_type="navigate",
        category="story",
        target_location="Cave of Origin Entrance",
        navigation_hint="Steven and Wallace (Gym Leader) discussing crisis near cave.",
        completion_condition="steven_wallace_found",
        priority=1
    ),
    DirectObjective(
        id="seafloor_176",
        description="Learn you must awaken Rayquaza at Sky Pillar",
        action_type="dialogue",
        category="story",
        target_location="Cave of Origin Entrance",
        navigation_hint="Wallace says only Rayquaza can stop them. Go to Sky Pillar on Route 131.",
        completion_condition="sky_pillar_quest_received",
        priority=1
    ),
    DirectObjective(
        id="sky_177",
        description="Surf to Route 131 to reach Sky Pillar",
        action_type="navigate",
        category="story",
        target_location="Route 131",
        navigation_hint="From Sootopolis, surf west through Route 130 to Route 131. Sky Pillar entrance here.",
        completion_condition="route131_reached",
        priority=1
    ),
    DirectObjective(
        id="sky_178",
        description="Enter Sky Pillar (Wallace opens door)",
        action_type="navigate",
        category="story",
        target_location="Sky Pillar",
        navigation_hint="Tall ancient tower. Wallace meets you, opens door. Begin climb.",
        completion_condition="sky_pillar_entered",
        priority=1
    ),
    DirectObjective(
        id="sky_179",
        description="Navigate Sky Pillar 1F-2F",
        action_type="navigate",
        category="story",
        target_location="Sky Pillar 2F",
        navigation_hint="First floors are straightforward. Cracked tiles appear higher up.",
        completion_condition="sky_pillar_2f_cleared",
        priority=1
    ),
    DirectObjective(
        id="sky_180",
        description="Navigate Sky Pillar 3F-4F (cracked floor puzzle)",
        action_type="navigate",
        category="story",
        target_location="Sky Pillar 4F",
        navigation_hint="Cracked tiles fall when stepped on! Use Mach Bike to cross quickly before falling.",
        completion_condition="sky_pillar_4f_cleared",
        priority=1
    ),
    DirectObjective(
        id="sky_181",
        description="Reach Sky Pillar summit",
        action_type="navigate",
        category="story",
        target_location="Sky Pillar Summit",
        navigation_hint="Final floor leads to rooftop. Rayquaza awaits!",
        completion_condition="sky_pillar_summit_reached",
        priority=1
    ),
    DirectObjective(
        id="sky_182",
        description="Approach and awaken Rayquaza",
        action_type="dialogue",
        category="story",
        target_location="Sky Pillar Summit",
        navigation_hint="Interact with Rayquaza. It awakens and flies to Sootopolis!",
        completion_condition="rayquaza_awakened",
        priority=1
    ),
    DirectObjective(
        id="sky_183",
        description="Return to Sootopolis City",
        action_type="navigate",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Fly or surf back. Weather should be calmer.",
        completion_condition="returned_sootopolis",
        priority=1
    ),
    DirectObjective(
        id="sky_184",
        description="Watch Rayquaza stop Kyogre and Groudon",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis City Center",
        navigation_hint="Rayquaza arrives, calms both legendaries. They retreat to caves.",
        completion_condition="rayquaza_intervention_witnessed",
        priority=1
    ),
    DirectObjective(
        id="sky_185",
        description="Talk to Archie and Maxie - they apologize",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Both leaders realize mistakes. Teams mostly disband. Weather normal.",
        completion_condition="crisis_resolved",
        priority=1
    ),
    DirectObjective(
        id="sky_186",
        description="Talk to Wallace - can now challenge Gym",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Wallace thanks you. Sootopolis Gym now open.",
        completion_condition="wallace_thanks",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_187",
        description="Heal at Sootopolis Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Sootopolis Pokemon Center",
        navigation_hint="Prepare for final Gym battle.",
        completion_condition="healed_sootopolis",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_188",
        description="Enter Sootopolis Gym",
        action_type="navigate",
        category="story",
        target_location="Sootopolis Gym",
        navigation_hint="Water-type gym. Ice stepping puzzle - step on each tile once.",
        completion_condition="sootopolis_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_189",
        description="Solve ice stepping puzzle",
        action_type="navigate",
        category="story",
        target_location="Sootopolis Gym",
        navigation_hint="Step on each ice tile exactly once. Creates stairs. Multiple sections.",
        completion_condition="sootopolis_puzzle_solved",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_190",
        description="Battle Gym Leader Juan for Rain Badge",
        action_type="battle",
        category="story",
        target_location="Sootopolis Gym",
        navigation_hint="Juan: Luvdisc Lv41, Whiscash Lv41, Sealeo Lv43, Crawdaunt Lv43, Kingdra Lv46. KINGDRA ONLY WEAK TO DRAGON. High SpDef, knows Water Pulse. Electric/Grass good for others.",
        completion_condition="rain_badge_obtained",
        priority=1,
        recommended_battling_objectives=["battle_044", "battle_043"]
    ),
    DirectObjective(
        id="sootopolis_191",
        description="Receive TM03 Water Pulse from Juan - ALL 8 BADGES!",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis Gym",
        navigation_hint="Rain Badge allows Waterfall outside. All Pokemon obey. You have all 8 badges!",
        completion_condition="tm03_received",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_192",
        description="Receive HM07 Waterfall from Wallace",
        action_type="dialogue",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Wallace gives HM07 Waterfall. Climb waterfalls. Needed for Victory Road.",
        completion_condition="hm07_received",
        priority=1
    ),
    DirectObjective(
        id="sootopolis_193",
        description="Teach Waterfall to a Water-type Pokemon",
        action_type="menu",
        category="story",
        target_location="Sootopolis City",
        navigation_hint="Teach HM07. Enables ascending waterfalls.",
        completion_condition="waterfall_taught",
        priority=1
    ),
    DirectObjective(
        id="league_194",
        description="Navigate to Route 128 and head east to Ever Grande City",
        action_type="navigate",
        category="story",
        target_location="Ever Grande City",
        navigation_hint="From Sootopolis, go to Route 128, continue east. Use Waterfall to climb up.",
        completion_condition="ever_grande_reached",
        priority=1
    ),
    DirectObjective(
        id="league_195",
        description="Heal at Ever Grande Pokemon Center",
        action_type="interact",
        category="story",
        target_location="Ever Grande Pokemon Center",
        navigation_hint="Last Pokemon Center before Victory Road.",
        completion_condition="healed_ever_grande",
        priority=1
    ),
    DirectObjective(
        id="league_196",
        description="Pass through badge check gates",
        action_type="navigate",
        category="story",
        target_location="Ever Grande Badge Gates",
        navigation_hint="8 gates, each checks for a badge. Walk through all.",
        completion_condition="badge_check_passed",
        priority=1
    ),
    DirectObjective(
        id="league_197",
        description="Enter Victory Road",
        action_type="navigate",
        category="story",
        target_location="Victory Road",
        navigation_hint="Long cave. Requires Surf, Strength, Rock Smash. Strong wild Pokemon and trainers.",
        completion_condition="victory_road_entered",
        priority=1
    ),
    DirectObjective(
        id="league_198",
        description="Navigate Victory Road lower floors (boulder puzzles)",
        action_type="navigate",
        category="story",
        target_location="Victory Road B1F/1F",
        navigation_hint="Push boulders into holes to create paths. Strong trainers.",
        completion_condition="victory_road_lower_cleared",
        priority=1
    ),
    DirectObjective(
        id="league_199",
        description="Battle Wally in Victory Road",
        action_type="battle",
        category="story",
        target_location="Victory Road",
        navigation_hint="Wally challenges near exit. Team: Altaria Lv44, Delcatty Lv44, Roselia Lv44, Magneton Lv44, Gardevoir Lv45. Gardevoir is ace.",
        completion_condition="wally_victory_road_defeated",
        priority=1
    ),
    DirectObjective(
        id="league_200",
        description="Navigate Victory Road upper floors to exit",
        action_type="navigate",
        category="story",
        target_location="Pokemon League Entrance",
        navigation_hint="Continue through upper sections. Exit to Pokemon League plateau.",
        completion_condition="victory_road_exited",
        priority=1
    ),
    DirectObjective(
        id="league_201",
        description="Heal at final Pokemon Center and save game",
        action_type="interact",
        category="story",
        target_location="Pokemon League",
        navigation_hint="LAST CHANCE TO HEAL AND SAVE. Cannot exit once you enter Elite Four!",
        completion_condition="final_heal_done",
        priority=1
    ),
    DirectObjective(
        id="league_202",
        description="Enter Pokemon League building - BEGIN ELITE FOUR",
        action_type="navigate",
        category="story",
        target_location="Pokemon League Building",
        navigation_hint="No turning back! 5 consecutive battles: 4 Elite Four + Champion.",
        completion_condition="elite_four_entered",
        priority=1
    ),
    DirectObjective(
        id="league_203",
        description="Battle Elite Four Sidney (Dark-type)",
        action_type="battle",
        category="story",
        target_location="Elite Four Room 1",
        navigation_hint="Sidney: Mightyena Lv46, Shiftry Lv48, Cacturne Lv46, Crawdaunt Lv48, Absol Lv49. Fighting and Bug types excellent.",
        completion_condition="sidney_defeated",
        priority=1,
        recommended_battling_objectives=["battle_047", "battle_046", "battle_039", "battle_011", "battle_045", "battle_048", "battle_049", "battle_050"]
    ),
    DirectObjective(
        id="league_204",
        description="Battle Elite Four Phoebe (Ghost-type)",
        action_type="battle",
        category="story",
        target_location="Elite Four Room 2",
        navigation_hint="Phoebe: Dusclops Lv48, Banette Lv49, Sableye Lv50, Banette Lv49, Dusclops Lv51. Dark types strong. Watch Confuse Ray.",
        completion_condition="phoebe_defeated",
        priority=1,
        recommended_battling_objectives=["battle_047", "battle_046", "battle_039", "battle_011", "battle_045", "battle_048", "battle_049", "battle_050"]
    ),
    DirectObjective(
        id="league_205",
        description="Battle Elite Four Glacia (Ice-type)",
        action_type="battle",
        category="story",
        target_location="Elite Four Room 3",
        navigation_hint="Glacia: Sealeo Lv50, Glalie Lv50, Sealeo Lv52, Glalie Lv52, Walrein Lv53. Fire, Fighting, Rock, Steel effective.",
        completion_condition="glacia_defeated",
        priority=1,
        recommended_battling_objectives=["battle_047", "battle_046", "battle_039", "battle_011", "battle_045", "battle_048", "battle_049", "battle_050"]
    ),
    DirectObjective(
        id="league_206",
        description="Battle Elite Four Drake (Dragon-type)",
        action_type="battle",
        category="story",
        target_location="Elite Four Room 4",
        navigation_hint="Drake: Shelgon Lv52, Altaria Lv54, Flygon Lv53, Flygon Lv53, Salamence Lv55. ICE TYPES DESTROY DRAGONS. Watch Fire coverage.",
        completion_condition="drake_defeated",
        priority=1,
        recommended_battling_objectives=["battle_047", "battle_046", "battle_039", "battle_011", "battle_045", "battle_048", "battle_049", "battle_050"]
    ),
    DirectObjective(
        id="league_207",
        description="Enter Champion's chamber",
        action_type="navigate",
        category="story",
        target_location="Champion's Chamber",
        navigation_hint="Final room. Champion Wallace awaits. Heal items only, no Pokemon Center.",
        completion_condition="champion_chamber_entered",
        priority=1
    ),
    DirectObjective(
        id="league_208",
        description="Battle Champion Wallace",
        action_type="battle",
        category="story",
        target_location="Champion's Chamber",
        navigation_hint="Wallace: Wailord Lv57, Tentacruel Lv55, Ludicolo Lv56, Whiscash Lv56, Gyarados Lv56, Milotic Lv58. MILOTIC IS TOUGH - high SpDef, Recover, Ice Beam. Electric/Grass recommended.",
        completion_condition="champion_wallace_defeated",
        priority=1,
        recommended_battling_objectives=["battle_047", "battle_046", "battle_039", "battle_011", "battle_045", "battle_048", "battle_049", "battle_050"]
    ),
    DirectObjective(
        id="league_209",
        description="Enter Hall of Fame - CONGRATULATIONS!",
        action_type="dialogue",
        category="story",
        target_location="Hall of Fame",
        navigation_hint="Wallace leads you to Hall of Fame. Your Pokemon are recorded. Credits roll. MAIN STORY COMPLETE!",
        completion_condition="hall_of_fame_entered",
        priority=1
    ),
    DirectObjective(
        id="post_210",
        description="Battle Frontier unlocks - talk to Scott",
        action_type="dialogue",
        category="story",
        target_location="Littleroot Town",
        navigation_hint="After credits, you're home. Scott invites you to Battle Frontier.",
        completion_condition="battle_frontier_unlocked",
        priority=2,
        optional=True
    ),
    DirectObjective(
        id="post_211",
        description="Explore Battle Frontier (7 facilities)",
        action_type="navigate",
        category="story",
        target_location="Battle Frontier",
        navigation_hint="Take SS Tidal from Slateport/Lilycove. Tower, Dome, Palace, Arena, Factory, Pike, Pyramid. Earn Symbols!",
        completion_condition="battle_frontier_explored",
        priority=2,
        optional=True
    ),
]


# Battling objectives (55 total)
BATTLING_OBJECTIVES = [
    DirectObjective(
        id="battle_000",
        description="Train starter to Lv 7+ by battling wild Pokemon",
        action_type="battle",
        category="battling",
        target_location="Route 101/103",
        navigation_hint="Battle wild Zigzagoon, Poochyena, Wurmple in grass. Reach Lv 7 minimum before rival battle.",
        completion_condition="starter_level_7",
        priority=1,
        prerequisite_story_objective="early_011"
    ),
    DirectObjective(
        id="battle_001",
        description="Buy Great Balls and Antidotes at Oldale Poke Mart",
        action_type="shop",
        category="battling",
        target_location="Oldale Poke Mart",
        navigation_hint="Buy: 5 Great Balls (3000), 3 Antidotes (300). Total ~3300. Great Balls have better catch rate than Poke Balls.",
        completion_condition="oldale_shopping_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="early_011"
    ),
    DirectObjective(
        id="battle_002",
        description="Heal team at Oldale Pokemon Center before Route 102",
        action_type="interact",
        category="battling",
        target_location="Oldale Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Recommended before continuing to Route 102.",
        completion_condition="oldale_heal_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="early_011"
    ),
    DirectObjective(
        id="battle_003",
        description="Manage PC and organize team before Route 103 rival battle",
        action_type="menu",
        category="battling",
        target_location="Oldale Pokemon Center",
        navigation_hint="Use PC to organize your party. Place strongest Pokemon in lead position for rival battle.",
        completion_condition="team_organized_rival",
        priority=2,
        optional=True,
        prerequisite_story_objective="early_011"
    ),
    DirectObjective(
        id="battle_004",
        description="Heal at Petalburg Pokemon Center before Petalburg Woods",
        action_type="interact",
        category="battling",
        target_location="Petalburg Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Recommended before entering Petalburg Woods.",
        completion_condition="petalburg_heal_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="petalburg_019"
    ),
    DirectObjective(
        id="battle_005",
        description="Catch Ralts on Route 102 (highly recommended)",
        action_type="catch",
        category="battling",
        target_location="Route 102",
        navigation_hint="4% encounter rate - very rare. Walk in grass patiently. Lv 4. Weaken to red HP, throw Poke Ball. Evolves to Gardevoir - excellent Psychic type for entire game.",
        completion_condition="ralts_caught",
        priority=1,
        prerequisite_story_objective="petalburg_019"
    ),
    DirectObjective(
        id="battle_006",
        description="Catch Wingull or Taillow for Flying type coverage",
        action_type="catch",
        category="battling",
        target_location="Route 104",
        navigation_hint="Wingull (Water/Flying) common near water. Taillow (Normal/Flying) in grass. Both useful for Gym 2 (Fighting).",
        completion_condition="flying_type_caught",
        priority=1,
        prerequisite_story_objective="petalburg_019"
    ),
    DirectObjective(
        id="battle_007",
        description="Catch Shroomish in Petalburg Woods",
        action_type="catch",
        category="battling",
        target_location="Petalburg Woods",
        navigation_hint="15% encounter rate. Grass type good for Gym 1 (Rock). Evolves to Breloom (Grass/Fighting) - excellent!",
        completion_condition="shroomish_caught",
        priority=1,
        prerequisite_story_objective="petalburg_019"
    ),
    DirectObjective(
        id="battle_008",
        description="Buy Potions and Poke Balls at Rustboro Poke Mart",
        action_type="shop",
        category="battling",
        target_location="Rustboro Poke Mart",
        navigation_hint="Buy: 10 Potions (3000), 10 Poke Balls (2000), 3 Antidotes (300). Total ~5300.",
        completion_condition="rustboro_shopping_done",
        priority=1,
        prerequisite_story_objective="rustboro_030"
    ),
    DirectObjective(
        id="battle_009",
        description="Train team to Lv 13-14 for Roxanne",
        action_type="battle",
        category="battling",
        target_location="Route 104 / Route 116",
        navigation_hint="Grind in grass. Roxanne's ace is Lv 15 Nosepass. Need Lv 13+ minimum. Water/Grass types ideal. Must be all Pokemon on team, not just lead. Make sure to switch.",
        completion_condition="team_level_13",
        priority=1,
        prerequisite_story_objective="rustboro_030"
    ),
    DirectObjective(
        id="battle_010",
        description="Heal at Rustboro Pokemon Center before entering gym",
        action_type="interact",
        category="battling",
        target_location="Rustboro Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Roxanne.",
        completion_condition="rustboro_heal_done",
        priority=1,
        prerequisite_story_objective="rustboro_030"
    ),
    DirectObjective(
        id="battle_011",
        description="Catch Makuhita in Granite Cave (important for Gym 5)",
        action_type="catch",
        category="battling",
        target_location="Granite Cave B1F",
        navigation_hint="20% encounter rate. Fighting type essential for Norman later. Evolves to bulky Hariyama.",
        completion_condition="makuhita_caught",
        priority=1,
        prerequisite_story_objective="dewford_043"
    ),
    DirectObjective(
        id="battle_012",
        description="Optionally catch Abra (teleports - use Quick Ball)",
        action_type="catch",
        category="battling",
        target_location="Granite Cave 1F",
        navigation_hint="10% rate. Teleports turn 1! Quick Ball or status move required. Evolves to powerful Alakazam.",
        completion_condition="abra_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="dewford_043"
    ),
    DirectObjective(
        id="battle_013",
        description="Train team to Lv 17-18 for Brawly",
        action_type="battle",
        category="battling",
        target_location="Granite Cave / Route 106",
        navigation_hint="Brawly's ace is Lv 19 Makuhita. Need Flying or Psychic types. Train Ralts/Taillow/Wingull. Lvl 17-18 must be all Pokemon on team, not just lead. Make sure to switch.",
        completion_condition="team_level_17",
        priority=1,
        prerequisite_story_objective="dewford_043"
    ),
    DirectObjective(
        id="battle_014",
        description="Heal at Dewford Pokemon Center before gym",
        action_type="interact",
        category="battling",
        target_location="Dewford Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Brawly.",
        completion_condition="dewford_heal_done",
        priority=1,
        prerequisite_story_objective="dewford_043"
    ),
    DirectObjective(
        id="battle_015",
        description="Catch Electrike on Route 110 for Electric coverage",
        action_type="catch",
        category="battling",
        target_location="Route 110",
        navigation_hint="25% encounter rate. Fast Electric type. Useful for Gyms 6, 8, and Champion.",
        completion_condition="electrike_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_016",
        description="Catch Geodude for Ground type (Electric immunity) or evolve Shroomish into Breloom (Lv 23)",
        action_type="catch",
        category="battling",
        target_location="Route 111 / Granite Cave",
        navigation_hint="Ground types immune to Electric. Geodude/Graveler destroys Wattson. Also good Rock coverage.",
        completion_condition="ground_type_caught",
        priority=1,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_017",
        description="Train team to Lv 22-24 for Wattson",
        action_type="battle",
        category="battling",
        target_location="Route 110 / Route 117",
        navigation_hint="Wattson's ace is Lv 24 Manectric. Ground types essential. Train Marshtomp if you have it. Train full team for the gym. Switch lead if necessary.",
        completion_condition="team_level_22",
        priority=1,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_018",
        description="Buy Paralysis Heals before Mauville Gym",
        action_type="shop",
        category="battling",
        target_location="Mauville Poke Mart",
        navigation_hint="Buy: 5 Paralysis Heals (1000), 10 Super Potions (7000). Electric gym causes paralysis.",
        completion_condition="mauville_shopping_done",
        priority=1,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_019",
        description="Organize team to counter Wattson",
        action_type="menu",
        category="battling",
        target_location="Mauville Pokemon Center",
        navigation_hint="Use PC to organize team. Place Ground-type (Marshtomp/Geodude) in lead position. Electric attacks have no effect on Ground types.",
        completion_condition="team_organized_wattson",
        priority=1,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_020",
        description="Heal at Mauville Pokemon Center before gym",
        action_type="interact",
        category="battling",
        target_location="Mauville Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Wattson.",
        completion_condition="mauville_heal_done",
        priority=1,
        prerequisite_story_objective="mauville_068"
    ),
    DirectObjective(
        id="battle_021",
        description="Buy Super Potions and Full Heals at Lavaridge",
        action_type="shop",
        category="battling",
        target_location="Lavaridge Poke Mart",
        navigation_hint="Buy: 10 Super Potions (7000), 5 Full Heals (3000). Total ~10000. Full Heals cure all status conditions.",
        completion_condition="lavaridge_advanced_shopping_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="chimney_078"
    ),
    DirectObjective(
        id="battle_022",
        description="Buy Burn Heals at Lavaridge Poke Mart",
        action_type="shop",
        category="battling",
        target_location="Lavaridge Poke Mart",
        navigation_hint="Buy: 10 Burn Heals (2500), 10 Super Potions (7000). Fire gym causes burns.",
        completion_condition="lavaridge_shopping_done",
        priority=1,
        prerequisite_story_objective="chimney_078"
    ),
    DirectObjective(
        id="battle_023",
        description="Train team to Lv 26-28 for Flannery",
        action_type="battle",
        category="battling",
        target_location="Jagged Pass / Route 112",
        navigation_hint="Flannery's ace is Lv 29 Torkoal with Overheat. Need Water/Ground/Rock types.",
        completion_condition="team_level_26",
        priority=1,
        prerequisite_story_objective="chimney_078"
    ),
    DirectObjective(
        id="battle_024",
        description="Check team typing before Flannery battle",
        action_type="menu",
        category="battling",
        target_location="Lavaridge Town",
        navigation_hint="Review team composition. Ensure Water/Ground/Rock types are ready. Fire types will struggle.",
        completion_condition="team_checked_flannery",
        priority=2,
        optional=True,
        prerequisite_story_objective="chimney_078"
    ),
    DirectObjective(
        id="battle_025",
        description="Heal before Lavaridge Gym",
        action_type="interact",
        category="battling",
        target_location="Lavaridge Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Flannery.",
        completion_condition="lavaridge_heal_done",
        priority=1,
        prerequisite_story_objective="chimney_078"
    ),
    DirectObjective(
        id="battle_026",
        description="Organize team with Fighting type in battle-ready position",
        action_type="menu",
        category="battling",
        target_location="Petalburg Pokemon Center",
        navigation_hint="Use PC to organize team. Place Fighting-type (Hariyama/Breloom) in lead position. Fighting is super effective against Norman's Normal types.",
        completion_condition="team_organized_norman",
        priority=1,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_027",
        description="Buy strong healing items for Norman fight",
        action_type="shop",
        category="battling",
        target_location="Mauville/Petalburg Mart",
        navigation_hint="Buy: 10 Hyper Potions (12000), 5 Revives (7500). Norman hits HARD.",
        completion_condition="norman_shopping_done",
        priority=1,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_028",
        description="Ensure you have a Fighting type (Makuhita/Hariyama/Breloom)",
        action_type="catch",
        category="battling",
        target_location="Various",
        navigation_hint="Fighting is the ONLY type super effective against Normal. If no Makuhita, train Breloom or catch Machop.",
        completion_condition="fighting_type_ready",
        priority=1,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_029",
        description="Train team to Lv 29-31 for Norman",
        action_type="battle",
        category="battling",
        target_location="Route 118 / Route 117",
        navigation_hint="Norman's Slaking is Lv 31 and hits extremely hard. Need Lv 29+ minimum. Fighting type essential.",
        completion_condition="team_level_29",
        priority=1,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_030",
        description="Buy Max Potions and Revives before Norman",
        action_type="shop",
        category="battling",
        target_location="Mauville/Petalburg Mart",
        navigation_hint="Buy: 5 Max Potions (12500), 5 Revives (7500). Total ~20000. Norman's Slaking hits extremely hard.",
        completion_condition="norman_advanced_shopping_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_031",
        description="Heal at Pokemon Center before Petalburg Gym",
        action_type="interact",
        category="battling",
        target_location="Petalburg Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. CRITICAL before challenging Norman - toughest gym battle.",
        completion_condition="petalburg_heal_done",
        priority=1,
        prerequisite_story_objective="norman_100"
    ),
    DirectObjective(
        id="battle_032",
        description="Catch Absol on Route 120 (important for Gym 7)",
        action_type="catch",
        category="battling",
        target_location="Route 120",
        navigation_hint="8% encounter rate in tall grass. Dark type destroys Psychic gym. High attack.",
        completion_condition="absol_caught",
        priority=1,
        prerequisite_story_objective="fortree_108"
    ),
    DirectObjective(
        id="battle_033",
        description="Catch Dark-type if missing (Mt. Pyre area). If already caught, transfer from PC into party at the nearest pokemon center asap. Use fly if you can.",
        action_type="catch",
        category="battling",
        target_location="Mt. Pyre / Route 120",
        navigation_hint="Dark types essential for Gym 7 (Psychic). Absol on Route 120, or Shuppet at Mt. Pyre. Dark immune to Psychic.",
        completion_condition="dark_type_caught",
        priority=1,
        prerequisite_story_objective="fortree_108"
    ),
    DirectObjective(
        id="battle_034",
        description="Train team to Lv 32-34 for Winona",
        action_type="battle",
        category="battling",
        target_location="Route 119 / Route 120",
        navigation_hint="Winona's ace is Lv 33 Altaria with Dragon Dance. Need Electric/Ice/Rock types.",
        completion_condition="team_level_32",
        priority=1,
        prerequisite_story_objective="fortree_108"
    ),
    DirectObjective(
        id="battle_035",
        description="Heal before Fortree Gym battle",
        action_type="interact",
        category="battling",
        target_location="Fortree Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Winona.",
        completion_condition="fortree_heal_done",
        priority=1,
        prerequisite_story_objective="fortree_108"
    ),
    DirectObjective(
        id="battle_036",
        description="Catch Ghost type at Mt. Pyre (Shuppet or Duskull)",
        action_type="catch",
        category="battling",
        target_location="Mt. Pyre Interior",
        navigation_hint="Common encounters. Ghost types useful for Gym 7 (Psychic). Duskull very bulky.",
        completion_condition="ghost_type_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_037",
        description="Find Master Ball in hideout",
        action_type="item",
        category="battling",
        target_location="Team Aqua Hideout",
        navigation_hint="In one of the item rooms. MASTER BALL - 100% catch rate! Save for legendary!",
        completion_condition="master_ball_found",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_038",
        description="Buy Full Restores before Mossdeep Gym",
        action_type="shop",
        category="battling",
        target_location="Mossdeep Poke Mart",
        navigation_hint="Buy: 10 Full Restores (30000), 5 Revives (7500). Total ~37500. Double battle is very challenging.",
        completion_condition="mossdeep_shopping_done",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_039",
        description="Catch Spheal at Shoal Cave (Ice type for Drake)",
        action_type="catch",
        category="battling",
        target_location="Shoal Cave",
        navigation_hint="Shoal Cave on Route 125 (north of Mossdeep). Ice room at low tide. Spheal evolves to Walrein - great for Elite Four Drake (Dragons).",
        completion_condition="spheal_caught",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_040",
        description="Train team to Lv 40-42 for Tate & Liza DOUBLE BATTLE",
        action_type="battle",
        category="battling",
        target_location="Route 124 / Route 125",
        navigation_hint="Twins' aces are Lv 42 Lunatone/Solrock. DOUBLE BATTLE! Need Dark types (Absol), Ghost types. Surf hits both enemies.",
        completion_condition="team_level_40",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_041",
        description="Organize team for Double Battle mechanics",
        action_type="menu",
        category="battling",
        target_location="Mossdeep Pokemon Center",
        navigation_hint="Use PC to organize team. First TWO Pokemon will be sent out together. Place best dual attackers in positions 1-2.",
        completion_condition="team_organized_double_battle",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_042",
        description="Heal before Mossdeep Gym",
        action_type="interact",
        category="battling",
        target_location="Mossdeep Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. CRITICAL before challenging Tate & Liza - unique double battle.",
        completion_condition="mossdeep_heal_done",
        priority=1,
        prerequisite_story_objective="mossdeep_146"
    ),
    DirectObjective(
        id="battle_043",
        description="Heal before Sootopolis Gym",
        action_type="interact",
        category="battling",
        target_location="Sootopolis Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. Critical before challenging Juan - final gym battle.",
        completion_condition="sootopolis_heal_done",
        priority=1,
        prerequisite_story_objective="sootopolis_187"
    ),
    DirectObjective(
        id="battle_044",
        description="Train team to Lv 44-46 for Juan",
        action_type="battle",
        category="battling",
        target_location="Route 128 / Victory Road entrance area",
        navigation_hint="Juan's ace is Lv 46 Kingdra. Kingdra only weak to Dragon! Need Electric/Grass for others.",
        completion_condition="team_level_44",
        priority=1,
        prerequisite_story_objective="sootopolis_187"
    ),
    DirectObjective(
        id="battle_045",
        description="Buy massive healing stock before Victory Road",
        action_type="shop",
        category="battling",
        target_location="Ever Grande / Lilycove Department Store",
        navigation_hint="Buy: 20 Hyper Potions (24000), 10 Revives (15000), 10 Full Heals (6000). Total ~45000. Victory Road is long and difficult.",
        completion_condition="victory_road_shopping_done",
        priority=2,
        optional=True,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_046",
        description="CRITICAL: Buy full stock of healing items for Elite Four",
        action_type="shop",
        category="battling",
        target_location="Ever Grande / Lilycove Department Store",
        navigation_hint="Buy: 30 Full Restores (90000), 20 Revives (30000), 15 Full Heals (9000). NO POKEMON CENTER inside Elite Four!",
        completion_condition="elite_four_shopping_done",
        priority=1,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_047",
        description="Train team to Lv 50-55 for Elite Four + Champion",
        action_type="battle",
        category="battling",
        target_location="Victory Road",
        navigation_hint="Champion Wallace's ace is Lv 58 Milotic. Need diverse coverage: Fighting for Sidney (Dark), Ghost/Dark for Phoebe (Ghost), Fire for Glacia (Ice), Ice for Drake (Dragon), Electric/Grass for Wallace (Water).",
        completion_condition="team_level_50",
        priority=1,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_048",
        description="Check team coverage before Elite Four",
        action_type="menu",
        category="battling",
        target_location="Ever Grande Pokemon Center",
        navigation_hint="Review team moves and types. Ensure coverage for: Dark, Ghost, Ice, Dragon, Water. Check moveset for super-effective attacks.",
        completion_condition="team_checked_elite_four",
        priority=1,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_049",
        description="Ensure Ice-type Pokemon trained for Drake",
        action_type="battle",
        category="battling",
        target_location="Victory Road",
        navigation_hint="Drake's entire team is Dragon-type. Ice attacks are 4x effective on most. Train Walrein, or teach Ice Beam to Water types.",
        completion_condition="ice_type_ready",
        priority=1,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_050",
        description="Final heal before Elite Four chamber",
        action_type="interact",
        category="battling",
        target_location="Ever Grande Pokemon Center",
        navigation_hint="Talk to Nurse Joy at counter. Free full healing. ABSOLUTELY CRITICAL - no healing available inside Elite Four. Save game after healing.",
        completion_condition="elite_four_final_heal",
        priority=1,
        prerequisite_story_objective="league_194"
    ),
    DirectObjective(
        id="battle_051",
        description="Return to Sky Pillar to catch Rayquaza (Lv 70)",
        action_type="catch",
        category="battling",
        target_location="Sky Pillar Summit",
        navigation_hint="Rayquaza Lv70. SAVE BEFORE BATTLE. Dragon/Flying. Use Master Ball or Ultra Balls + status.",
        completion_condition="rayquaza_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
    DirectObjective(
        id="battle_052",
        description="Track and catch roaming Latios or Latias",
        action_type="catch",
        category="battling",
        target_location="Various Routes",
        navigation_hint="After E4, Mom asks about TV color. Blue=Latios, Red=Latias roams Hoenn. Track via Pokedex.",
        completion_condition="lati_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
    DirectObjective(
        id="battle_053",
        description="Catch Kyogre in Marine Cave (Lv 70)",
        action_type="catch",
        category="battling",
        target_location="Marine Cave",
        navigation_hint="Check weather reports. Heavy rain on route = Marine Cave nearby. Underwater entrance.",
        completion_condition="kyogre_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
    DirectObjective(
        id="battle_054",
        description="Catch Groudon in Terra Cave (Lv 70)",
        action_type="catch",
        category="battling",
        target_location="Terra Cave",
        navigation_hint="Check weather reports. Harsh sun on route = Terra Cave nearby. Land cave entrance.",
        completion_condition="groudon_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
    DirectObjective(
        id="battle_055",
        description="Solve Regi puzzle and catch Regirock, Regice, Registeel",
        action_type="catch",
        category="battling",
        target_location="Sealed Chamber / Regi Caves",
        navigation_hint="Complex puzzle starting at Sealed Chamber (Route 134). Opens Desert Ruins, Island Cave, Ancient Tomb. Each Regi Lv40.",
        completion_condition="regis_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
    DirectObjective(
        id="battle_056",
        description="Battle Steven Stone in Meteor Falls (postgame boss)",
        action_type="battle",
        category="battling",
        target_location="Meteor Falls Depths",
        navigation_hint="Deepest part of Meteor Falls (needs Waterfall). Steven's team includes Lv77 Metagross. Very tough!",
        completion_condition="steven_postgame_defeated",
        priority=2,
        optional=True,
        prerequisite_story_objective="post_210"
    ),
]


# Validation
print(f"✅ Categorized objectives loaded: {len(STORY_OBJECTIVES)} story + {len(BATTLING_OBJECTIVES)} battling = {len(STORY_OBJECTIVES) + len(BATTLING_OBJECTIVES)} total")
assert len(STORY_OBJECTIVES) + len(BATTLING_OBJECTIVES) == 269, f"Expected 269 total, got {len(STORY_OBJECTIVES) + len(BATTLING_OBJECTIVES)}"

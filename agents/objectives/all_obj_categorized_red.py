"""
Categorized Game Objectives for Pokemon Red

~100 objectives split into 2 categories:
- STORY_OBJECTIVES (78): Narrative progression aligned with Bulbapedia walkthrough (IDs 001-078, continuous)
- BATTLING_OBJECTIVES (22): Team building and training

Location names use human-readable strings matching in-game map names.
"""

from .objective_types import DirectObjective


# Story objectives (~89 total)
STORY_OBJECTIVES = [

    # ============================================================
    # PHASE 1: PALLET TOWN — Getting Started
    # ============================================================
    # DirectObjective(
    #     id="pallet_000",
    #     description="Start the game, enter your name and rival's name, and wake up in your bedroom",
    #     action_type="dialogue",
    #     category="story",
    #     target_location="Pallet Town",
    #     navigation_hint="Press A through Professor Oak's introduction. Enter your name (e.g., RED) and your rival's name (e.g., BLUE). You start in your bedroom on the second floor of your house.",
    #     completion_condition="game_started",
    #     priority=1
    # ),
    DirectObjective(
        id="pallet_001",
        description="Retrieve the Potion from the PC in your bedroom",
        action_type="interact",
        category="story",
        target_location="Pallet Town",
        navigation_hint="Walk to the PC on the top left corner of your bedroom and press A. Select WITHDRAW ITEM and take the Potion. This is your only healing item at the start.",
        completion_condition="potion_retrieved_from_pc",
        priority=1
    ),
    DirectObjective(
        id="pallet_002",
        description="Go downstairs, talk to Mom, then exit the house",
        action_type="navigate",
        category="story",
        target_location="Pallet Town",
        navigation_hint=(
            "ROUTE: (a) From bedroom RedsHouse2f, walk to stairs at (7,1) "
            "and press A to descend. (b) In RedsHouse1f, walk to Mom near "
            "(5,3), press A to talk, advance dialogue with A. (c) Then "
            "WALK SOUTH-WEST to the front door at (2,7) RedsHouse1f. The "
            "door tile (2,7) is on the bottom row, two from the left wall. "
            "Step on (2,7) to transition to Pallet Town. After Mom dialogue "
            "ends, your job is movement (DOWN/LEFT) to reach (2,7), NOT "
            "more A presses. Avoid camping at (5,3) or (7,7) — neither is "
            "the exit."
        ),
        completion_condition="talked_to_mom",
        priority=1
    ),
    DirectObjective(
        id="pallet_003",
        description="Try to walk north out of Pallet Town — Oak will stop you",
        action_type="navigate",
        category="story",
        target_location="Pallet Town",
        navigation_hint="Walk north toward the tall grass on Route 1. Oak calls out and stops you from going without a Pokémon. He leads you back to his lab.",
        completion_condition="oak_stops_player",
        priority=1
    ),
    DirectObjective(
        id="pallet_004",
        description="Follow Professor Oak to his lab and choose your starter Pokémon",
        action_type="interact",
        category="story",
        target_location="Oak's Lab",
        navigation_hint="Oak leads you inside his lab. Three Poké Balls are on the table: Bulbasaur (Grass — good vs Brock/Misty), Charmander (Fire — hardest early, great late), Squirtle (Water — easiest overall). Pick one and confirm your choice.",
        completion_condition="starter_pokemon_obtained",
        priority=1,
    ),
    DirectObjective(
        id="pallet_005",
        description="Battle rival Blue in Oak's Lab (first rival battle)",
        action_type="battle",
        category="story",
        target_location="Oak's Lab",
        navigation_hint="Blue picked the starter with type advantage over yours (Squirtle beats Charmander, Bulbasaur beats Squirtle, Charmander beats Bulbasaur). Both are Lv 5. Use Tackle and win — it's close. Reward: $175.",
        completion_condition="rival_battle_1_won",
        priority=1
    ),

    # ============================================================
    # PHASE 2: ROUTE 1, VIRIDIAN CITY, OAK'S PARCEL
    # ============================================================
    DirectObjective(
        id="viridian_006",
        description="Walk north on Route 1 to Viridian City",
        action_type="navigate",
        category="story",
        target_location="Route 1",
        navigation_hint="Walk north through Route 1. A Poké Mart employee near the entrance gives you a free Potion. Wild Pidgey and Rattata appear in tall grass. Viridian City is at the north end.",
        completion_condition="reached_viridian_city",
        priority=1, 
        recommended_battling_objectives=["battle_000"]
    ),
    DirectObjective(
        id="viridian_007",
        description="Visit Viridian City Poké Mart and receive Oak's Parcel from the shopkeeper",
        action_type="interact",
        category="story",
        target_location="Viridian City",
        navigation_hint="Enter the Poké Mart (blue-roofed building). Talk to the shopkeeper — he gives you Oak's Parcel to deliver to Professor Oak in Pallet Town. Item inventory is limited until you deliver the parcel.",
        completion_condition="oaks_parcel_received",
        priority=1
    ),
    DirectObjective(
        id="viridian_008",
        description="Return south on Route 1 to Pallet Town and deliver Oak's Parcel",
        action_type="interact",
        category="story",
        target_location="Oak's Lab",
        navigation_hint="Walk south back to Pallet Town. Enter Oak's Lab and talk to Oak. Give him the parcel. Oak gives you the Pokédex for the first time and asks you to complete it.",
        completion_condition="oaks_parcel_delivered",
        priority=1,
        recommended_battling_objectives=["battle_001"]
    ),
    DirectObjective(
        id="pallet_009",
        description="Visit Daisy (Blue's sister) in Pallet Town to receive the Town Map",
        action_type="dialogue",
        category="story",
        target_location="Pallet Town",
        navigation_hint="After receiving the Pokédex from Oak, go next door to the rival's house. Talk to Daisy (Blue's sister) — she gives you the Town Map, a Key Item showing all of Kanto.",
        completion_condition="town_map_obtained",
        priority=1
    ),
    DirectObjective(
        id="viridian_010",
        description="Return to Viridian City and heal at the Pokémon Center",
        action_type="interact",
        category="story",
        target_location="Viridian City",
        navigation_hint="Walk north on Route 1 to Viridian City. Heal at the Pokémon Center (red-roof building). This is your base before Viridian Forest.",
        completion_condition="healed_at_viridian",
        priority=1,
        recommended_battling_objectives=["battle_002"]
    ),
    DirectObjective(
        id="viridian_011",
        description="Walk west to Route 22 and battle rival Blue (second rival battle)",
        action_type="battle",
        category="story",
        target_location="Route 22",
        navigation_hint="Go west from Viridian City to Route 22. Blue ambushes you with Pidgey Lv 9 and his starter Lv 8. Use your best attacks. Reward: $280.",
        completion_condition="rival_battle_2_won",
        priority=1,
        recommended_battling_objectives=["battle_002"]
    ),
    DirectObjective(
        id="viridian_012",
        description="Walk north through Route 2 to the entrance of Viridian Forest",
        action_type="navigate",
        category="story",
        target_location="Route 2",
        navigation_hint="From Viridian City, walk north on Route 2. The Viridian Forest entrance is a small building at the north end of Route 2. Walk through it to enter the forest.",
        completion_condition="reached_viridian_forest_entrance",
        priority=1
    ),

    # ============================================================
    # PHASE 3: VIRIDIAN FOREST & PEWTER CITY
    # ============================================================
    DirectObjective(
        id="pewter_013",
        description="Navigate through Viridian Forest, battling Bug Catchers",
        action_type="navigate",
        category="story",
        target_location="Viridian Forest",
        navigation_hint="Viridian Forest has 3 Bug Catcher trainers with Caterpie, Weedle, Kakuna (Lv 6-9). Pick up items on the ground: Poké Ball, Antidote ×2. Wild Pikachu has ~5% encounter rate — worth catching for Misty. Exit through the north gate.",
        completion_condition="viridian_forest_cleared",
        priority=1,
        recommended_battling_objectives=["battle_003"]
    ),
    DirectObjective(
        id="pewter_014",
        description="Reach Pewter City and heal at the Pokémon Center",
        action_type="interact",
        category="story",
        target_location="Pewter City",
        navigation_hint="Exit the forest north gate to reach Pewter City. Heal at the Pokémon Center. Stock up at the Poké Mart if you need Potions or Antidotes before the gym.",
        completion_condition="healed_at_pewter",
        priority=1,
        recommended_battling_objectives=["battle_004"]
    ),
    DirectObjective(
        id="pewter_015",
        description="Enter Pewter Gym and defeat the Jr. Trainer",
        action_type="battle",
        category="story",
        target_location="Pewter Gym",
        navigation_hint="Pewter Gym is northwest of the city. A Jr. Trainer♂ blocks the path with Diglett Lv 11 and Sandshrew Lv 11. Defeat him to clear the way to Brock. Reward: $220.",
        completion_condition="pewter_gym_trainer_defeated",
        priority=1,
        recommended_battling_objectives=["battle_004"]
    ),
    DirectObjective(
        id="pewter_016",
        description="Battle Gym Leader Brock for the Boulder Badge",
        action_type="battle",
        category="story",
        target_location="Pewter Gym",
        navigation_hint="Brock: Geodude Lv 12 (Tackle, Defense Curl), Onix Lv 14 (Tackle, Screech, Bide). Water/Grass 4× on Geodude, 2× on Onix. Charmander is at a disadvantage — spam Growl and Ember. Reward: $1386, Boulder Badge, TM34 Bide.",
        completion_condition="boulder_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 4: ROUTE 3, MT. MOON, ROUTE 4
    # ============================================================
    DirectObjective(
        id="mtmoon_017",
        description="Walk east from Pewter City along Route 3, battling trainers",
        action_type="navigate",
        category="story",
        target_location="Route 3",
        navigation_hint="Route 3 has 8 trainers (Pidgey, Spearow, Jigglypuff, Lv 9-14). Good EXP. A Pokémon Center at the Route 4 west entrance sells Potions. Fight all trainers before Mt. Moon to level up.",
        completion_condition="route3_cleared",
        priority=1
    ),
    DirectObjective(
        id="mtmoon_018",
        description="Heal at the Route 4 Pokémon Center before entering Mt. Moon",
        action_type="interact",
        category="story",
        target_location="Route 4",
        navigation_hint="A Pokémon Center sits just west of Mt. Moon's entrance on Route 4. Heal your team. A man here also sells Magikarp for $500 — Gyarados is very powerful later.",
        completion_condition="healed_at_route4",
        priority=1
    ),
    DirectObjective(
        id="mtmoon_019",
        description="Navigate Mt. Moon to defeat Super Nerd and obtain a fossil",
        action_type="navigate",
        category="story",
        target_location="Mt. Moon B2F",
        navigation_hint="From Mt. Moon 1F, head to the top-left area to find the ladder (warp point) that leads down to B1F. In B1F, navigate to the other end of this isolated area and take the ladder (warp point) down to B2F. In B2F, navigate to the Super Nerd guarding two fossils. Defeat him, then choose: Dome Fossil → Kabuto or Helix Fossil → Omanyte (both revived on Cinnabar Island later). You may encounter trainers and Team Rocket grunts along the way — defeat them as needed. Wild encounters: Zubat, Geodude, Paras, Clefairy (rare).",
        completion_condition="fossil_obtained",
        recommended_battling_objectives=["battle_005"],
        priority=1
    ),
    DirectObjective(
        id="mtmoon_020",
        description="Exit Mt. Moon to Route 4",
        action_type="navigate",
        category="story",
        target_location="Route 4",
        navigation_hint="After obtaining the fossil, go to the top-left ladder of B2F which leads back up to B1F. Navigate to the other end of this isolated B1F area to find the exit that leads out to Route 4.",
        completion_condition="exited_mt_moon_to_route4",
        priority=1
    ),
    DirectObjective(
        id="mtmoon_021",
        description="Walk east from Route 4 to reach Cerulean City",
        action_type="navigate",
        category="story",
        target_location="Cerulean City",
        navigation_hint="From the Mt. Moon exit on Route 4, head east along Route 4 to reach Cerulean City.",
        completion_condition="reached_cerulean_city",
        priority=1,
    ),

    # ============================================================
    # PHASE 5: CERULEAN CITY
    # ============================================================
    DirectObjective(
        id="cerulean_022",
        description="Heal at Cerulean City Pokémon Center",
        action_type="interact",
        category="story",
        target_location="Cerulean City",
        navigation_hint="Pokémon Center is near the south entrance of Cerulean City. Heal before tackling the gym or Nugget Bridge.",
        completion_condition="healed_at_cerulean",
        priority=1,
        recommended_battling_objectives=["battle_006"]
    ),
    DirectObjective(
        id="cerulean_023",
        description="Enter the Cerulean Gym",
        action_type="navigate",
        category="story",
        target_location="Cerulean Gym",
        navigation_hint="The Cerulean Gym is in the north of Cerulean City (swimming/pool theme). Defeat the two swimmers inside before reaching Misty.",
        completion_condition="entered_cerulean_gym",
        priority=1
    ),
    DirectObjective(
        id="cerulean_024",
        description="Battle Gym Leader Misty for the Cascade Badge",
        action_type="battle",
        category="story",
        target_location="Cerulean Gym",
        navigation_hint="Misty: Staryu Lv 18 (Water Gun, Swift, Harden, Tackle), Starmie Lv 21 (Water Gun, BubbleBeam, Swift, Harden). Grass or Electric types are super effective. Pikachu from Viridian Forest is ideal. Oddish/Bellsprout also work. Reward: $2142, Cascade Badge, TM11 BubbleBeam.",
        completion_condition="cascade_badge_obtained",
        priority=1
    ),
    DirectObjective(
        id="cerulean_025",
        description="Battle rival Blue at the Route 24 entrance (third rival battle)",
        action_type="battle",
        category="story",
        target_location="Route 24",
        navigation_hint="Head north from Cerulean City toward Route 24. Blue ambushes you at the entrance to Nugget Bridge. He has Pidgeotto Lv 18, Abra Lv 15, Rattata Lv 15, and his evolved starter (Lv 17 — Charmeleon/Wartortle/Ivysaur). Use your strongest moves. Reward: $1,008.",
        completion_condition="rival_battle_3_won",
        priority=1
    ),
    DirectObjective(
        id="cerulean_026",
        description="Walk north on Route 24 (Nugget Bridge) and defeat 5 trainers plus Team Rocket grunt",
        action_type="battle",
        category="story",
        target_location="Route 24",
        navigation_hint="Route 24 = Nugget Bridge. Defeat 5 trainers in a row; the 6th is a Team Rocket grunt who tries to recruit you — refuse and defeat him to complete the challenge and claim the Nugget ($5000). TM45 Thunder Wave is also on the northwest hill.",
        completion_condition="nugget_bridge_cleared",
        priority=1
    ),
    DirectObjective(
        id="cerulean_027",
        description="Walk east on Route 25 to Bill's Sea Cottage and help Bill restore himself",
        action_type="interact",
        category="story",
        target_location="Bill's Sea Cottage",
        navigation_hint="Bill accidentally fused with a Pokémon. Talk to the Pokémon-Bill, then use his PC to run the gene restoration program. He thanks you and gives you the S.S. Ticket — required to board S.S. Anne in Vermilion City.",
        completion_condition="ss_ticket_obtained",
        priority=1
    ),
    DirectObjective(
        id="cerulean_028",
        description="Defeat Team Rocket grunt near the robbed house in Cerulean City",
        action_type="battle",
        category="story",
        target_location="Cerulean City",
        navigation_hint="A Team Rocket grunt is near a house northeast of the Pokémon Center. He was stealing. Battle him to recover TM28 Dig (Ground-type, good coverage move). The house has items inside.",
        completion_condition="cerulean_rocket_grunt_defeated",
        priority=1
    ),

    # ============================================================
    # PHASE 6: VERMILION CITY & S.S. ANNE
    # ============================================================
    DirectObjective(
        id="vermilion_029",
        description="Walk south via Routes 5-6 and Underground Path to reach Vermilion City",
        action_type="navigate",
        category="story",
        target_location="Vermilion City",
        navigation_hint="Route 5 has a Pokémon Day Care and an Underground Path entrance. The Underground Path (Routes 5-6) skips trainer encounters. Exit on Route 6 and continue south to Vermilion City. Heal at the Pokémon Center.",
        completion_condition="reached_vermilion_city",
        priority=1,
        recommended_battling_objectives=["battle_007", "battle_009"]
    ),
    DirectObjective(
        id="vermilion_030",
        description="Visit the Pokémon Fan Club in Vermilion City to get the Bike Voucher",
        action_type="dialogue",
        category="story",
        target_location="Vermilion City",
        navigation_hint="The Pokémon Fan Club is in the west part of Vermilion. Talk to the Chairman — let him talk about his Pokémon. He gives you a Bike Voucher. Bring it to the Bike Shop in Cerulean City for a free Bicycle later.",
        completion_condition="bike_voucher_obtained",
        priority=1
    ),
    DirectObjective(
        id="vermilion_031",
        description="Board the S.S. Anne using the S.S. Ticket and explore the ship",
        action_type="navigate",
        category="story",
        target_location="S.S. Anne 1F",
        navigation_hint="Show the S.S. Ticket at the Vermilion dock to board. Explore all floors for trainers (good EXP) and items: TM08 Body Slam (1F room), TM44 Rest (2F room). The ship has ~12 trainers total.",
        completion_condition="boarded_ss_anne",
        priority=1
    ),
    DirectObjective(
        id="vermilion_032",
        description="Battle rival Blue on S.S. Anne 2F (fourth rival battle)",
        action_type="battle",
        category="story",
        target_location="S.S. Anne 2F",
        navigation_hint="Blue is waiting on 2F. His team: Pidgeotto, Raticate, Kadabra, and his starter's 2nd evolution (~Lv 19-20). Use your best type matchups. Reward: $1386.",
        completion_condition="rival_battle_4_won",
        priority=1
    ),
    DirectObjective(
        id="vermilion_033",
        description="Get HM01 Cut from the Captain in his cabin on S.S. Anne",
        action_type="interact",
        category="story",
        target_location="S.S. Anne Captain's Room",
        navigation_hint="Go to the bow (front) of the ship and find the Captain's cabin. Talk to the sick Captain — press A to rub his back. He gives you HM01 Cut. Teach Cut to a Pokémon to chop small trees blocking paths.",
        completion_condition="hm01_cut_obtained",
        priority=1,
        recommended_battling_objectives=["battle_008"]
    ),
    DirectObjective(
        id="vermilion_034",
        description="Enter Vermilion Gym and solve the two trash can switch puzzle",
        action_type="interact",
        category="story",
        target_location="Vermilion Gym",
        navigation_hint="Use Cut on the small tree blocking the gym entrance. Inside: find two hidden switches in trash cans. The first switch is random — check each can. The second is always adjacent to the first. Flip both to unlock the door to Lt. Surge.",
        completion_condition="vermilion_gym_door_open",
        priority=1
    ),
    DirectObjective(
        id="vermilion_035",
        description="Battle Gym Leader Lt. Surge for the Thunder Badge",
        action_type="battle",
        category="story",
        target_location="Vermilion Gym",
        navigation_hint="The trash can puzzle is already solved — walk straight to Lt. Surge. His team: Voltorb Lv 21, Pikachu Lv 18, Raichu Lv 24. Use Ground-types (Diglett/Dugtrio) for full Electric immunity.",
        completion_condition="thunder_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 7: BICYCLE, ROCK TUNNEL, LAVENDER TOWN
    # ============================================================
    DirectObjective(
        id="lavender_036",
        description="Return to Cerulean City and exchange the Bike Voucher for a Bicycle",
        action_type="interact",
        category="story",
        target_location="Cerulean City",
        navigation_hint="Fly or walk back north to Cerulean City. The Bike Shop is in the south of the city. Give the Bike Voucher to the shopkeeper for a free Bicycle. The Bicycle doubles your movement speed.",
        completion_condition="bicycle_obtained",
        priority=1
    ),
    DirectObjective(
        id="lavender_037",
        description="Walk east from Cerulean City along Route 9 toward Rock Tunnel",
        action_type="navigate",
        category="story",
        target_location="Route 9",
        navigation_hint="Route 9 heads east and southeast from Cerulean City. Multiple trainers (Lv 18-23, Lv ranges). Wild Rattata, Spearow. Route 10 connects north to the Power Plant and south to the Rock Tunnel entrance. Battle all trainers for EXP.",
        completion_condition="reached_route9",
        priority=1
    ),
    DirectObjective(
        id="lavender_038",
        description="Enter Rock Tunnel from Route 10 and navigate through both floors",
        action_type="navigate",
        category="story",
        target_location="Rock Tunnel 1F",
        navigation_hint="Rock Tunnel is dark — Flash (HM05) helps but isn't required. Enter from Route 10 north. Wild: Zubat, Geodude, Machop, Onix. Multiple Hiker/Pokemaniac trainers on both floors.",
        completion_condition="entered_rock_tunnel",
        priority=1
    ),
    DirectObjective(
        id="lavender_039",
        description="Navigate through Rock Tunnel and exit south to Lavender Town",
        action_type="navigate",
        category="story",
        target_location="Lavender Town",
        navigation_hint="Rock Tunnel path: On 1F, head east from the entrance to the ladder in the NE corner. On B1F, go southwest then west then northwest to the ladder in the NW corner. Back on 1F, head south past trainers to the exit in the south. Exit leads to Route 10 south — walk south to Lavender Town and heal. Battle with trainers and wild encounters on the way for team building.",
        completion_condition="reached_lavender_town",
        priority=1
    ),

    # ============================================================
    # PHASE 8: CELADON CITY & TEAM ROCKET HIDEOUT
    # ============================================================
    DirectObjective(
        id="celadon_040",
        description="Walk west from Lavender Town through Route 8 and Underground Path to Celadon City",
        action_type="navigate",
        category="story",
        target_location="Celadon City",
        navigation_hint="Route 8 goes west from Lavender toward Saffron. The Saffron City gates are blocked — take the Route 7-8 Underground Path to bypass them and emerge on Route 7. Then walk west to Celadon City. Heal at the Pokémon Center.",
        completion_condition="reached_celadon_city",
        priority=1,
        recommended_battling_objectives=["battle_010", "battle_011"]
    ),
    DirectObjective(
        id="celadon_041",
        description="Get the Coin Case from the restaurant near the Celadon Game Corner",
        action_type="interact",
        category="story",
        target_location="Celadon City",
        navigation_hint="The restaurant/diner is south of the Game Corner. A man inside lost his money gambling. Talk to him and he gives you the Coin Case. You need it to use the slot machines. Also check the rooftop vending machines — Fresh Water → Ice Beam TM from a girl up there.",
        completion_condition="coin_case_obtained",
        priority=1
    ),
    DirectObjective(
        id="celadon_042",
        description="Enter Celadon Gym and battle Gym Leader Erika for the Rainbow Badge",
        action_type="battle",
        category="story",
        target_location="Celadon Gym",
        navigation_hint="Celadon Gym is west of the city (tree-filled interior). It is not directly reachable - HM01 Cut required to create the path. Erika: Victreebel Lv 29 (Razor Leaf, Acid), Tangela Lv 24 (Constrict, Sleep Powder), Vileplume Lv 29 (Petal Dance, Sleep Powder). Fire, Poison, Flying, Psychic, Ice all work. Watch for Sleep — use Awakening. Reward: $2958, Rainbow Badge, TM21 Mega Drain.",
        completion_condition="rainbow_badge_obtained",
        priority=1
    ),
    DirectObjective(
        id="celadon_043",
        description="Find the Team Rocket poster in the Game Corner and reveal the hidden switch",
        action_type="interact",
        category="story",
        target_location="Celadon City",
        navigation_hint="Inside the Celadon Game Corner, find the poster on the back wall. Interact with it — a Team Rocket grunt confronts you. Defeat him. The hidden switch behind the poster opens the staircase to Team Rocket Hideout B1F.",
        completion_condition="rocket_hideout_entrance_found",
        priority=1
    ),
    DirectObjective(
        id="celadon_044",
        description="Navigate Team Rocket Hideout B1F, then B2F, and descend to B3F",
        action_type="navigate",
        category="story",
        target_location="Team Rocket Hideout B1F",
        navigation_hint="B1F: Grab the Escape Rope in the western room, then take the stairs down to B2F. B2F (first visit): Battle the Rocket Grunt near the entrance. Walk directly to the NE staircase (Staircase 2) that leads down to B3F — do NOT enter the spinner maze on this first visit, the NE stairs are reachable without touching any spinner tiles.",
        completion_condition="reached_rocket_hideout_b3f",
        priority=1
    ),
    DirectObjective(
        id="celadon_045",
        description="Navigate B3F: defeat Grunt, collect TM10, solve the spinner maze, and descend to B4F",
        action_type="navigate",
        category="story",
        target_location="Team Rocket Hideout B3F",
        navigation_hint="Go south past the Rocket Grunt and pick up TM10 Self-Destruct. Then enter the spinner maze from the top-left area. Call the red_puzzle_agent tool with location_name='RocketHideoutB3f' for step-by-step guidance through the maze. General route: left spinner → right spinner to the south → go SW to the second-from-bottom right spinner → exit the spinner maze to the south → take the stairs down to B4F. Do NOT press any buttons while being pushed by a spinner — wait until fully stopped.",
        completion_condition="reached_rocket_hideout_b4f",
        priority=1
    ),
    DirectObjective(
        id="celadon_046",
        description="Pick up the Lift Key and items on B4F northwest area",
        action_type="navigate",
        category="story",
        target_location="Team Rocket Hideout B4F",
        navigation_hint="You arrive on B4F from the B3F staircase. Navigate to the NW area of B4F. Pick up HP Up and TM02 Razor Wind. Defeat the Rocket Grunt guarding the Lift Key — he drops it. The Lift Key lets you use the elevator on B2F.",
        completion_condition="lift_key_obtained",
        priority=1
    ),
    DirectObjective(
        id="celadon_047",
        description="Climb back to B2F, navigate the spinner maze to the SE Lift, ride it to B4F SE",
        action_type="navigate",
        category="story",
        target_location="Team Rocket Hideout B2F",
        navigation_hint="Climb the stairs back up from B4F through B3F to B2F. On B2F, you now need to navigate the spinner maze to reach the Lift (elevator) in the SE area. Call the red_puzzle_agent tool with location_name='RocketHideoutB2f' for step-by-step guidance. General route: go west past the Rocket Grunt → second-from-the-top left spinner → Moon Stone area → first right spinner down from Moon Stone → go east and south through the SE portion of the maze → reach the Lift. Use the Lift Key to ride the elevator down to B4F SE.",
        completion_condition="reached_rocket_hideout_b4f_se",
        priority=1
    ),
    DirectObjective(
        id="celadon_048",
        description="Defeat the last 2 Rocket Grunts on B4F SE to open the door to Giovanni",
        action_type="battle",
        category="story",
        target_location="Team Rocket Hideout B4F",
        navigation_hint="You arrive at B4F SE via the Lift. Grab the Iron item nearby. Defeat the last 2 Rocket Grunts — this opens the locked door leading to Giovanni's room.",
        completion_condition="Giovanni_door_opened",
        priority=1
    ),
    DirectObjective(
        id="celadon_049",
        description="Ride the elevator to Giovanni and defeat the Team Rocket boss to get the Silph Scope",
        action_type="battle",
        category="story",
        target_location="Team Rocket Hideout B4F",
        navigation_hint="Giovanni (Rocket Boss): Onix Lv 25, Rhyhorn Lv 24, Kangaskhan Lv 29. Water/Grass/Ice for Rock/Ground. After winning, Giovanni gives you the Silph Scope. The Silph Scope lets you identify Ghost Pokémon in Pokémon Tower.",
        completion_condition="Giovanni_defeated",
        priority=1
    ),

    # ============================================================
    # PHASE 9: POKÉMON TOWER (LAVENDER TOWN)
    # ============================================================
    DirectObjective(
        id="pokemontower_050",
        description="Return to Lavender Town and enter Pokémon Tower",
        action_type="navigate",
        category="story",
        target_location="Pokémon Tower 1F",
        navigation_hint="Walk east back through Routes 7-8 underground path to Lavender Town. Pokémon Tower is the tall building in the center of the city. Enter from 1F. Floors 3+ require the Silph Scope to identify Ghost types.",
        completion_condition="entered_pokemon_tower",
        priority=1
    ),
    DirectObjective(
        id="pokemontower_051",
        description="Battle rival Blue on Pokémon Tower 2F (fifth rival battle)",
        action_type="battle",
        category="story",
        target_location="Pokémon Tower 2F",
        navigation_hint="Blue is on 2F. His team now has Pidgeotto, Gyarados, Growlithe, Kadabra, and his starter's evolution (~Lv 25). Use Ice for Gyarados. Reward: $1750.",
        completion_condition="rival_battle_5_won",
        priority=1
    ),
    DirectObjective(
        id="pokemontower_052",
        description="Climb Pokémon Tower floors 3-6, battling Channelers",
        action_type="battle",
        category="story",
        target_location="Pokémon Tower 3F",
        navigation_hint="With the Silph Scope, Ghost Pokémon (Gastly, Haunter) are now identifiable. Channeler trainers use Ghost types. IMPORTANT: Normal and Fighting moves have NO EFFECT on Gastly/Haunter. Use Psychic, Water, Fire, or Ground moves. Awakenings help against Sleep Powder.",
        completion_condition="reached_pokemon_tower_7f",
        priority=1
    ),
    DirectObjective(
        id="pokemontower_053",
        description="Reach Pokémon Tower 7F, free Mr. Fuji from Team Rocket",
        action_type="battle",
        category="story",
        target_location="Pokémon Tower 7F",
        navigation_hint="7F is the top floor, occupied by Team Rocket. Defeat all grunts and free Mr. Fuji. He was being held captive. He thanks you and takes you back to his house in Lavender Town.",
        completion_condition="mr_fuji_rescued",
        priority=1
    ),
    DirectObjective(
        id="pokemontower_054",
        description="Receive the Poké Flute from Mr. Fuji in his house",
        action_type="interact",
        category="story",
        target_location="Mr. Fuji's House",
        navigation_hint="Mr. Fuji's house is in the south of Lavender Town. He gives you the Poké Flute — a Key Item that wakes sleeping Pokémon. Two Snorlax (Lv 30) block Routes 12 and 16. Use the Flute to wake them and clear the path.",
        completion_condition="poke_flute_obtained",
        priority=1,
        recommended_battling_objectives=["battle_012", "battle_013"]
    ),

    # ============================================================
    # PHASE 10: SAFFRON CITY & SILPH CO.
    # ============================================================
    DirectObjective(
        id="saffron_055",
        description="Enter Saffron City after giving a drink to the gate guards",
        action_type="navigate",
        category="story",
        target_location="Saffron City",
        navigation_hint="From Lavender Town, go west through Route 8 to Celadon City. Go to Celadon Dept. Store — take the elevator or stairs to 5F, then walk to the staircase on 5F to reach the rooftop. Buy a Fresh Water from the vending machine on the rooftop. Then go to any Saffron City gatehouse (the Route 8 gate east of Celadon is closest) and give the drink to the thirsty guard — he shares it with all guards, opening all four gates.",
        completion_condition="reached_saffron_city",
        priority=1,
        recommended_battling_objectives=["battle_014"]
    ),
    DirectObjective(
        id="saffron_056",
        description="Enter Silph Co. and find the Card Key on 5F",
        action_type="navigate",
        category="story",
        target_location="Silph Co. 1F",
        navigation_hint="Silph Co. has 11 floors, warp tiles, and locked electronic doors. Team Rocket has taken over. Get the Card Key from 5F — it opens all locked doors in the building. Warp from 2F → 8F shortcut exists. Talk to NPCs for items including Protein (5F).",
        completion_condition="silph_co_card_key_obtained",
        priority=1
    ),
    DirectObjective(
        id="saffron_057",
        description="Battle rival Blue on Silph Co. 7F (sixth rival battle)",
        action_type="battle",
        category="story",
        target_location="Silph Co. 7F",
        navigation_hint="Use the Card Key to open locked doors by walking into them and press A. Shortest route to Blue: take the elevator or stairs down to 3F. Use the Card Key to unlock the door blocking the warp tile in the central room on 3F. Step on that warp tile — it teleports you to the northwest room on 7F. Walk left and Blue will challenge you. Team: Pidgeot Lv 37, Alakazam Lv 35, Growlithe/Gyarados/Exeggcute Lv 38, starter evolution (~Lv 40). The NPC next to Blue gives you a free Lapras after the battle.",
        completion_condition="rival_battle_6_won",
        priority=1
    ),
    DirectObjective(
        id="saffron_058",
        description="Battle Giovanni on Silph Co. 11F and receive the Master Ball from the president",
        action_type="battle",
        category="story",
        target_location="Silph Co. 11F",
        navigation_hint="After beating Blue on 7F, step on the warp tile right past him — it teleports you directly to 11F. If you need to heal first, use the bed on 9F (take elevator to 9F, talk to the woman near the bed). On 11F, defeat the last Rocket Grunt then enter the boardroom to face Giovanni. Giovanni: Nidorino Lv 37, Kangaskhan Lv 35, Rhyhorn Lv 37, Nidoqueen Lv 41. Water/Grass/Ice effective. After winning, the president gives you the Master Ball — save it for a legendary.",
        completion_condition="master_ball_obtained",
        priority=1
    ),
    DirectObjective(
        id="saffron_059",
        description="Challenge the Fighting Dojo west of Silph Co. and win a Pokémon",
        action_type="battle",
        category="story",
        target_location="Saffron City",
        navigation_hint="The Fighting Dojo is just west of Silph Co. Battle the Karate Master (Hitmonlee Lv 37, Hitmonchan Lv 37). After winning, choose Hitmonlee (strong physical kicks) or Hitmonchan (elemental punches: Fire/Ice/Thunder). Both are rare Fighting-types.",
        completion_condition="fighting_dojo_cleared",
        priority=1
    ),
    DirectObjective(
        id="saffron_060",
        description="Battle Gym Leader Sabrina in Saffron Gym for the Marsh Badge",
        action_type="battle",
        category="story",
        target_location="Saffron Gym",
        navigation_hint="Saffron Gym is a warp tile maze — navigate to reach Sabrina at the back. Sabrina: Mr. Mime Lv 28, Kadabra Lv 37, Alakazam Lv 43 (all Psychic). Normal-type moves are your best bet (Ghost is immune to Psychic, Normal is immune to Ghost — only Normal/Ghost is mutual, so use Normal). Reward: Marsh Badge, TM46 Psywave.",
        completion_condition="marsh_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 11: FUCHSIA CITY & SAFARI ZONE
    # ============================================================
    DirectObjective(
        id="fuchsia_061",
        description="Wake the Snorlax on Route 16 using the Poké Flute and continue west",
        action_type="interact",
        category="story",
        target_location="Route 16",
        navigation_hint="Go west from Celadon City. A Snorlax (Lv 30) blocks Route 16 — use the Poké Flute from Key Items to wake it. It attacks! You can catch it (Normal-type, very bulky) or just defeat it. Clear the path to continue west on Route 16.",
        completion_condition="snorlax_route16_cleared",
        priority=1
    ),
    DirectObjective(
        id="fuchsia_062",
        description="Get HM02 Fly from the girl north of the Route 16 gate",
        action_type="interact",
        category="story",
        target_location="Route 16",
        navigation_hint="Cut the small tree just north of the Route 16 gate. Enter the house. A girl inside gives you HM02 Fly. Teach Fly to a Flying-type Pokémon (Pidgeot, Fearow, Dodrio) — lets you fast-travel to any previously visited city.",
        completion_condition="hm02_fly_obtained",
        priority=1
    ),
    DirectObjective(
        id="fuchsia_063",
        description="Ride the Cycling Road (Routes 16-18) south to reach Fuchsia City",
        action_type="navigate",
        category="story",
        target_location="Fuchsia City",
        navigation_hint="Route 17 is the Cycling Road — use your Bicycle to ride downhill fast. Multiple Biker trainers with Fighting/Poison types. Route 18 leads east to Fuchsia City. Enter Fuchsia from the west. Heal at the Pokémon Center.",
        completion_condition="reached_fuchsia_city",
        priority=1,
        recommended_battling_objectives=["battle_015", "battle_016"]
    ),
    DirectObjective(
        id="fuchsia_064",
        description="Enter the Safari Zone and collect the Gold Teeth and HM03 Surf",
        action_type="navigate",
        category="story",
        target_location="Safari Zone",
        navigation_hint="Safari Zone entrance is in the north of Fuchsia. Pay $500. Navigate to Area 3 (the deepest zone). Find the Gold Teeth in an item ball here. Then, enter the Secret House right next to it and talk to the man inside to receive HM03 Surf.",
        completion_condition="hm03_surf_and_gold_teeth_obtained",
        priority=1
    ),
    DirectObjective(
        id="fuchsia_065",
        description="Give the Gold Teeth to the Safari Zone Warden to get HM04 Strength",
        action_type="interact",
        category="story",
        target_location="Fuchsia City",
        navigation_hint="The Safari Zone Warden lives in a house in the southeast of Fuchsia City, directly east of the Pokémon Center. Give him the Gold Teeth. He rewards you with HM04 Strength. Teach Strength to a strong Pokémon — required for pushing boulders in Victory Road.",
        completion_condition="hm04_strength_obtained",
        priority=1
    ),
    DirectObjective(
        id="fuchsia_066",
        description="Battle Gym Leader Koga in the Fuchsia Gym for the Soul Badge",
        action_type="battle",
        category="story",
        target_location="Fuchsia Gym",
        navigation_hint="Fuchsia Gym has invisible glass wall maze — navigate carefully to Koga. Koga: Koffing Lv 37 (Smog, Toxic, SelfDestruct), Muk Lv 39 (Sludge, Minimize), Koffing Lv 37, Weezing Lv 43. All Poison-type. Ground/Psychic super effective. Watch for Toxic status — use Antidotes freely. Reward: Soul Badge (enables Surf outside battle), TM06 Toxic.",
        completion_condition="soul_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 12: CINNABAR ISLAND
    # ============================================================
    DirectObjective(
        id="cinnabar_067",
        description="Surf south from Pallet Town or Fuchsia City to reach Cinnabar Island",
        action_type="navigate",
        category="story",
        target_location="Cinnabar Island",
        navigation_hint="Fly to Pallet Town and Surf south on Route 21. Stay centered in the water channel. When you reach Cinnabar Island, land on the northeast shore (the walkable tiles at the top-right of the island). Or Surf south from Fuchsia City via Routes 19-20 — this route passes through Seafoam Islands, a multi-floor cave requiring Strength. Push boulders on 1F and B1F down the holes, then on B3F push boulders into the two pits to block the river current. Once the current is blocked, Surf through B4F and exit west to Route 20 West, then continue west to Cinnabar. Heal at the Pokémon Center on arrival.",
        completion_condition="reached_cinnabar_island",
        priority=1,
        recommended_battling_objectives=["battle_017", "battle_018"]
    ),
    DirectObjective(
        id="cinnabar_068",
        description="Explore the Pokémon Mansion and find the Secret Key for Cinnabar Gym",
        action_type="navigate",
        category="story",
        target_location="Pokémon Mansion 1F",
        navigation_hint="Pokémon Mansion is north of the Pokémon Center. 4 floors (1F, 2F, 3F, B1F). Find the Secret Key in B1F. Read journals for lore about Mew research. Items: Moon Stone (1F hidden), Calcium (2F), Iron/Max Potion (3F), Blizzard/Full Restore/Rare Candy ×2/TM22 SolarBeam (B1F).",
        completion_condition="secret_key_obtained",
        priority=1
    ),
    DirectObjective(
        id="cinnabar_069",
        description="Visit the Cinnabar Lab to revive your fossil from Mt. Moon",
        action_type="interact",
        category="story",
        target_location="Cinnabar Island",
        navigation_hint="The Pokémon Lab is next to the Pokémon Center. Give your fossil to the scientist in the Research/Testing Room. He revives it: Dome Fossil → Kabuto Lv 30, Helix Fossil → Omanyte Lv 30. Both evolve at Lv 40 into powerful Water/Rock types.",
        completion_condition="fossil_revived",
        priority=1
    ),
    DirectObjective(
        id="cinnabar_070",
        description="Use the Secret Key to enter Cinnabar Gym and solve the quiz to reach Blaine",
        action_type="interact",
        category="story",
        target_location="Cinnabar Gym",
        navigation_hint="Use the Secret Key on the Cinnabar Gym door. Inside: quiz stations with Fire-type trivia. Correct answers open doors; wrong answers trigger trainer battles. Navigate past trainers (Lv 29-42) to reach Blaine at the back.",
        completion_condition="cinnabar_gym_entered",
        priority=1
    ),
    DirectObjective(
        id="cinnabar_071",
        description="Battle Gym Leader Blaine for the Volcano Badge",
        action_type="battle",
        category="story",
        target_location="Cinnabar Gym",
        navigation_hint="Blaine's Red team: Growlithe Lv 42, Ponyta Lv 40, Rapidash Lv 42, Arcanine Lv 47. Water is super effective against all. Rock also resists Fire. Keep Pokémon healthy — Fire Spin can trap targets. Reward: Volcano Badge, TM38 Fire Blast.",
        completion_condition="volcano_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 13: VIRIDIAN GYM
    # ============================================================
    DirectObjective(
        id="viridian_gym_072",
        description="Fly to Viridian City and enter the now-open Viridian Gym",
        action_type="navigate",
        category="story",
        target_location="Viridian Gym",
        navigation_hint="Fly to Viridian City. The Viridian Gym was previously locked — Giovanni is now here. Enter from the south. Multiple trainers with Ground/Rock types inside. Navigate spinner tiles to reach trainers, then Giovanni.",
        completion_condition="entered_viridian_gym",
        priority=1,
        recommended_battling_objectives=["battle_019", "battle_020"]
    ),
    DirectObjective(
        id="viridian_gym_073",
        description="Battle Gym Leader Giovanni for the Earth Badge (eighth and final badge)",
        action_type="battle",
        category="story",
        target_location="Viridian Gym",
        navigation_hint="Giovanni: Rhyhorn Lv 45 (Horn Drill, Fissure), Dugtrio Lv 42 (Dig, Slash), Nidoqueen Lv 44 (Poison Sting, Tail Whip), Nidoking Lv 45 (Thrash, Poison Sting), Rhydon Lv 50 (Horn Drill, Fissure). Water/Grass/Ice all super effective. Flying Pokémon immune to Ground. Rhydon is bulky — use high-power moves. Reward: Earth Badge (all Pokémon obey), TM27 Fissure.",
        completion_condition="earth_badge_obtained",
        priority=1
    ),

    # ============================================================
    # PHASE 14: VICTORY ROAD
    # ============================================================
    DirectObjective(
        id="victory_074",
        description="Walk west to Route 22 and battle rival Blue (seventh rival battle)",
        action_type="battle",
        category="story",
        target_location="Route 22",
        navigation_hint="Blue confronts you on Route 22 before Victory Road. His team: Pidgeot Lv 47, Alakazam Lv 50, Rhyhorn Lv 45, plus 3 others depending on your starter (Exeggutor/Gyarados/Arcanine variants). Be at Lv 45+ across your team. Reward: $3150.",
        completion_condition="rival_battle_7_won",
        priority=1
    ),
    DirectObjective(
        id="victory_075",
        description="Walk north on Route 23 and show all 8 badges to the guards",
        action_type="navigate",
        category="story",
        target_location="Route 23",
        navigation_hint="Route 23 has 8 badge-check guards — one for each badge. Show them in order: Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth. Wild Pokémon: Ditto (rare), Fearow, Nidorino/Nidorina. Walk north after each check toward Victory Road.",
        completion_condition="route23_cleared",
        priority=1
    ),
    DirectObjective(
        id="victory_076",
        description="Navigate Victory Road floors 1-3 using Strength to solve boulder puzzles",
        action_type="navigate",
        category="story",
        target_location="Victory Road 1F",
        navigation_hint="Victory Road is a 3-floor cave. Use Strength (from Safari Zone Warden) to push boulders onto pressure switches that open gates. Items: Rare Candy (1F), TM43 Sky Attack (1F), TM05 Mega Kick (2F), TM17 Submission (2F), Max Revive (3F), TM47 Explosion (3F). Exit 3F north to reach Indigo Plateau.",
        completion_condition="victory_road_cleared",
        priority=1
    ),

    # ============================================================
    # PHASE 15: INDIGO PLATEAU & ELITE FOUR
    # ============================================================
    DirectObjective(
        id="elite_077",
        description="Arrive at Indigo Plateau and stock up at the Poké Mart",
        action_type="interact",
        category="story",
        target_location="Indigo Plateau",
        navigation_hint="The Indigo Plateau Poké Mart sells top-tier items. Buy: 20 Full Restores ($60,000), 15 Revives ($22,500), 10 Max Potions, Full Heals, Max Repels. There is NO healing between Elite Four rooms — stock heavily.",
        completion_condition="indigo_plateau_stocked",
        priority=1,
        recommended_battling_objectives=["battle_021", "battle_022"]
    ),
    DirectObjective(
        id="elite_078",
        description="Battle Elite Four Lorelei (Ice/Water specialist)",
        action_type="battle",
        category="story",
        target_location="Lorelei's Room",
        navigation_hint="Lorelei: Dewgong Lv 54 (Surf, Ice Beam, Rest), Cloyster Lv 53 (Surf, Spike Cannon, Clamp), Slowbro Lv 54 (Amnesia, Surf, Psychic — do NOT let it use Amnesia), Jynx Lv 56 (Ice Punch, Lovely Kiss, Psychic), Lapras Lv 56 (Blizzard, Psychic, Thunderbolt). Electric destroys Water/Ice. Fire/Rock/Fighting also useful.",
        completion_condition="lorelei_defeated",
        priority=1
    ),
    DirectObjective(
        id="elite_079",
        description="Battle Elite Four Bruno (Fighting/Rock specialist)",
        action_type="battle",
        category="story",
        target_location="Bruno's Room",
        navigation_hint="Bruno: Onix Lv 53 (Rock Throw, Slam, Bide), Hitmonchan Lv 55 (Fire/Ice/ThunderPunch), Hitmonlee Lv 55 (Hi Jump Kick, Mega Kick), Onix Lv 56, Machamp Lv 58 (Submission, Leer, Karate Chop, Fissure). Psychic and Flying tear through Fighting. Water/Grass for Onix.",
        completion_condition="bruno_defeated",
        priority=1
    ),
    DirectObjective(
        id="elite_080",
        description="Battle Elite Four Agatha (Ghost/Poison specialist)",
        action_type="battle",
        category="story",
        target_location="Agatha's Room",
        navigation_hint="Agatha: Gengar Lv 56 (Hypnosis, Dream Eater, Confuse Ray, Night Shade), Golbat Lv 56 (Wing Attack, Confuse Ray), Haunter Lv 55 (Hypnosis, Dream Eater), Arbok Lv 58 (Glare, Wrap, Screech), Gengar Lv 60. Normal/Fighting useless vs Ghost. Carry many Awakenings — Hypnosis hits frequently. Ground/Psychic for Arbok.",
        completion_condition="agatha_defeated",
        priority=1
    ),
    DirectObjective(
        id="elite_081",
        description="Battle Elite Four Lance (Dragon/Flying specialist)",
        action_type="battle",
        category="story",
        target_location="Lance's Room",
        navigation_hint="Lance: Gyarados Lv 58 (Hydro Pump, Dragon Rage), Dragonair Lv 56 (Slam, Agility, Leer) ×2, Aerodactyl Lv 60 (Hyper Beam, Supersonic, Wing Attack), Dragonite Lv 62 (Agility, Slam, Dragon Rage, Hyper Beam). Ice Beam/Blizzard is super effective on ALL his Dragon/Flying types. Electric for Gyarados. This is the hardest Elite Four member.",
        completion_condition="lance_defeated",
        priority=1
    ),
    DirectObjective(
        id="elite_082",
        description="Battle Champion Blue to become the Pokémon League Champion",
        action_type="battle",
        category="story",
        target_location="Champion's Room",
        navigation_hint="Blue (Champion): Pidgeot Lv 61, Alakazam Lv 59, Rhydon Lv 61, + 3 based on your starter choice — Bulbasaur: Exeggutor/Gyarados/Charizard; Charmander: Exeggutor/Blastoise/Arcanine; Squirtle: Gyarados/Arcanine/Venusaur. Use Full Restores freely — this is the final battle. Electric vs Pidgeot/Gyarados; Water vs Rhydon; Ice vs Exeggutor; varied vs the starter. Congratulations — you are CHAMPION!",
        completion_condition="champion_defeated",
        priority=1
    ),
]


# ================================================================
# BATTLING OBJECTIVES (~22 total)
# ================================================================
BATTLING_OBJECTIVES = [

    # Early Game
    DirectObjective(
        id="battle_000",
        description="Train your starter Pokémon to Lv 10+ by battling wild Pokémon on Route 1",
        action_type="battle",
        category="battling",
        target_location="Route 1",
        navigation_hint="Battle wild Pokémons on Route 1. Reach Lv 10 minimum before going back to Pallet Town. More levels = better stats and possibly a new move.",
        completion_condition="starter_level_10",
        priority=1,
        prerequisite_story_objective="pallet_004"
    ),
    DirectObjective(
        id="battle_001",
        description="Buy Poké Balls and Antidotes at Viridian City Poké Mart",
        action_type="shop",
        category="battling",
        target_location="Viridian City",
        navigation_hint="Buy: 3 Poké Balls ($600), 6 Antidotes ($600). Total: $1200. Essential for Viridian Forest. The Poké Mart is unlocked after delivering Oak's Parcel.",
        completion_condition="viridian_shopping_done",
        priority=1,
        prerequisite_story_objective="viridian_008"
    ),
    DirectObjective(
        id="battle_002",
        description="Catch one wild Pokémon and train it to Lv 10+ before battling Blue on Route 22",
        action_type="battle",
        category="battling",
        target_location="Route 1 / Route 22",
        navigation_hint=(
            "Grind on Route 1 or in the wild grass on Route 22 (west of Viridian City) until "
            "ALL party members reach Lv 10+. Catch a Rattata, Pidgey, or Nidoran♂/♀ as a "
            "backup team member — the extra Pokémon helps absorb hits from Blue's Pidgey Lv 9 "
            "and his Lv 8 starter. When training, send your stronger Pokémon to the back and "
            "open each wild battle with your weakest team member so they gain full EXP. "
            "Do this before heading west to Route 22 for the rival fight."
        ),
        completion_condition="second_pokemon_caught_lv10",
        priority=1,
        prerequisite_story_objective="viridian_010"
    ),

    # Pre-Brock
    DirectObjective(
        id="battle_003",
        description="Catch a Pikachu in Viridian Forest for type advantage against Misty",
        action_type="catch",
        category="battling",
        target_location="Viridian Forest",
        navigation_hint="Pikachu has ~5% encounter rate in Viridian Forest (Lv 3-5). Electric is super effective against Misty's Water types. Weaken to red HP then throw a Poké Ball (available in items).",
        completion_condition="pikachu_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="viridian_012"
    ),
    DirectObjective(
        id="battle_004",
        description="Train the entire team — Pikachu and other non-starters — to Lv 14+ using wild encounters",
        action_type="battle",
        category="battling",
        target_location="Viridian Forest / Route 2",
        navigation_hint=(
            "Fight wild encounters in Viridian Forest or Route 2 grass. The level requirement "
            "is for ALL party members, not just the starter. Send your starter to the back and "
            "open each fight with the weakest Pokémon (e.g. Pikachu or newly caught mons) so "
            "they gain full EXP. Target Lv 12+ for every team member. Pikachu learns Quick "
            "Attack at Lv 13, which is crucial for Misty. Building bench depth now means you "
            "have real backup for Brock and beyond."
        ),
        completion_condition="non_starter_pokemon_level_14",
        priority=1,
        prerequisite_story_objective="viridian_012"
    ),
    # Pre-Misty
    DirectObjective(
        id="battle_005",
        description="Train your entire team to Lv 21+ and prepare for Misty's Starmie",
        action_type="battle",
        category="battling",
        target_location="Mt. Moon 1F / Mt. Moon B2F",
        navigation_hint=(
            "Misty's Starmie is Lv 21 with BubbleBeam. All party members should be Lv 21+ — "
            "not just your lead. Train ALL your Pokémons with wild encounters and trainer battles -"
            "you should lead with your weakest Pokémon first so they gain full EXP; switch to a stronger mon to finish the fight if needed. "
        ),
        completion_condition="team_level_21",
        priority=1,
        prerequisite_story_objective="mtmoon_018"
    ),
    DirectObjective(
        id="battle_006",
        description="Buy Potions and Repels at Cerulean Poké Mart",
        action_type="shop",
        category="battling",
        target_location="Cerulean City",
        navigation_hint="Buy: 10 Potions ($3000), 5 Repels ($1750). Stock up on basic healing before Nugget Bridge.",
        completion_condition="cerulean_shopping_done",
        priority=1,
        prerequisite_story_objective="cerulean_022"
    ),

    # Pre-Surge
    DirectObjective(
        id="battle_007",
        description="Catch a Diglett or Dugtrio in Diglett's Cave — Ground immunity to Electric completely counters Surge",
        action_type="catch",
        category="battling",
        target_location="Diglett's Cave",
        navigation_hint="Diglett's Cave is on Route 11 east of Vermilion City. Diglett (95%, Lv 15-22) or rare Dugtrio (5%, Lv 29-31). Ground-type is IMMUNE to all Electric moves — this destroys Lt. Surge's entire team.",
        completion_condition="ground_type_for_surge",
        priority=1,
        prerequisite_story_objective="vermilion_029"
    ),
    DirectObjective(
        id="battle_008",
        description="Catch a Pokémon that can learn HM01 Cut (Oddish on Route 6) and teach it Cut",
        action_type="catch",
        category="battling",
        target_location="Route 6",
        navigation_hint=(
            "You need a Pokémon that knows Cut to chop the tree blocking Vermilion Gym. "
            "If your starter (Bulbasaur/Charmander) or Paras from Mt. Moon can learn Cut, "
            "just teach them HM01 directly from the bag. Otherwise, catch an Oddish on Route 6 "
            "(south of Vermilion, 25% encounter rate, Lv 13-16) — Oddish learns Cut via HM01. "
            "Open the bag, select HM01 Cut, choose 'USE', and pick the Pokémon to teach it to."
        ),
        completion_condition="hm01_cut_taught",
        priority=1,
        prerequisite_story_objective="vermilion_033"
    ),
    DirectObjective(
        id="battle_009",
        description="Train your entire team to Lv 22+ before battling Lt. Surge",
        action_type="battle",
        category="battling",
        target_location="Route 11",
        navigation_hint="Lt. Surge's Raichu is Lv 24. Fight Route 11 trainers (10 trainers, Lv 18-21) and grind in Diglett's Cave (Diglett Lv 15-22). Lead with your weakest Pokémon for full EXP.",
        completion_condition="team_level_22_surge",
        priority=1,
        prerequisite_story_objective="vermilion_029"
    ),

    # Pre-Erika
    DirectObjective(
        id="battle_010",
        description="Train your entire team to Lv 28+ before battling Erika",
        action_type="battle",
        category="battling",
        target_location="Route 9",
        navigation_hint=(
            "Erika's Vileplume is Lv 29. All party members should reach Lv 28+ — not just your "
            "lead. Train on Route 9 or through Rock Tunnel. In wild encounters, open each fight "
            "with your weakest Pokémon so they get full EXP, then switch to a stronger mon if "
            "you need to finish the fight. Fire, Poison, Flying, Psychic, or Ice are super "
            "effective vs Grass. Don't use Water or Ground."
        ),
        completion_condition="team_level_28_erika",
        priority=1,
        prerequisite_story_objective="celadon_040"
    ),
    DirectObjective(
        id="battle_011",
        description="Buy key items at Celadon Department Store (TMs, healing, drinks)",
        action_type="shop",
        category="battling",
        target_location="Celadon City",
        navigation_hint="Celadon Dept Store: elevator covers 1F-5F only — take stairs from 5F to reach the Rooftop. Buy Super Potions/Great Balls (2F), Revives/X items (4F). On the Rooftop, buy Fresh Water/Soda Pop/Lemonade from the vending machine and give them to the girl for TM13 Ice Beam, TM48 Rock Slide, TM49 Tri Attack. Keep one extra drink for Saffron gate guards.",
        completion_condition="celadon_shopping_done",
        priority=1,
        prerequisite_story_objective="celadon_040"
    ),

    # Silph Co. area
    DirectObjective(
        id="battle_012",
        description="Get Lapras (Lv 15) from the Silph Co. employee on 7F",
        action_type="interact",
        category="battling",
        target_location="Silph Co. 7F",
        navigation_hint="An employee on Silph Co. 7F gives you a free Lapras (Lv 15). Lapras learns Ice Beam and Blizzard — devastating for Lance's Dragon types later. Also an excellent Surfer. Don't miss this!",
        completion_condition="lapras_obtained",
        priority=1,
        prerequisite_story_objective="saffron_056"
    ),
    DirectObjective(
        id="battle_013",
        description="Train your entire team to Lv 40+ before tackling Silph Co. and Saffron Gym",
        action_type="battle",
        category="battling",
        target_location="Saffron City",
        navigation_hint=(
            "Sabrina's Alakazam is Lv 43. Giovanni in Silph Co. has Nidoqueen Lv 41. Every "
            "party member should be Lv 40+ before entering Silph Co. Train on Routes 7-8 or in "
            "the Safari Zone area. In wild encounters, lead with your lowest-level Pokémon first "
            "and switch to a stronger mon to finish — this ensures all team members level up "
            "evenly instead of only your lead getting EXP."
        ),
        completion_condition="team_level_40_saffron",
        priority=1,
        prerequisite_story_objective="pokemontower_054"
    ),

    # Pre-Sabrina
    DirectObjective(
        id="battle_014",
        description="Get TM29 Psychic from Mr. Psychic's house in Saffron City",
        action_type="interact",
        category="battling",
        target_location="Saffron City",
        navigation_hint="Mr. Psychic lives in a house in east Saffron City. He gives you TM29 Psychic for free. Psychic is the strongest Psychic-type move in Gen 1 — teach it to Kadabra, Starmie, Jynx, or similar. Essential for late-game.",
        completion_condition="tm29_psychic_obtained",
        priority=1,
        prerequisite_story_objective="saffron_055"
    ),

    # Pre-Koga
    DirectObjective(
        id="battle_015",
        description="Train your entire team to Lv 40+ and prepare Antidotes/Awakenings before battling Koga",
        action_type="battle",
        category="battling",
        target_location="Route 15",
        navigation_hint=(
            "Koga's Weezing is Lv 43. Every party member should be Lv 40+ — not just your lead. "
            "Train on Routes 13-15 (diverse wild Pokémon). In wild encounters, open each battle "
            "with your weakest Pokémon first so they gain full EXP; switch to a stronger mon "
            "to finish. Psychic and Ground are super effective vs Poison. Stock Antidotes and "
            "Awakenings — Koga uses Toxic and Sleep Powder constantly."
        ),
        completion_condition="team_level_40_koga",
        priority=1,
        prerequisite_story_objective="fuchsia_063"
    ),
    DirectObjective(
        id="battle_016",
        description="Catch a Pokémon in the Safari Zone (Chansey, Scyther, or Exeggcute)",
        action_type="catch",
        category="battling",
        target_location="Safari Zone",
        navigation_hint="Safari Zone rare catches: Chansey (1-4%, very bulky support), Scyther (version-exclusive, strong), Exeggcute (Grass/Psychic, evolves to Exeggutor). Use Bait to reduce flee rate, then throw balls. Limited to 30 balls and 500 steps.",
        completion_condition="safari_zone_catch",
        priority=2,
        optional=True,
        prerequisite_story_objective="fuchsia_064"
    ),

    # Pre-Blaine
    DirectObjective(
        id="battle_017",
        description="Train your entire team to Lv 42+ before battling Blaine",
        action_type="battle",
        category="battling",
        target_location="Route 21",
        navigation_hint=(
            "Blaine's Arcanine is Lv 47. Target Lv 42+ for ALL party members. Train on Route 21 "
            "(ocean trainers: Fishermen, Swimmers) or revisit Routes 13-15. In wild encounters, "
            "lead with your lowest-level Pokémon first and switch to a stronger mon to finish — "
            "this keeps the whole team leveling evenly. Water-types are best against Blaine's "
            "Fire squad."
        ),
        completion_condition="team_level_42_blaine",
        priority=1,
        prerequisite_story_objective="cinnabar_067"
    ),
    DirectObjective(
        id="battle_018",
        description="Buy Hyper Potions and Revives before the Cinnabar Gym",
        action_type="shop",
        category="battling",
        target_location="Cinnabar Island",
        navigation_hint="Cinnabar Island Poké Mart stocks Hyper Potions ($1500) and Revives ($1500). Buy 10 Hyper Potions and 5 Revives before the Pokémon Mansion and Cinnabar Gym. Full Restores are not available until the Elite Four.",
        completion_condition="cinnabar_supplies_bought",
        priority=1,
        prerequisite_story_objective="cinnabar_067"
    ),

    # Pre-Giovanni (Viridian Gym)
    DirectObjective(
        id="battle_019",
        description="Train your entire team to Lv 45+ before battling Giovanni in Viridian Gym",
        action_type="battle",
        category="battling",
        target_location="Route 23",
        navigation_hint=(
            "Giovanni's Rhydon is Lv 50. Every party member should be Lv 45+ minimum. Route 23 "
            "trainers (Lv 39-47) are excellent for EXP. In wild encounters, always open with "
            "your weakest Pokémon first and switch to a stronger mon to finish — this ensures "
            "the whole team levels up, not just your lead. Water, Grass, and Ice are super "
            "effective on his Ground/Rock team. Flying-types are immune to Ground moves."
        ),
        completion_condition="team_level_45_giovanni",
        priority=1,
        prerequisite_story_objective="viridian_gym_072"
    ),
    DirectObjective(
        id="battle_020",
        description="Catch Zapdos at the Power Plant (optional legendary, great for Elite Four)",
        action_type="catch",
        category="battling",
        target_location="Power Plant",
        navigation_hint="Surf east on Route 10 to the Power Plant. Zapdos (Lv 50, Electric/Flying) is at the end. Save before fighting. Use Master Ball or weaken and throw Ultra Balls. Zapdos's Thunderbolt/Thunder devastates Lorelei and Lance's Gyarados.",
        completion_condition="zapdos_caught",
        priority=2,
        optional=True,
        prerequisite_story_objective="viridian_gym_072"
    ),

    # Pre-Elite Four
    DirectObjective(
        id="battle_021",
        description="Train your full team to Lv 55+ before entering the Elite Four",
        action_type="battle",
        category="battling",
        target_location="Victory Road",
        navigation_hint=(
            "Elite Four have Lv 53-62 Pokémon. Champion Blue has Lv 59-65. Aim for Lv 55+ "
            "across ALL 6 party members — no healing between Elite Four rooms, so every slot "
            "must pull its weight. Train on Victory Road (Machop, Marowak, Onix, Geodude, "
            "Lv 35-50) or Route 23. In wild encounters, lead with your lowest-level Pokémon "
            "first and switch to a stronger mon to finish the fight, so no team member falls "
            "behind. A bench that caps at Lv 40 while your lead is Lv 55 will cost you."
        ),
        completion_condition="team_level_55_elite4",
        priority=1,
        prerequisite_story_objective="victory_076"
    ),
    DirectObjective(
        id="battle_022",
        description="Stock up on 20+ Full Restores, 15+ Revives at Indigo Plateau Poké Mart",
        action_type="shop",
        category="battling",
        target_location="Indigo Plateau",
        navigation_hint="Indigo Plateau Poké Mart: Full Restore $3000, Revive $1500, Max Potion $2500, Full Heal $600, Ultra Ball $1200, Max Repel $700. Buy: 20 Full Restores ($60k), 15 Revives ($22.5k), 10 Max Potions ($25k), 10 Full Heals ($6k). Spend generously — this is your last chance before the final gauntlet.",
        completion_condition="elite4_supplies_bought",
        priority=1,
        prerequisite_story_objective="victory_076"
    ),
]

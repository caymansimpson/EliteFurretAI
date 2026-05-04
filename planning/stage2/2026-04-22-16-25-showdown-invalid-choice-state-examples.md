# Showdown Invalid-Choice Examples

This document contains representative examples of invalid moves rejected by the Showdown server after the target-legality centralization pass. It is meant to make the debugging loop concrete by showing what the battle turns looked like, what the poke-env battle state looked like at the rejected move, and what request payload the websocket player was acting on.

## Commander-family: Dondozo using Tatsugiri move
- Battle tag: battle-gen9vgc2024regg-623017
- Turn: 3
- Attempted choice: /choose move dracometeor 1, move orderup 1
- Error: [Invalid choice] Can't move: Your Dondozo doesn't have a move matching dracometeor

### Battle Turns Leading To The Rejected Move
Turn #0:
Observed My Team: <empty>
Observed Opp Team: <empty>
Observed Events:
  - |init|battle
  - |title|showdiagp1985440 vs. showdiagp2985440
  - |j|☆showdiagp1985440
  - |j|☆showdiagp2985440
  - |gametype|doubles
  - |player|p1|showdiagp1985440|2|
  - |player|p2|showdiagp2985440|169|
  - |gen|9
  - |tier|[Gen 9] VGC 2024 Reg G
  - |rule|Species Clause: Limit one of each Pokémon
  - |rule|Item Clause: Limit 1 of each item
  - |clearpoke
  - |poke|p1|Tatsugiri, L50, F|
  - |poke|p1|Chi-Yu, L50|
  - |poke|p1|Dragonite, L50, F|
  - |poke|p1|Dondozo, L50, F|
  - |poke|p1|Gholdengo, L50|
  - |poke|p1|Miraidon, L50|
  - |poke|p2|Rillaboom, L50, F|
  - |poke|p2|Pelipper, L50, F|
  - |poke|p2|Incineroar, L50, M|
  - |poke|p2|Zacian-*, L50|
  - |poke|p2|Chien-Pao, L50|
  - |poke|p2|Urshifu-*, L50, F|
  - |teampreview|4
  - |uhtml|otsrequest|<button name="send" value="/acceptopenteamsheets" class="button" style="margin-right: 10px;"><strong>Accept Open Team Sheets</strong></button><button name="send" value="/rejectopenteamsheets" class="button" style="margin-top: 10px"><strong>Deny Open Team Sheets</strong></button>
  - |
  - |teamsize|p1|4
  - |teamsize|p2|4
  - |start
  - |switch|p1a: Chi-Yu|Chi-Yu, L50|131/131
  - |switch|p1b: Dondozo|Dondozo, L50, F|226/226
  - |switch|p2a: Pelipper|Pelipper, L50, F|100/100
  - |switch|p2b: Incineroar|Incineroar, L50, M|100/100
  - |-ability|p1a: Chi-Yu|Beads of Ruin
  - |-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
  - |-ability|p2b: Incineroar|Intimidate|boost
  - |-unboost|p1a: Chi-Yu|atk|1
  - |-unboost|p1b: Dondozo|atk|1
  - |turn|1
Turn #1:
Observed My Team: p1: Tatsugiri: tatsugiri [active=None, fainted=None]; p1: Chi-Yu: chiyu [active=None, fainted=None]; p1: Dragonite: dragonite [active=None, fainted=None]; p1: Dondozo: dondozo [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Miraidon: miraidon [active=None, fainted=None]
Observed Opp Team: p2: Pelipper: pelipper [active=None, fainted=None]; p2: Incineroar: incineroar [active=None, fainted=None]
Observed Events:
  - |
  - |switch|p2b: Urshifu|Urshifu-Rapid-Strike, L50, F|100/100
  - |move|p1a: Chi-Yu|Heat Wave|p2b: Urshifu|[spread] p2a,p2b
  - |-resisted|p2a: Pelipper
  - |-resisted|p2b: Urshifu
  - |-damage|p2a: Pelipper|81/100
  - |-damage|p2b: Urshifu|79/100
  - |-status|p2a: Pelipper|brn
  - |move|p1b: Dondozo|Wave Crash|p2a: Pelipper
  - |-resisted|p2a: Pelipper
  - |-damage|p2a: Pelipper|59/100 brn
  - |-damage|p1b: Dondozo|189/226|[from] item: Rocky Helmet|[of] p2a: Pelipper
  - |-damage|p1b: Dondozo|177/226|[from] Recoil
  - |move|p2a: Pelipper|Hurricane|p1a: Chi-Yu
  - |-damage|p1a: Chi-Yu|68/131
  - |
  - |-weather|RainDance|[upkeep]
  - |-heal|p1b: Dondozo|191/226|[from] item: Leftovers
  - |-damage|p2a: Pelipper|53/100 brn|[from] brn
  - |upkeep
  - |turn|2
Turn #2:
Observed My Team: p1: Tatsugiri: tatsugiri [active=None, fainted=None]; p1: Chi-Yu: chiyu [active=None, fainted=None]; p1: Dragonite: dragonite [active=None, fainted=None]; p1: Dondozo: dondozo [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Miraidon: miraidon [active=None, fainted=None]
Observed Opp Team: p2: Pelipper: pelipper [active=None, fainted=None]; p2: Incineroar: incineroar [active=None, fainted=None]; p2: Urshifu: urshifurapidstrike [active=None, fainted=None]
Observed Events:
  - |
  - |-terastallize|p1b: Dondozo|Grass
  - |move|p2b: Urshifu|Aqua Jet|p1b: Dondozo
  - |-resisted|p1b: Dondozo
  - |-damage|p1b: Dondozo|163/226
  - |move|p1a: Chi-Yu|Dark Pulse|p2b: Urshifu
  - |-resisted|p2b: Urshifu
  - |-damage|p2b: Urshifu|34/100
  - |move|p1b: Dondozo|Order Up|p2b: Urshifu
  - |-damage|p2b: Urshifu|19/100
  - |move|p2a: Pelipper|Weather Ball|p1a: Chi-Yu
  - |-supereffective|p1a: Chi-Yu
  - |-damage|p1a: Chi-Yu|0 fnt
  - |faint|p1a: Chi-Yu
  - |
  - |-weather|RainDance|[upkeep]
  - |-heal|p1b: Dondozo|177/226|[from] item: Leftovers
  - |-damage|p2a: Pelipper|47/100 brn|[from] brn
  - |upkeep
  - |
  - |switch|p1a: Tatsugiri|Tatsugiri, L50, F|149/149
  - |-activate|p1a: Tatsugiri|ability: Commander|[of] p1b: Dondozo
  - |-boost|p1b: Dondozo|atk|2
  - |-boost|p1b: Dondozo|spa|2
  - |-boost|p1b: Dondozo|spe|2
  - |-boost|p1b: Dondozo|def|2
  - |-boost|p1b: Dondozo|spd|2
  - |turn|3
Current Observation; Turn #3:
Observed My Team: p1: Tatsugiri: tatsugiri [active=None, fainted=None]; p1: Chi-Yu: chiyu [active=None, fainted=None]; p1: Dragonite: dragonite [active=None, fainted=None]; p1: Dondozo: dondozo [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Miraidon: miraidon [active=None, fainted=None]
Observed Opp Team: p2: Pelipper: pelipper [active=None, fainted=None]; p2: Incineroar: incineroar [active=None, fainted=None]; p2: Urshifu: urshifurapidstrike [active=None, fainted=None]
Observed Events:
  - <no events yet>

### Poke-Env Battle State At The Rejected Move
```text
Battle battle-gen9vgc2024regg-623017 turn=3 player_role=p1
Perspective: showdiagp1985440 vs showdiagp2985440
Force Switch: [False, False]
Can Tera: [False, False]
My Active:
  - slot 0: tatsugiri [active=True, fainted=False, hp=1.0, status=None]
  - slot 1: dondozo [active=True, fainted=False, hp=0.7831858407079646, status=None]
Opp Active:
  - slot 0: pelipper [active=True, fainted=False, hp=0.47, status=BRN (status) object]
  - slot 1: urshifurapidstrike [active=True, fainted=False, hp=0.19, status=None]
My Team:
  - p1: Tatsugiri: tatsugiri [active=True, fainted=False, hp=1.0, status=None]
  - p1: Chi-Yu: chiyu [active=False, fainted=True, hp=0, status=FNT (status) object]
  - p1: Dragonite: dragonite [active=False, fainted=False, hp=1.0, status=None]
  - p1: Dondozo: dondozo [active=True, fainted=False, hp=0.7831858407079646, status=None]
  - p1: Gholdengo: gholdengo [active=False, fainted=False, hp=1.0, status=None]
  - p1: Miraidon: miraidon [active=False, fainted=False, hp=1.0, status=None]
Opp Team:
  - p2: Pelipper: pelipper [active=True, fainted=False, hp=0.47, status=BRN (status) object]
  - p2: Incineroar: incineroar [active=False, fainted=False, hp=1.0, status=None]
  - p2: Urshifu: urshifurapidstrike [active=True, fainted=False, hp=0.19, status=None]
Current Observation:
Observed My Team: p1: Tatsugiri: tatsugiri [active=None, fainted=None]; p1: Chi-Yu: chiyu [active=None, fainted=None]; p1: Dragonite: dragonite [active=None, fainted=None]; p1: Dondozo: dondozo [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Miraidon: miraidon [active=None, fainted=None]
Observed Opp Team: p2: Pelipper: pelipper [active=None, fainted=None]; p2: Incineroar: incineroar [active=None, fainted=None]; p2: Urshifu: urshifurapidstrike [active=None, fainted=None]
Observed Events:
  - <no events yet>
```

### Request At The Rejected Move
```text
Request type: turn
rqid=10
wait=False
teamPreview=False
forceSwitch=None
Active Request Payloads:
  - slot 0: canTera=None trapped=True maybeTrapped=None commanding=None
    moves:
      - id=muddywater move=Muddy Water disabled=False target=allAdjacentFoes pp=16
      - id=icywind move=Icy Wind disabled=False target=allAdjacentFoes pp=24
      - id=dracometeor move=Draco Meteor disabled=False target=normal pp=8
      - id=sleeptalk move=Sleep Talk disabled=False target=self pp=16
  - slot 1: canTera=None trapped=True maybeTrapped=None commanding=None
    moves:
      - id=orderup move=Order Up disabled=False target=normal pp=15
      - id=protect move=Protect disabled=False target=self pp=16
      - id=wavecrash move=Wave Crash disabled=False target=normal pp=15
      - id=earthquake move=Earthquake disabled=False target=allAdjacent pp=16
Side Pokemon:
  - slot 0: p1: Tatsugiri active=True condition=149/149 item=choicescarf teraUsed= commanding=True
  - slot 1: p1: Dondozo active=True condition=177/226 item=leftovers teraUsed=Grass commanding=False
  - slot 2: p1: Chi-Yu active=False condition=0 fnt item=focussash teraUsed= commanding=False
  - slot 3: p1: Miraidon active=False condition=193/193 item=assaultvest teraUsed= commanding=False
```

## Target mismatch: Uproar
- Battle tag: battle-gen9vgc2024regg-622906
- Turn: 5
- Attempted choice: /choose move uproar 1, move woodhammer 1
- Error: [Invalid choice] Can't move: You can't choose a target for Uproar

### Battle Turns Leading To The Rejected Move
Turn #0:
Observed My Team: <empty>
Observed Opp Team: <empty>
Observed Events:
  - |init|battle
  - |title|showdiagp1985440 vs. showdiagp2985440
  - |j|☆showdiagp1985440
  - |j|☆showdiagp2985440
  - |gametype|doubles
  - |player|p1|showdiagp1985440|2|
  - |player|p2|showdiagp2985440|169|
  - |gen|9
  - |tier|[Gen 9] VGC 2024 Reg G
  - |rule|Species Clause: Limit one of each Pokémon
  - |rule|Item Clause: Limit 1 of each item
  - |clearpoke
  - |poke|p1|Calyrex-Ice, L50|
  - |poke|p1|Incineroar, L50, F|
  - |poke|p1|Rillaboom, L50, F|
  - |poke|p1|Gholdengo, L50|
  - |poke|p1|Farigiraf, L50, M|
  - |poke|p1|Amoonguss, L50, M|
  - |poke|p2|Weezing-Galar, L50, M|
  - |poke|p2|Dondozo, L50, M|
  - |poke|p2|Calyrex-Shadow, L50|
  - |poke|p2|Maushold-Four, L50|
  - |poke|p2|Tatsugiri, L50, F|
  - |poke|p2|Iron Hands, L50|
  - |teampreview|4
  - |uhtml|otsrequest|<button name="send" value="/acceptopenteamsheets" class="button" style="margin-right: 10px;"><strong>Accept Open Team Sheets</strong></button><button name="send" value="/rejectopenteamsheets" class="button" style="margin-top: 10px"><strong>Deny Open Team Sheets</strong></button>
  - |
  - |teamsize|p1|4
  - |teamsize|p2|4
  - |start
  - |switch|p1a: Calyrex|Calyrex-Ice, L50|207/207
  - |switch|p1b: Rillaboom|Rillaboom, L50, F|207/207
  - |switch|p2a: Dondozo|Dondozo, L50, M, shiny|100/100
  - |switch|p2b: Maushold|Maushold-Four, L50|100/100
  - |-ability|p1a: Calyrex|As One
  - |-ability|p1a: Calyrex|Unnerve
  - |-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p1b: Rillaboom
  - |turn|1
Turn #1:
Observed My Team: p1: Calyrex: calyrexice [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Amoonguss: amoonguss [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]
Observed Events:
  - |
  - |move|p2b: Maushold|Protect|p2b: Maushold
  - |-singleturn|p2b: Maushold|Protect
  - |move|p1b: Rillaboom|Wood Hammer|p2a: Dondozo
  - |-supereffective|p2a: Dondozo
  - |-damage|p2a: Dondozo|12/100
  - |-damage|p1b: Rillaboom|141/207|[from] Recoil
  - |move|p2a: Dondozo|Wave Crash|p1b: Rillaboom
  - |-resisted|p1b: Rillaboom
  - |-damage|p1b: Rillaboom|94/207
  - |-damage|p2a: Dondozo|5/100|[from] Recoil
  - |move|p1a: Calyrex|Glacial Lance|p2b: Maushold|[spread] p2a
  - |-activate|p2b: Maushold|move: Protect
  - |-resisted|p2a: Dondozo
  - |-damage|p2a: Dondozo|0 fnt
  - |faint|p2a: Dondozo
  - |-ability|p1a: Calyrex|Chilling Neigh|boost
  - |-boost|p1a: Calyrex|atk|1
  - |
  - |-heal|p1b: Rillaboom|106/207|[from] Grassy Terrain
  - |upkeep
  - |
  - |switch|p2a: Calyrex|Calyrex-Shadow, L50|100/100
  - |-ability|p2a: Calyrex|As One
  - |-ability|p2a: Calyrex|Unnerve
  - |turn|2
Turn #2:
Observed My Team: p1: Calyrex: calyrexice [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Amoonguss: amoonguss [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]; p2: Calyrex: calyrexshadow [active=None, fainted=None]
Observed Events:
  - |
  - |-terastallize|p2b: Maushold|Grass
  - |move|p2b: Maushold|Feint|p1b: Rillaboom
  - |-damage|p1b: Rillaboom|90/207
  - |move|p2a: Calyrex|Shadow Ball|p1a: Calyrex
  - |-supereffective|p1a: Calyrex
  - |-damage|p1a: Calyrex|0 fnt
  - |faint|p1a: Calyrex
  - |-ability|p2a: Calyrex|Grim Neigh|boost
  - |-boost|p2a: Calyrex|spa|1
  - |move|p1b: Rillaboom|Wood Hammer|p2b: Maushold
  - |-resisted|p2b: Maushold
  - |-damage|p2b: Maushold|62/100
  - |-damage|p1b: Rillaboom|56/207|[from] item: Rocky Helmet|[of] p2b: Maushold
  - |-damage|p1b: Rillaboom|33/207|[from] Recoil
  - |
  - |-heal|p2b: Maushold|68/100|[from] Grassy Terrain
  - |-heal|p1b: Rillaboom|45/207|[from] Grassy Terrain
  - |upkeep
  - |
  - |switch|p1a: Amoonguss|Amoonguss, L50, M|218/218
  - |turn|3
Turn #3:
Observed My Team: p1: Amoonguss: amoonguss [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Calyrex: calyrexice [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]; p2: Calyrex: calyrexshadow [active=None, fainted=None]
Observed Events:
  - |
  - |move|p2b: Maushold|Protect|p2b: Maushold
  - |-singleturn|p2b: Maushold|Protect
  - |move|p1a: Amoonguss|Rage Powder|p1a: Amoonguss
  - |-singleturn|p1a: Amoonguss|move: Rage Powder
  - |move|p1b: Rillaboom|Grassy Glide|p2b: Maushold
  - |-activate|p2b: Maushold|move: Protect
  - |move|p2a: Calyrex|Shadow Ball|p1a: Amoonguss
  - |-damage|p1a: Amoonguss|1/218
  - |
  - |-heal|p2b: Maushold|75/100|[from] Grassy Terrain
  - |-heal|p1b: Rillaboom|57/207|[from] Grassy Terrain
  - |-heal|p1a: Amoonguss|14/218|[from] Grassy Terrain
  - |upkeep
  - |turn|4
Turn #4:
Observed My Team: p1: Amoonguss: amoonguss [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Calyrex: calyrexice [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]; p2: Calyrex: calyrexshadow [active=None, fainted=None]
Observed Events:
  - |
  - |-terastallize|p1b: Rillaboom|Fire
  - |move|p2b: Maushold|Feint|p1a: Amoonguss
  - |-damage|p1a: Amoonguss|1/218
  - |move|p1b: Rillaboom|Grassy Glide|p2b: Maushold
  - |-resisted|p2b: Maushold
  - |-damage|p2b: Maushold|58/100
  - |-damage|p1b: Rillaboom|23/207|[from] item: Rocky Helmet|[of] p2b: Maushold
  - |move|p2a: Calyrex|Shadow Ball|p1a: Amoonguss
  - |-damage|p1a: Amoonguss|0 fnt
  - |faint|p1a: Amoonguss
  - |-ability|p2a: Calyrex|Grim Neigh|boost
  - |-boost|p2a: Calyrex|spa|1
  - |
  - |-heal|p2b: Maushold|64/100|[from] Grassy Terrain
  - |-heal|p1b: Rillaboom|35/207|[from] Grassy Terrain
  - |upkeep
  - |
  - |switch|p1a: Farigiraf|Farigiraf, L50, M|225/225
  - |turn|5
Current Observation; Turn #5:
Observed My Team: p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Amoonguss: amoonguss [active=None, fainted=None]; p1: Calyrex: calyrexice [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]; p2: Calyrex: calyrexshadow [active=None, fainted=None]
Observed Events:
  - <no events yet>

### Poke-Env Battle State At The Rejected Move
```text
Battle battle-gen9vgc2024regg-622906 turn=5 player_role=p1
Perspective: showdiagp1985440 vs showdiagp2985440
Force Switch: [False, False]
Can Tera: [False, False]
My Active:
  - slot 0: farigiraf [active=True, fainted=False, hp=1.0, status=None]
  - slot 1: rillaboom [active=True, fainted=False, hp=0.16908212560386474, status=None]
Opp Active:
  - slot 0: calyrexshadow [active=True, fainted=False, hp=1.0, status=None]
  - slot 1: mausholdfour [active=True, fainted=False, hp=0.64, status=None]
My Team:
  - p1: Farigiraf: farigiraf [active=True, fainted=False, hp=1.0, status=None]
  - p1: Incineroar: incineroar [active=True, fainted=False, hp=1.0, status=None]
  - p1: Rillaboom: rillaboom [active=True, fainted=False, hp=0.16908212560386474, status=None]
  - p1: Gholdengo: gholdengo [active=False, fainted=False, hp=1.0, status=None]
  - p1: Amoonguss: amoonguss [active=False, fainted=True, hp=0, status=FNT (status) object]
  - p1: Calyrex: calyrexice [active=False, fainted=True, hp=0, status=FNT (status) object]
Opp Team:
  - p2: Dondozo: dondozo [active=False, fainted=True, hp=0, status=FNT (status) object]
  - p2: Maushold: mausholdfour [active=True, fainted=False, hp=0.64, status=None]
  - p2: Calyrex: calyrexshadow [active=True, fainted=False, hp=1.0, status=None]
Current Observation:
Observed My Team: p1: Farigiraf: farigiraf [active=None, fainted=None]; p1: Incineroar: incineroar [active=None, fainted=None]; p1: Rillaboom: rillaboom [active=None, fainted=None]; p1: Gholdengo: gholdengo [active=None, fainted=None]; p1: Amoonguss: amoonguss [active=None, fainted=None]; p1: Calyrex: calyrexice [active=None, fainted=None]
Observed Opp Team: p2: Dondozo: dondozo [active=None, fainted=None]; p2: Maushold: mausholdfour [active=None, fainted=None]; p2: Calyrex: calyrexshadow [active=None, fainted=None]
Observed Events:
  - <no events yet>
```

### Request At The Rejected Move
```text
Request type: turn
rqid=18
wait=False
teamPreview=False
forceSwitch=None
Active Request Payloads:
  - slot 0: canTera=None trapped=None maybeTrapped=None commanding=None
    moves:
      - id=uproar move=Uproar disabled=False target=randomNormal pp=16
      - id=helpinghand move=Helping Hand disabled=False target=adjacentAlly pp=32
      - id=trickroom move=Trick Room disabled=False target=all pp=8
      - id=dazzlinggleam move=Dazzling Gleam disabled=False target=allAdjacentFoes pp=16
  - slot 1: canTera=None trapped=None maybeTrapped=None commanding=None
    moves:
      - id=woodhammer move=Wood Hammer disabled=False target=normal pp=22
      - id=uturn move=U-turn disabled=False target=normal pp=32
      - id=grassyglide move=Grassy Glide disabled=False target=normal pp=30
      - id=fakeout move=Fake Out disabled=False target=normal pp=16
Side Pokemon:
  - slot 0: p1: Farigiraf active=True condition=225/225 item=safetygoggles teraUsed= commanding=False
  - slot 1: p1: Rillaboom active=True condition=35/207 item=assaultvest teraUsed=Fire commanding=False
  - slot 2: p1: Amoonguss active=False condition=0 fnt item=rockyhelmet teraUsed= commanding=False
  - slot 3: p1: Calyrex active=False condition=0 fnt item=clearamulet teraUsed= commanding=False
```
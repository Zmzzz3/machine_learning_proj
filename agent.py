# Utility helpers
import heapq
from itertools import count
from PIL import Image

class Agent:

    def __init__(self):
        self.actions = []
        self.visited = {}
        self.info = None

        self.model = get_model()
        self.dataset = ['boots', 'box', 'coin', 'exit', 'floor', 'gem', 'ghost', 'human', 'key', 'lava', 'locked', 'shield', 'wall']
    
    class Information:

        def __init__(self):
            # globally shared 
            self.width = 0
            self.height = 0
            self.item_reg = {}
            self.obj_count = 0
            self.walls = set()
            self.lava = set()
            self.exit = (0, 0)
            self.costs = [[]]
            self.cost_cache = {}

            # node to node updated
            self.objectives = []
            self.bit_idx = 0
            self.item_mask = 0
            self.doors = []
            self.boxes = []
            self.initial = (0, 0)

    def initialize(self, state: GridState):
        info = Agent.Information()
        info.width = state.width
        info.height = state.height
        hp = 5
        items = 0
        simple_map = [[3] * info.width for i in range(info.height)]

        for y in range(state.height):
            for x in range(state.width):
                for entity in state.objects_at((x, y)):
                    match entity:
                        case FloorEntity():
                            pass
                        case AgentEntity():
                            info.initial = (x, y)
                            hp = entity.health.current_health
                        case ExitEntity():
                            info.exit = (x, y)
                        case LockedDoorEntity():
                            info.doors.append((x, y))
                        case WallEntity():
                            simple_map[y][x] = 0
                            info.walls.add((x, y))
                        case LavaEntity():
                            info.lava.add((x, y))
                        case BoxEntity():
                            info.boxes.append((x, y))
                        case GemEntity():
                            info.objectives.append((x, y))
                            info.obj_count += 1
                        case CoinEntity():
                            simple_map[y][x] = 1

                    if isinstance(entity, (CoinEntity, GemEntity, KeyEntity, SpeedPowerUpEntity, PhasingPowerUpEntity, ShieldPowerUpEntity)):
                        info.item_reg[(x, y)] = (info.bit_idx, entity)
                        info.bit_idx += 1
                        items += 1

        # representing item pickup state as sequence of bits
        info.item_mask = (1 << items) - 1
        self.all_paths(simple_map, info.objectives + [info.exit], info)

        player = Agent.Player(hp, 0, 0, 0, 0)
        node = Agent.Node(None, None, info.initial, 0, player, info.objectives, info.item_mask, info.doors, info.boxes, info)
        self.info = info
        return node

    def extract_tiles(self, state, width, height):
        img = Image.fromarray(state["image"])
        tile_w = img.size[0] // width
        tile_h = img.size[1] // height
        
        tiles = []
        for row in range(height):
            row_tiles = []
            for col in range(width):
                tile = img.crop((
                    col * tile_w,
                    row * tile_h,
                    (col + 1) * tile_w,
                    (row + 1) * tile_h
                ))
                row_tiles.append(tile)
            tiles.append(row_tiles)
        return tiles
    
    def parse(self, state: ImageObservation):
        from torchvision.transforms import v2

        info = Agent.Information()
        info.width = state["info"]["config"]["width"]
        info.height = state["info"]["config"]["height"]
        hp = 5
        items = 0
        simple_map = [[3] * info.width for i in range(info.height)]
    
        tiles = self.extract_tiles(state, info.width, info.height)
        for y in range(info.height):
            for x in range(info.width):
                transform = v2.Compose([
                    v2.ToTensor(),
                    v2.Resize((128, 128)),
                    v2.Lambda(lambda x: x[:3]),        
                    v2.Normalize([0.5] * 3, [0.5] * 3)
                ])
                
                entity = self.model(transform(tiles[y][x]).unsqueeze(0)) 
                label_idx = entity.argmax(dim=1).item()
                label = self.dataset[label_idx]
                match label:
                    case "floor":
                        pass
                    case "human":
                        info.initial = (x, y)
                        hp = state["info"]["agent"]["health"]["current_health"]
                    case "exit":
                        info.exit = (x, y)
                    case "locked":
                        info.doors.append((x, y))
                    case "wall":
                        simple_map[y][x] = 0
                        info.walls.add((x, y))
                    case "lava":
                        info.lava.add((x, y))
                    case "box":
                        info.boxes.append((x, y))
                    case "gem":
                        info.objectives.append((x, y))
                        info.obj_count += 1
                    case "coin":
                        simple_map[y][x] = 1

                if label in ("coin", "gem", "key", "boots", "ghost", "shield"):
                    info.item_reg[(x, y)] = (info.bit_idx, label)
                    info.bit_idx += 1
                    items += 1

        # representing item pickup state as sequence of bits
        info.item_mask = (1 << items) - 1
        self.all_paths(simple_map, info.objectives + [info.exit], info)

        player = Agent.Player(hp, 0, 0, 0, 0)
        node = Agent.Node(None, None, info.initial, 0, player, info.objectives, info.item_mask, info.doors, info.boxes, info)
        self.info = info
        return node

    # precompute on map with only walls and coins, to use for heuristic
    def all_paths(self, map, objectives, info):
        h, w, o = len(map), len(map[0]), len(objectives)
        info.costs = [[[450] * o for i in range(w)] for j in range(h)]

        for idx, objective in enumerate(objectives):
            visited = [[False] * w for j in range(h)]
            pq = []
            tiebreak = count()
            heapq.heappush(pq, (0, next(tiebreak), objective))

            # ucs to store costs at each position from each objective
            while pq:
                cost, tie, node = heapq.heappop(pq)
                x, y = node[0], node[1]
                if visited[y][x]: continue
                visited[y][x] = True
                info.costs[y][x][idx] = cost

                adj = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                for child in adj:
                    cx, cy = child
                    
                    # map[y][x] to give walls a cost for when agent has phase
                    if cx < 0 or cx >= w or cy < 0 or cy >= h or not map[y][x]: continue
                    heapq.heappush(pq, (cost + map[cy][cx], next(tiebreak), child))

    class Player:
        
        def __init__(self, hp, key, speed, phase, shield):
            self.hp = hp
            self.key = key
            self.speed = speed
            self.phase = phase
            self.shield = shield

        def copy(self):
            return Agent.Player(self.hp, self.key, self.speed, self.phase, self.shield)

        def update_status(self, hp, key, speed, phase, shield):
            while hp: 
                if self.phase: pass
                elif self.shield: self.shield = max(0, self.shield - 1)
                else: self.hp -= 2 
                hp -= 1
            if key: self.key += key
            
            self.speed = 5 if speed else max(0, self.speed - 1)
            self.phase = 5 if phase else max(0, self.phase - 1)
            if shield: self.shield = 5
            return self.hp > 0
        
        def __hash__(self): 
            return hash((self.hp, self.key, self.speed, self.phase, self.shield))
        
        def __eq__(self, other):
            return (self.hp == other.hp and
                    self.key == other.key and
                    self.speed == other.speed and
                    self.phase == other.phase and
                    self.shield == other.shield)
        
    class Node:
        
        def __init__(self, prev, action, pos, score, player, obj, items, door, box, info):
            self.prev = prev
            self.action = action
            self.pos = pos
            self.score = score
            self.player = player
            self.obj = obj
            self.items = items
            self.door = door
            self.box = box
            self.info = info

            # score is left out since it is part of priority and obj is left out since it is accounted for in item
            self.hash = hash((self.pos, self.player, self.items, tuple(self.door), tuple(self.box)))

        def use_key(self, successors):
            if not self.player.key: return
            x, y = self.pos
            player = self.player.copy()
            door = self.door.copy()
            unlocked = False

            for pos in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if not player.key: break
                if pos in door:
                    unlocked = player.update_status(0, -1, 0, 0, 0)
                    door = [d for d in door if d != pos]

            if unlocked:
                successors.append(Agent.Node(self, Action.USE_KEY, self.pos, self.score - 3, player,
                                            self.obj.copy(), self.items, door, self.box.copy(), self.info))
                
        def wait(self, successors):
            # no reason to wait otherwise, since movement is not impaired
            if not self.player.speed: return
            player = self.player.copy()
            player.update_status(0, 0, 0, 0, 0)
            successors.append(Agent.Node(self, Action.WAIT, self.pos, self.score - 3, player, 
                                         self.obj.copy(), self.items, self.door.copy(), self.box.copy(), self.info))
            return

        def pick_up(self, successors):
            if self.pos in self.info.item_reg and (self.items >> self.info.item_reg[self.pos][0]) & 1:
                bit_idx, item = self.info.item_reg[self.pos]
                items = self.items & ~(1 << bit_idx)
                player = self.player.copy()
                match item:

                    case "coin" | CoinEntity():
                        player.update_status(0, 0, 0, 0, 0)
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score + 2, player, 
                                                 self.obj.copy(), items, self.door.copy(), self.box.copy(), self.info))
                        return
                    
                    case "gem" | GemEntity():
                        player.update_status(0, 0, 0, 0, 0)
                        obj = [o for o in self.obj if o != self.pos]
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score - 3, player, 
                                                 obj, items, self.door.copy(), self.box.copy(), self.info))
                        return

                    case "key" | KeyEntity():
                        player.update_status(0, 1, 0, 0, 0)
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score - 3, player, 
                                                 self.obj.copy(), items, self.door.copy(), self.box.copy(), self.info))
                        return

                    case "boots" | SpeedPowerUpEntity():
                        player.update_status(0, 0, 1, 0, 0)
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score - 3, player, 
                                                 self.obj.copy(), items, self.door.copy(), self.box.copy(), self.info))
                        return

                    case "ghost" | PhasingPowerUpEntity():
                        player.update_status(0, 0, 0, 1, 0)
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score - 3, player, 
                                                 self.obj.copy(), items, self.door.copy(), self.box.copy(), self.info))
                        return

                    case "shield" | ShieldPowerUpEntity():
                        player.update_status(0, 0, 0, 0, 1)
                        successors.append(Agent.Node(self, Action.PICK_UP, self.pos, self.score - 3, player, 
                                                 self.obj.copy(), items, self.door.copy(), self.box.copy(), self.info))
                        return

        def move(self, successors):
            x, y = self.pos
            directions = {
                Action.UP:    (0, -1),
                Action.DOWN:  (0, 1),
                Action.LEFT:  (-1, 0),
                Action.RIGHT: (1, 0),
            }

            for action, (dx, dy) in directions.items():
                next = None
                pos1 = (x + dx, y + dy)
                if self.invalid(pos1): continue
                pos2 = (x + 2*dx, y + 2*dy)
                invalid2 = self.invalid(pos2)

                new_box = self.box.copy()
                damage = 0

                blocked1 = self.blocked(pos1)
                blocked2 = self.blocked(pos2)

                lava1 = pos1 in self.info.lava
                lava2 = pos2 in self.info.lava

                box1 = pos1 in new_box
                box2 = pos2 in new_box

                if self.player.phase: next = self.move_player(action, pos1, 0, new_box)
                elif blocked1: continue
                elif box1:
                    if invalid2 or blocked2 or lava2 or box2:
                        continue
                    else:
                        new_box = self.move_box(pos1, pos2)
                        box2 = True
                        if lava1: damage += 1
                        next = self.move_player(action, pos1, damage, new_box)
                else:
                    if lava1: damage += 1
                    next = self.move_player(action, pos1, damage, new_box)

                if invalid2 or not self.player.speed: 
                    if next: successors.append(next)
                    continue

                pos3 = (x + 3*dx, y + 3*dy)
                invalid3 = self.invalid(pos3)
                blocked3 = self.blocked(pos3)
                lava3 = pos3 in self.info.lava
                box3 = pos3 in new_box

                if self.player.phase: next = self.move_player(action, pos2, 0, new_box)
                elif blocked2: pass
                elif box2:
                    if invalid3 or blocked3 or lava3 or box3:
                        pass
                    else:
                        new_box = self.move_box(pos2, pos3)
                        if lava2: damage += 1
                        next = self.move_player(action, pos2, damage, new_box)
                else: 
                    if lava2: damage += 1
                    next = self.move_player(action, pos2, damage, new_box)
                if next: successors.append(next)

        def invalid(self, pos):
            return pos[0] < 0 or pos[0] >= self.info.width or pos[1] < 0 or pos[1] >= self.info.height

        def blocked(self, pos):
            return pos in self.info.walls or pos in self.door 

        def move_box(self, curr, target):
            return [b for b in self.box if b != curr] + [target]

        def move_player(self, action, target, damage, box):
            player = self.player.copy()
            if not player.update_status(damage, 0, 0, 0, 0): return None
            return Agent.Node(self, action, target, self.score - 3, player, 
                              self.obj.copy(), self.items, self.door.copy(), box, self.info)        

        def successor(self): 
            successors = []
            self.use_key(successors)
            self.wait(successors)
            self.pick_up(successors)
            self.move(successors)
            return successors
        
        def win(self):
            return not self.obj and self.pos == self.info.exit

        def __hash__(self): 
            return self.hash
        
        def __eq__(self, other):
            return (self.pos == other.pos and 
                    self.player == other.player and
                    self.items == other.items and
                    tuple(self.door) == tuple(other.door) and
                    tuple(self.box) == tuple(other.box))

    # using previously calculated ucs costs to estimate total cost
    def estimate_cost(self, curr, objectives):
        cache = self.info.cost_cache
        key = (curr, frozenset(objectives))
        if key in cache: return cache[key]

        x, y = curr
        if not objectives:
            cost = self.info.costs[y][x][self.info.obj_count]
            cache[key] = cost
            return cost

        cost = min(
            self.info.costs[y][x][self.info.objectives.index(obj)] + self.estimate_cost(obj, [o for o in objectives if o != obj])
            for obj in objectives
        )
        cache[key] = cost
        return cost

    # the heuristic may overestimate due to the presence of movement-altering powerups
    # however the weight can be tuned to provide either a tighter or more a aggressive approach
    def heuristic(self, node):
        x, y = node.pos
        uncollected = [self.info.objectives[i] for i, bit in enumerate(
            (node.items >> self.info.item_reg[g][0]) & 1 
            for g in self.info.objectives
        ) if bit == 1]

        if not uncollected:
            return self.info.costs[y][x][self.info.obj_count]

        return min(
            self.info.costs[y][x][self.info.objectives.index(obj)] + self.estimate_cost(obj, [o for o in uncollected if o != obj])
            for obj in uncollected
        )

    def astar(self, node, heuristic, weight):
        pq = []
        tiebreak = count()
        heapq.heappush(pq, (0, next(tiebreak), node))
        last = None

        while pq:
            node = heapq.heappop(pq)[2]
            self.visited[node] = node.score
            if node.win():
                last = node
                break

            for child in node.successor():
                # only take the best score for a given game state (see node for comparison details)
                if child in self.visited and self.visited[child] > child.score: continue

                g = -child.score
                h = heuristic(child)
                heapq.heappush(pq, (g + h * weight, next(tiebreak), child))

        seq = []
        while last.action:
            seq.append(last.action)
            last = last.prev
        return seq

    def step(self, state: GridState | ImageObservation) -> Action:
        if self.actions: return self.actions.pop()
        node = None
        if isinstance (state, GridState): node = self.initialize(state)
        else: node = self.parse(state)
        self.actions = self.astar(node, self.heuristic, 1.0)
        return self.actions.pop()

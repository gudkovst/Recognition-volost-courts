from functools import singledispatch


class Coordinates:
    x = None
    y = None
    h = None
    w = None
    xc = None # x_center
    yc = None
    xf = None # x_final
    yf = None

    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.xc = x + w / 2
        self.yc = y + h / 2
        self.xf = x + w
        self.yf = y + h

    def contains(self, coords) -> bool:
        ans = self.x <= coords.x and self.y <= coords.y and self.xf >= coords.xf and self.yf >= coords.yf
        return ans
    

class String:
    coords: Coordinates = None

    def __init__(self, x, y, h, w):
        self.coords = Coordinates(x, y, h, w)
    

class Block:
    coords: Coordinates = None
    strings: list[String] = []

    def __init__(self, x, y, h, w):
        self.coords = Coordinates(x, y, h, w)

    def add_string(self, x, y, h, w):
        self.strings.append(String(x, y, h, w))

    def get_strings(self):
        return sorted(self.strings, key=lambda s: s.coords.y)

    def str_in_coords(self, coords) -> bool:
        return self.coords.contains(coords)
    
    def str_in_list(self, st: String) -> bool:
        return st in self.strings
        

class LabelingPage:
    coords: Coordinates = None
    blocks: list[Block] = []

    def __init__(self, h, w):
        self.coords = Coordinates(0, 0, h, w)

    def add_block(self, x, y, h, w):
        self.blocks.append(Block(x, y, h, w))

    def get_blocks(self):
        return sorted(self.blocks, key=lambda b: b.coords.y)

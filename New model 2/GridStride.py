from dataclasses import dataclass
from typing import List

@dataclass 
class GridAndStride:
    grid0: int
    grid1: int
    stride: int
    
    def __str__(self) -> str:
        return f"Grid0: {self.grid0}, Grid1: {self.grid1}, Stride: {self.stride}"

def generateGridsAndStride() -> List[GridAndStride]:
    INPUT_W = 416
    INPUT_H = 416
    STRIDES = [8, 16, 32]
    output = []
    for stride in STRIDES:
        grid_h = INPUT_H // stride
        grid_w = INPUT_W // stride
        for g1 in range (grid_h):
            for g0 in range(grid_w):
                output.append(GridAndStride(g0, g1, stride))
    return output

if __name__ == "__main__":
    grids = generateGridsAndStride()
    import numpy as np
    grids = np.array(grids)
    print("Shape of grids:", grids.shape)
    print("First 5 grids:", grids[:5])

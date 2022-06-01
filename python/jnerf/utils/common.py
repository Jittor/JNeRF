import jittor as jt
import numpy as np

def enlarge(x: jt.Var, size: int):
    if x.shape[0] < size:
        y = jt.empty([size],x.dtype)
        x.assign(y)

class BoundingBox():
    def __init__(self,min=[0,0,0],max=[0,0,0]) -> None:
        self.min=np.array(min)
        self.max=np.array(max)
    def inflate(self,amount:float):
        self.min-=amount
        self.max+=amount
        pass

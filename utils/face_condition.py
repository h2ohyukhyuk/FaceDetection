
from enum import Enum

class FD_Blur(Enum):
  CLEAR = 0
  NORMAL = 1
  HEAVY = 2

class FD_Expression(Enum):
  TYPICAL = 0
  EXAGGERATE = 1

class FD_ILLUMINAION(Enum):
  NORMAL = 0
  EXTREME = 1

class FD_OCCLUSION(Enum):
  NO = 0
  PARTIAL = 1
  HEAVY = 2

class FD_POSE(Enum):
  TYPICAL = 0
  ATYPICAL = 1

class FD_INVALID(Enum):
  FALSE = 0
  TRUE =1
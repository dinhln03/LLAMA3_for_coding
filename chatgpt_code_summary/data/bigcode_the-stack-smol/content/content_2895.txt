class Const:
  """
  常量
  """
  class ConstError(TypeError):pass
  def __setattr__(self, name, value):
    if name in self.__dict__:
      raise self.ConstError("Can't rebind const (%s)" %name)
    self.__dict__[name]=value

LAYOUT = Const()
"""
布局
"""
LAYOUT.SCREEN_WIDTH = 500
LAYOUT.SCREEN_HEIGHT = 600
LAYOUT.SIZE = 4
LAYOUT.TERRAIN_X = 50
LAYOUT.TERRAIN_Y = 20
LAYOUT.TILE_WIDTH = 100
LAYOUT.TILE_HEIGHT = 90
LAYOUT.SCOREBOARD_X = 50
LAYOUT.SCOREBOARD_Y = 400
LAYOUT.POPUP_X = 100
LAYOUT.POPUP_Y = 400
LAYOUT.POPUP_WIDTH = 300
LAYOUT.POPUP_HEIGHT = 200

IMAGE = Const()
"""
图片
"""
IMAGE.TILE = "assets/tile.png"# 地砖
IMAGE.MIST = "assets/mist.png"# 战争迷雾
IMAGE.HERO = "assets/hero.png" # 英雄
IMAGE.MONSTER = "assets/monster.png" # 怪物
IMAGE.PIT = "assets/pit.png" # 陷阱
IMAGE.GOLD = "assets/gold.png" # 黄金
IMAGE.BREEZE = "assets/breeze.png" # 微风
IMAGE.STRENCH = "assets/strench.png" # 臭气

EVENT = Const()
"""
事件
"""
EVENT.GAME_OVER = "gameOver" # 游戏结束
EVENT.GAME_CLEAR = "gameClear" # 游戏通关
EVENT.MONSTER_DEAD = "monsterDead" # 怪兽死亡
EVENT.HERO_WALK = "heroWalk" # 英雄走动
EVENT.HERO_ATTACK = "heroAttack" # 英雄攻击
EVENT.DANGER = "danger" # 遭遇危险

ENCOUNTER = Const()
"""
遭遇
"""
ENCOUNTER.MONSTER = 21 # 怪物
ENCOUNTER.PIT = 22 # 坑洞
ENCOUNTER.GOLD = 10 # 黄金

SCORE = Const()
"""
分数
"""
SCORE.WALK = -1 # 行走
SCORE.WIN = 1000 # 胜利
SCORE.LOSE = -1000 # 失败
SCORE.ATTACK = -10 # 攻击
from bedlam import Game
from bedlam import Scene
from bedlam import Sprite
from balls import Ball

# __pragma__('skip')
document = window = Math = Date = console = 0  # Prevent complaints by optional static checker

# __pragma__('noskip')
# __pragma__('noalias', 'clear')

DEBUG = False


class PVector:
    def __init__(self, xx=0, yy=0):
        self.x = xx
        self.y = yy

    def __str__(self):
        return "PVector({},{})".format(self.x, self.y)

    def reset(self, xx, yy):
        self.x = xx
        self.y = yy
        return self

    def copy(self):
        return PVector.Instance(self.x, self.y)

    def add(self, v):
        self.x = self.x + v.x
        self.y = self.y + v.y
        return self

    def sub(self, v):
        self.x = self.x - v.x
        self.y = self.y - v.y
        return self

    def mult(self, mag):
        self.x = self.x * mag
        self.y = self.y * mag
        return self

    def div(self, mag):
        self.x = self.x / mag
        self.y = self.y / mag
        return self

    def normalize(self, mag=1.0):
        d = Math.sqrt(self.x * self.x + self.y * self.y)
        if d == 0 or mag == 0:
            self.x = 0
            self.y = 0
        else:
            self.x = mag * self.x / d
            self.y = mag * self.y / d
        return self

    def limit(self, mag):
        d = Math.sqrt(self.x * self.x + self.y * self.y)
        if d == 0 or mag == 0:
            return
        if d > mag:
            self.x = mag * self.x / d
            self.y = mag * self.y / d
        return self

    def mag(self):
        return Math.sqrt(self.x * self.x + self.y * self.y)

    @classmethod
    def Instance(cls, xx, yy):
        if cls.pool is None:
            cls.pool = []
            cls.pool_max_size = 10
        if len(cls.pool) == 0:
            return PVector(xx, yy)
        else:
            v = cls.pool.pop()
            v.x = xx
            v.y = yy
            return v

    @classmethod
    def Free(cls, pvector):
        if len(cls.pool) < cls.pool_max_size:
            cls.pool.append


class Boid(Sprite):

    def __init__(self, game, w=10):
        Sprite.__init__(self, game, w, w)
        self.color = 'white'
        self.x = self.game.canvas.width * Math.random()
        self.y = self.game.canvas.height * Math.random()
        angle = 2 * Math.PI * Math.random()
        self.dx = self.game.speed * Math.cos(angle)
        self.dy = self.game.speed * Math.sin(angle)

    def is_close(self, sprite, dist):
        return self.distance(sprite) + self.width / 2 + sprite.width / 2 <= dist

    def distance(self, sprite):
        vx = self.x - sprite.x
        vy = self.y - sprite.y
        self_radius = (self.width + self.height) / 2
        sprite_radius = (sprite.width + sprite.height) / 2
        dist = Math.sqrt(vx * vx + vy * vy) - (self_radius + sprite_radius)
        return dist if dist >= 0 else 0

    def draw(self, ctx):
        global DEBUG
        Sprite.draw(self, ctx)
        angle = self._angle()
        ctx.save()
        ctx.globalCompositeOperation = 'source-over'
        if DEBUG:
            ctx.strokeStyle = '#808080'
            ctx.beginPath()
            ctx.arc(self.x, self.y, self.game.cohesion_radius, 0, 2 * Math.PI)
            ctx.stroke()
            ctx.strokeStyle = '#696969'
            ctx.beginPath()
            ctx.arc(self.x, self.y, self.game.separation_radius + self.width/2, 0, 2 * Math.PI)
            ctx.stroke()
        ctx.lineWidth = 2
        ctx.strokeStyle = self.color
        ctx.fillStyle = self.color
        ctx.beginPath()
        ctx.translate(self.x, self.y)
        ctx.rotate(angle)
        ctx.moveTo(-1 * self.width, -0.5 * self.width)
        ctx.lineTo(self.width, 0)
        ctx.lineTo(-1 * self.width, 0.5 * self.width)
        ctx.lineTo(-1 * self.width, -0.5 * self.width)
        ctx.translate(-1 * self.originX, -1 * self.originY)
        ctx.fill()
        ctx.stroke()
        ctx.restore()

    def _angle(self, a=0.0):
        angle = Math.atan2(self.dy, self.dx) + a
        while angle > 2 * Math.PI:
            angle = angle - 2 * Math.PI
        while angle < 0:
            angle = angle + 2 * Math.PI
        return angle

    def _find(self, boid, dist, clazz=None):
        return self.game.currentScene.find(boid, dist, clazz)

    def update(self, delta_time):
        global DEBUG
        move = PVector.Instance(self.dx, self.dy)

        allignment = self.__calc_allignment().mult(self.game.allignment_mult)
        separation = self.__calc_separation().mult(self.game.separation_mult)
        cohesion = self.__calc_cohesion().mult(self.game.cohesion_mult)
        noise = self.__calc_random_noise().mult(self.game.noise_mult)
        if DEBUG:
            console.log('time={} : allign={} : avoid={} : noise={} : cohese={}'.format(delta_time, allignment.mag(),
                                                                                       separation.mag(), noise.mag(),
                                                                                       cohesion.mag()))

        move.add(allignment)
        move.add(separation)
        move.add(cohesion)
        move.add(noise)
        move.limit(self.game.speed)

        self.dx = move.x
        self.dy = move.y
        self.x = self.x + self.dx * delta_time / 1000.0
        if self.x < 0:
            self.x = self.x + self.game.canvas.width
        elif self.x > self.game.canvas.width:
            self.x = self.x - self.game.canvas.width
        self.y = self.y + self.dy * delta_time / 1000.0
        if self.y < 0:
            self.y = self.y + self.game.canvas.height
        elif self.y > self.game.canvas.height:
            self.y = self.y - self.game.canvas.height

        PVector.Free(move)
        PVector.Free(allignment)
        PVector.Free(separation)
        PVector.Free(noise)

    def __calc_allignment(self):
        steer = PVector.Instance(0, 0)
        for sprite in self._find(self, self.game.allignment_radius, Boid):
            d = self.distance(sprite)
            if d == 0:
                continue
            copy = PVector.Instance(sprite.dx, sprite.dy)
            copy.normalize()
            copy.div(d)
            steer.add(copy)
        return steer

    def __calc_separation(self):
        steer = PVector.Instance(0, 0)
        for sprite in self._find(self, self.game.separation_radius, Sprite):
            d = self.distance(sprite)
            if d == 0:
                continue
            diff = PVector(self.x - sprite.x, self.y - sprite.y)
            diff.normalize()
            diff.div(d)
            steer.add(diff)
        return steer

    def __calc_random_noise(self):
        return PVector.Instance(Math.random() * 2 - 1, Math.random() * 2 - 1)

    def __calc_cohesion(self):
        steer = PVector.Instance(0, 0)
        count = 0
        for sprite in self._find(self, self.game.cohesion_radius, Boid):
            steer.x = steer.x + sprite.x
            steer.y = steer.y + sprite.y
            count = count + 1
        if count > 0:
            steer.x = steer.x / count
            steer.y = steer.y / count
            steer.normalize(0.05)
        return steer


class BoidsScene(Scene):
    def __init__(self, game, name=None, num_boids=8, w=10):
        Scene.__init__(self, game, name)
        self.color = 'black'
        for n in range(num_boids):
            self.append(Boid(self.game, w))
        for n in range(3):
            self.append(Ball(self.game, 30, 10, 'green'))
        for n in range(1):
            self.append(Ball(self.game, 30, 20, 'red'))

    def _clear_screen(self, ctx):
        ctx.save()
        ctx.globalCompositeOperation = 'copy'
        ctx.fillStyle = self.color
        ctx.fillRect(0, 0, self.game.canvas.width, self.game.canvas.height)
        ctx.restore()

    def find(self, boid, dist, clazz=None):
        sprite_list = []
        for sprite in self:
            if clazz is not None and not isinstance(sprite, clazz):
                continue
            if sprite == boid:
                continue
            if boid.is_close(sprite, dist):
                sprite_list.append(sprite)
        return sprite_list


class BoidsGame(Game):
    def __init__(self, name='Boids', loop_time=20):
        Game.__init__(self, name, loop_time)
        sprite_width = 5
        global_scale = sprite_width / 10.0
        self.speed = 100
        self.allignment_radius = 180 * global_scale
        self.separation_radius = 25 * global_scale
        self.cohesion_radius = self.allignment_radius
        self.allignment_mult = 3
        self.separation_mult = 30
        self.cohesion_mult = 25
        self.noise_mult = 5
        self.append(BoidsScene(self, 'BOIDS', 32, sprite_width))

    @staticmethod
    def set_debug(b):
        global DEBUG
        if b is not None and b == 'true':
            DEBUG = True

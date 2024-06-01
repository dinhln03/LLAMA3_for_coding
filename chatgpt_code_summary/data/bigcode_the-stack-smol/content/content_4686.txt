import pyglet


class Resources:
    
    # --- Player Parameters ---
    player_animation_started = False
    player_images = []
    player_animation_time = 1. / 9.
    player_animation_index = 0

    # --- Obstacle Parameters ---
    obstacle_images = []

    # --- Player Methods ---
    
    # loads the images needed for the player animation if they haven't been loaded already
    @staticmethod
    def load_images():
        if len(Resources.player_images) == 0:
            Resources.player_images.append(pyglet.image.load("res/dinosaur_left.png"))
            Resources.player_images.append(pyglet.image.load("res/dinosaur_right.png"))
            Resources.player_images.append(pyglet.image.load("res/dinosaur_normal.png"))

        if len(Resources.obstacle_images) == 0:
            Resources.obstacle_images.append(pyglet.image.load("res/cactus_small.png"))
            Resources.obstacle_images.append(pyglet.image.load("res/cactus_big.png"))
            
        Resources.start_player_animation()

    # starts the player's running animation by scheduling recurring updates to the player's image index
    @staticmethod
    def start_player_animation():
        if not Resources.player_animation_started:
            pyglet.clock.schedule_interval(Resources.trigger_player_update, Resources.player_animation_time)
            Resources.player_animation_started = True

    # updates the player's image index
    @staticmethod
    def trigger_player_update(_):
        Resources.player_animation_index = 1 - Resources.player_animation_index
        
    # returns the current image for the running player
    @staticmethod
    def player_running_image():
        return Resources.player_images[Resources.player_animation_index]

    # returns the image for the jumping player
    @staticmethod
    def player_jumping_image():
        return Resources.player_images[2]

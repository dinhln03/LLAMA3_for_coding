import pygame


class TextSprite(pygame.sprite.Sprite):
    """Subclass of sprite to draw text to the screen"""

    def __init__(self, position, text_lines, font, fg=(0, 0, 0), bg=None,
                 border_width=0, border_color=(0, 0, 0),
                 bold=False, italic=False, underline=False,
                 line_spacing=3, padding=5):
        pygame.sprite.Sprite.__init__(self)

        self.position = position
        self.font = font
        self.fg = fg
        self.bg = bg
        self.border_width = border_width
        self.border_color = border_color
        self.line_spacing = line_spacing
        self.padding = padding
        self.font.set_bold(bold)
        self.font.set_italic(italic)
        self.font.set_underline(underline)

        self.rect = None

        self.image = None

        self.text_lines = text_lines
        self.update()

    def update(self):
        """"""
        # Render all lines of text
        text_images = [self.font.render(t, False, self.fg, self.bg) for t in self.text_lines]

        # Find the largest width line of text
        max_width = max(text_images, key=lambda x: x.get_width()).get_width()
        # Produce an image to hold all of the text strings
        self.image = pygame.Surface(
            (max_width + 2 * (self.border_width + self.padding),
             text_images[0].get_height() * len(text_images) + self.line_spacing * (len(text_images) - 1) + 2 * (
                     self.border_width + self.padding)
             )
        )
        self.image.fill(self.bg)

        if self.border_width > 0:
            pygame.draw.rect(self.image, self.border_color,
                             (0, 0, self.image.get_width(), self.image.get_height()), self.border_width)

        for n, t in enumerate(text_images):
            self.image.blit(t, (self.border_width + self.padding,
                                self.border_width + self.padding + (self.line_spacing + t.get_height()) * n))

        # Store the last rect so if the new one is smaller we can update those bits of the screen too
        last_rect = self.rect
        self.rect = pygame.Rect(self.position[0], self.position[1], self.image.get_width(), self.image.get_height())

        if last_rect is None:
            return self.rect
        else:
            return last_rect.union(self.rect)

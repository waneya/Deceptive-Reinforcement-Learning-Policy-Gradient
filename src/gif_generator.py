
from PIL import Image, ImageDraw, ImageColor, ImageFont


COLOR_MAP = {'.': 'moccasin', 'T': 'green'}


class GifGenerator(object):
    def __init__(self, lmap, start, goal, poss_goal):
        self.lmap = lmap
        self.images = []
        self.scale = 8
        if lmap.width == 100:
            self.scale = 4
        init_img = Image.new('RGB', (lmap.width * self.scale, lmap.height * self.scale), (0, 0, 0))
        for x in range(lmap.width):
            for y in range(lmap.height):
                self.drawPoint(init_img, (x, y), COLOR_MAP[lmap.getCell((x, y))])
        self.drawCross(init_img, start, 'lightblue')
        goals = [goal]
        goals.extend(poss_goal)
        goals = sorted(goals, key=lambda e: (e[1], e[0]))
        for i, g in enumerate(goals):
            self.drawCross(init_img, g, 'red')
            self.drawText(init_img, g, str(i + 1))
        self.drawPoint(init_img, start, 'white')
        # init_img.show()
        self.last_img = init_img
        self.last_step = start
        self.images.append(init_img)

    def drawText(self, img, pos, text):
        if pos[0] + 3 >= self.lmap.width:
            new_pos = (pos[0] * self.scale - 3 * self.scale, pos[1] * self.scale)
        else:
            new_pos = (pos[0] * self.scale + 3 * self.scale, pos[1] * self.scale)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16)
        draw.text(new_pos, text, fill='black', font=font)

    def drawCross(self, img, pos, color):
        for n in range(-2, 3):
            self.drawPoint(img, (pos[0] + n, pos[1] + n), color)
            self.drawPoint(img, (pos[0] + n, pos[1] - n), color)

    def drawPoint(self, img, pos, color):
        draw = ImageDraw.Draw(img)
        color_rgb = ImageColor.getrgb(color)
        draw.rectangle(self.getRec(pos), fill=color_rgb)

    def getRec(self, pos):
        x, y = pos
        return [x * self.scale, y * self.scale, (x + 1) * self.scale - 1, (y + 1) * self.scale - 1]

    def step(self, pos):
        next_img = self.last_img.copy()
        if self.last_step is not None:
            self.drawPoint(next_img, self.last_step, 'blue')
        self.drawPoint(next_img, pos, 'white')
        # next_img.show()
        self.images.append(next_img)
        self.last_step = pos
        self.last_img = next_img

    def finalize(self, out_path):
        # quintile points
        l = len(self.images)
        quintiles = [int(l * 0.25), int(l * 0.5), int(l * 0.75)]
        for i, q in enumerate(quintiles):
            file_name = '{:s}_{:d}.gif'.format(out_path, i)
            self.images[0].save(file_name, save_all=True, append_images=self.images[1:q],
                                optimize=False, duration=int(100/(i+1)), loop=0)

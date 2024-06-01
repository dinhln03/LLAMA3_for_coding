"""
本文件用以练习 manim 的各种常用对象
    SVGMobject
    ImageMobject
    TextMobject
    TexMobeject
    Text
参考资料： https://www.bilibili.com/video/BV1CC4y1H7kp
XiaoCY 2020-11-27
"""

#%% 初始化
from manimlib.imports import *

"""
素材文件夹介绍
    在 manim 中使用各种素材时可以使用绝对路径声明素材。
    为了简单，可以创建 assets 文件夹并放置在 manim 路径下。
    如此做，使用素材时可以不加路径。

    assets/raster_images/ 放置 png 等格式的位图
    assets/svg_images/ 放置 svg 格式的矢量图
    assets/sounds/ 一般不用，也可以不创建
"""

#%% SVGMobject
"""
在 manim 中使用 SVG 图片可以直接使用 SVGMobject，
传入的第一个参数是指向 SVG 文件的字符串，
关键字参数包括 VMobject 的共有属性，有
    填充样式
        填充颜色 fill_color 或 color
        不透明度 fill_opacity
    线条样式
        线条颜色 stroke_color 或 color
        线条宽度 stroke_width
        线条不透明度 stroke_opacity
    背景线条样式
        背景线条颜色 background_stroke_color 或 color
        背景线条宽度 background_stroke_width
        背景线条不透明度 background_stroke_opacity
    光泽样式
        光泽尺度 sheen_factor
        光泽方向 sheen_direction
"""

class try_SVGMob(Scene):                # 使用 class 创建一个场景，名字可自定义
    def construct(self):                # 这里 class 和 def 暂且当成是固定的套路吧
        # 构造 SVGMobject --- 添加 SVG 图片
        mob = SVGMobject(
            "coin.svg",
            color = BLUE,               # manim 内置部分颜色，参见 https://manim.ml/constants.html#id7
            stroke_width = 1.00
        )

        # SVGMobject 可以使用 VMobject 的所有动画
        # 动画暂时不在本练习中具体讲解，这里仅作示意
        self.play(FadeInFromLarge(mob))
        self.wait(2)

#%% ImageMobject
"""
与 SVGMobject 相像，插入位图时可使用 ImageMobject，
传入的第一个参数是字符串表示的位图路径，
关键字参数仅有以下部分
    图片高度 height （默认为2）
    是否反色 invert （默认 False）
"""

class try_ImageMob(Scene):
    def construct(self):
        # 添加位图
        mob = ImageMobject(
            'smirk.png',
            height = 3
        )

        # 由于 ImageMobject 不是 VMobject 的子类，很多动画无法使用
        # 但是这里依然不对动画做深入讨论
        self.play(FadeInFromLarge(mob))
        self.wait(2)

#%% TextMobject
"""
TextMobject 会将字符串作为 LaTeX 的普通字符进行编译
传入的第一个参数为需要添加的字符串，其可以使用 LaTeX 表达式
由于 LaTeX 表达式中常含有反斜线，构造字符串时需要采用双反斜线
或在字符串前添加 r 以消除歧义
TextMobject 是 VMobject，其他属性同 SVGMobject

一个 TextMobject 中也可以传入多个字符串，会单独编译但连在一起显示
这时可以利用索引来访问各个字符串

其他可选参数
    arg_separator 传入多个字符串时，设置字符串之间的字符，默认为空格
    tex_to_color_map 为一个字典，根据键值自动拆分字符串进行上色
"""

class try_TextMob(Scene):
    def construct(self):
        # 定义一个字符串（数组），并通过下标进行访问
        text = TextMobject(
            "早安， \\TeX！",
            r"你好，\LaTeX！",
            tex_to_color_map = {"\\LaTeX": RED_B}
        )

        self.play(Write(text[0]))
        self.wait(0.5)

        # 注意这里用到 text[2] 和 text[3] 
        # 原本不存在，但 tex_to_color_map 为了上色将其自动拆分了
        # text[0] = r"早安， \TeX！"
        # text[1] = "你好，"
        # text[2] = r"\LaTeX"
        # text[3] = "！"
        self.play(Transform(text[0],text[1:4]))     
        self.wait(1)

#%% TexMobject
"""
TexMobject 实际上提供了 align* 的数学环境，用于辨析 LaTeX 数学公式
其使用方法和 TextMobject 一样

关于数学公式的 LaTeX 代码可以使用妈叔的在线编辑器
（友情推荐） https://www.latexlive.com/
"""

class try_TexMob(Scene):
    def construct(self):
        text = TextMobject('欧拉公式')         # 这是文字 
        tex = TexMobject(                    # 这是公式
            r'\mathrm{e}^{\mathrm{i}\pi} + 1 = 0',
            color = GREEN
        )

        self.play(FadeIn(text))
        self.wait(0.5)
        self.play(ReplacementTransform(text,tex))
        self.wait(1)

#%% Text
"""
TextMobject 的文字是经过 LaTeX 编译的，
若仅使用文字（不用 \LaTeX 之类的特殊符号），可以使用 Text

传入的第一个参数为文字字符串，可选参数包括
    颜色 color
    字体 font
    颜色 t2c （字典）
"""

class try_Text(Scene):
    def construct(self):
        text = Text(
            "Hello World!",
            font = "Adobe Heiti Std",
            t2c = {
                "H": BLUE,
                "W": RED
            }
        )

        self.play(Write(text))
        self.wait(1)
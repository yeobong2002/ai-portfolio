# korean_text.py
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "C:/Windows/Fonts/batang.ttc",
    "C:/Windows/Fonts/NanumGothic.ttf",
]

def _load_font(size=20):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def put_korean_text(frame, text, position,
                    font_size=22,
                    color=(255, 255, 255),
                    background=None):
    img_pil = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _load_font(font_size)
    x, y   = position
    r, g, b = color[2], color[1], color[0]

    if background is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        pad  = 4
        br, bg, bb = background[2], background[1], background[0]
        draw.rectangle(
            [bbox[0]-pad, bbox[1]-pad,
             bbox[2]+pad, bbox[3]+pad],
            fill=(br, bg, bb)
        )

    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        draw.text((x+dx, y+dy), text,
                  font=font, fill=(0, 0, 0))

    draw.text((x, y), text,
              font=font, fill=(r, g, b))

    return cv2.cvtColor(
        np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_status_bar(frame, texts_colors,
                   start_y=10,
                   font_size=22,
                   line_height=35):
    result = frame.copy()
    y      = start_y
    for text, color in texts_colors:
        result = put_korean_text(
            result, text, (10, y),
            font_size  = font_size,
            color      = color,
            background = (0, 0, 0)
        )
        y += line_height
    return result
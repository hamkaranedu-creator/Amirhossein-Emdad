# آماده‌سازی عکس ارائه‌دهنده برای اسلاید «معرفی ارائه‌دهنده»
# Presenter portrait for the intro slide: same framing and same background for every presenter.
#
#   python3 portrait.py photo.png img/name.webp
#
# ورودی: عکسی که پس‌زمینه‌اش جدا شده است (نیاز: Python 3، numpy و Pillow)
#   - مثل عکس‌هایی که طراح رویداد می‌دهد: آدم روی یک لایهٔ رنگی نیمه‌شفاف، یا
#   - عکس بریده‌شده با پس‌زمینهٔ کاملاً شفاف (PNG یا WebP).
# خروجی: WebP شفاف ۸۸۰×۱۰۰۰ (نسبت شش‌ضلعی) با قاب یکسان برای همه:
#   بالای مو در ۱۳٪ ارتفاع، چانه و گردن در ۴۷٪، صورت در وسط.
#   زمینهٔ بنفش یکسان را خود اسلاید پشت عکس می‌گذارد.
# عکس معمولی با پس‌زمینهٔ کامل را این ابزار جدا نمی‌کند: نسخهٔ بریده‌شده را از طراح بگیرید،
# یا همان عکس را در img/ بگذارید و در slide-data برایش "pos" تعیین کنید (مثلاً "50% 30%").
import sys
import numpy as np
from PIL import Image

W, H = 880, 1000
TOP, NECK = 0.13, 0.47

if len(sys.argv) < 3:
    sys.exit('usage: python3 portrait.py photo.png img/name.webp')
src, out = sys.argv[1], sys.argv[2]
im = np.asarray(Image.open(src).convert('RGBA')).astype(np.float64) / 255.0
rgb, a = im[..., :3], im[..., 3]
if a.min() > 0.98:
    sys.exit('این عکس پس‌زمینهٔ جداشده ندارد (کاملاً مات است). نسخهٔ بریده‌شده را از طراح بگیرید، '
             'یا عکس را مستقیم در img/ بگذارید و در slide-data برایش "pos" تعیین کنید.')

# ۱) لایهٔ پس‌زمینه: شفافیت هر ردیف در لبه‌های چپ و راست تصویر رنگی
cols = np.nonzero(a.max(0) > 0)[0]
x0, x1 = cols.min(), cols.max()
k = max(4, (x1 - x0) // 60)
edge = np.concatenate([a[:, x0 + 2:x0 + 2 + k], a[:, x1 - 1 - k:x1 - 1]], axis=1)
a_bg = np.median(edge, axis=1)[:, None]
if not (0.15 < a_bg.max() < 0.95):      # پس‌زمینهٔ کاملاً شفاف (عکس بریده‌شدهٔ معمولی)
    a_bg = np.zeros_like(a_bg)
near_bg = (np.abs(a - a_bg) < 0.01) & (a > 0.15)
bg_rgb = np.median(rgb[near_bg], axis=0) if near_bg.any() else np.zeros(3)

# ۲) سهم آدم در هر پیکسل و رنگ بدون اثر لایهٔ پس‌زمینه
ap = np.clip((a - a_bg) / np.maximum(1e-6, 1 - a_bg), 0, 1)
ap[:, :x0 + 1] = 0; ap[:, x1:] = 0
premul = rgb * a[..., None] - bg_rgb[None, None, :] * (a_bg * (1 - ap))[..., None]
person = np.clip(np.where(ap[..., None] > 0.02, premul / np.maximum(ap[..., None], 1e-6), rgb), 0, 1)
ap[ap < 0.03] = 0

# ۳) اندازه‌ها از روی شکل آدم: بالای مو، مرکز سر، گردن (باریک‌ترین ردیف زیر سر)
solid = ap > 0.5
rows = np.nonzero(solid.sum(1) > 6)[0]
if rows.size == 0:
    sys.exit('آدمی در عکس پیدا نشد.')
top, bottom = rows.min(), rows.max(); h = bottom - top
width = solid.sum(1).astype(float)
band = ap[top:top + max(10, int(0.2 * h))]
cx = float((band * np.arange(ap.shape[1])).sum() / band.sum())
search = np.arange(top + int(0.18 * h), min(bottom, top + int(0.6 * h)))
neck = int(search[np.argmin(width[search])])
scale = (NECK - TOP) * H / (neck - top)

# ۴) قرار دادن روی بوم استاندارد
img = Image.fromarray((np.dstack([person, ap]) * 255).round().astype(np.uint8), 'RGBA')
nw, nh = round(img.width * scale), round(img.height * scale)
img = np.asarray(img.resize((nw, nh), Image.LANCZOS))
ox, oy = round(W / 2 - cx * scale), round(TOP * H - top * scale)
c = np.zeros((H, W, 4), np.uint8)
sx0, sy0, dx0, dy0 = max(0, -ox), max(0, -oy), max(0, ox), max(0, oy)
w, hh = min(W - dx0, nw - sx0), min(H - dy0, nh - sy0)
if w > 0 and hh > 0:
    c[dy0:dy0 + hh, dx0:dx0 + w] = img[sy0:sy0 + hh, sx0:sx0 + w]

# ۵) اگر عکس بالاتر از پایین بوم تمام شده، آخرین ردیف بدن را تا پایین ادامه بده
last = oy + round(bottom * scale) - 2
if 0 <= last < H - 1:
    c[last + 1:] = c[last:last + 1]
Image.fromarray(c, 'RGBA').save(out, 'WEBP', quality=88, method=6)
print(f'{out}: 880x1000 (scale {scale:.2f})')

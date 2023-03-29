# import os
#
# font_dirs = ['/Library/Fonts', '/System/Library/Fonts']
#
# for font_dir in font_dirs:
#     print(f'Fonts in {font_dir}:')
#     for font_file in os.listdir(font_dir):
#         if font_file.endswith('.ttf') or font_file.endswith('.otf'):
#             print(f'  {font_file}')

import matplotlib.font_manager as fm

fonts = fm.findSystemFonts()
for font in fonts:
    prop = fm.FontProperties(fname=font)
    print(f'{prop.get_name()} - {prop.get_family()} - {prop.get_style()}')
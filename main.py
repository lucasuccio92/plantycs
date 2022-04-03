from cv2 import imread
from math import atan as arctan
from libxmp.utils import file_to_dict
import find_kornrows_fast as oriente

image = imread()
img_file = 'kornrows/DJI_20220211072437_0080_Z.JPG'

# read image exif data?
#altitude = 

# read image xmp data
def get_alt(filename):
    rel_alt = None
    xmp = file_to_dict(filename)
    for i in xmp['http://www.dji.com/drone-dji/1.0/']:
        if 'RelativeAltitude' in i[0]:
            rel_alt = float(i[1])
    return rel_alt

rel_altitude = get_alt(img_file)

# get orientation of corn rows
orientation = oriente(image)

# get all point to point lines

# filter by orientation and length

# GROUND SAMPLING DISTANCE - GSD - mm/px
focal_length = 220
ar = 4 / 3
fov = 2 * arctan(35 / focal_length * 2)

# PLANT SPACING

# VARIATION COEFFICIENT
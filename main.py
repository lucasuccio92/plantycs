from cv2 import imread
from math import atan as arctan
from libxmp.utils import file_to_dict
import find_kornrows as get_orientation
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from math import tan







def get_all_lines(plants):
    # get all point to point lines
    points = [Point(plant) for plant in plants]
    all_lines = []
    for p0 in points:
        for p1 in points:
            bearing = tan((p1.x - p0.x) / (p1.y - p0.y))
            # filter by orientation
            if bearing > (0.95 * orientation) or bearing < (1.05 * orientation):
                all_lines.append(LineString(p0, p1))

    print(len(all_lines))
    return all_lines

def main():
    image = imread()
    img_file = 'kornrows/DJI_20220211072437_0080_Z.JPG'
    # get orientation of corn rows
    orientation, plants = get_orientation(img_file)

# PLANT SPACING

# VARIATION COEFFICIENT
# CV = StD / avg_spacing * 100


if __name__ == '__main__':
    main()

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 


'''
# read image xmp data
def get_alt(filename):
    rel_alt = None
    xmp = file_to_dict(filename)
    for i in xmp['http://www.dji.com/drone-dji/1.0/']:
        if 'RelativeAltitude' in i[0]:
            rel_alt = float(i[1])
    return rel_alt

rel_altitude = get_alt(img_file)

# read labels file here:
#   send points to oriente
#   send points to all_lines

# GROUND SAMPLING DISTANCE - GSD - mm/px
focal_length = 220
ar = 4 / 3
fov = 2 * arctan(35 / focal_length * 2)
'''

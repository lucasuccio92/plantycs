import os
import find_kornrows
import json
import pandas as pd


def main(folder):
    geojson = []
    samples = os.listdir(folder)
    os.chdir(folder)
    samples = [s for s in samples if s.lower().endswith('.jpg')]
    for sample in samples:
        print('Processing', sample)
        result = find_kornrows.analyse_field(sample)
        if result is not None:
            geojson.append(result)
        print('\n')
    with open('output.json', 'w') as output:
        json.dump(geojson, output, indent=2)
    geojson.to_csv('output.csv')

if __name__ == '__main__':
    folder = 'kornrows/geojson_test/'
    main(folder)

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

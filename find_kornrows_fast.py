'''
this algorithm is supposed to find the orientation of crop rows
given an input image and yolov5-labelled objects

The Procedure:
1. Read image exif data to find relative (above-ground) altitude
2. Read image labels.txt and plot centroids of objects
3. For each image, find the angle of lines with the lowest offset from all plants (objects)

'''
import os
import math
import argparse
import matplotlib.pyplot as plt
import PIL.Image as img
import numpy as np

VISUALISE = False
HIST_BINS = 40

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    # For clockwise rotation, negate the angle
    #angle *= -1

    ox, oy = origin
    px, py = point

    sine = math.sin(angle)
    cosine = math.cos(angle)

    qx = ox + cosine * (px - ox) - sine * (py - oy)
    qy = oy + sine * (px - ox) + cosine * (py - oy)
    return qx, qy

def get_centroids(lbls, dims):
    centroids = []
    with open(lbls, 'r') as lbls_txt:
        for line in lbls_txt:
            obj_class, x, y, w, h = line.split(' ')
            centroids.append([float(x) * dims[0], float(y) * dims[1]])
    return centroids

def scrape(start, end, step, dims, centroids):
    p1 = np.array(dims) / 2
    # p2 starts at top of image
    img_top = p1 * np.array([1, 0])
    orientation = [None]
    max_clust = [0]
    hist_data = [None]
    for angle in range(start,end,step):  # ADD TQDM LATER
        # plot a line of orientation "angle"
        print(f'Scraping {angle:03d}°:\t',end='')
        # line drawn between two points, p1 (img_centre) and p2 (offset by angle):
        p2 = rotate(p1, img_top, math.radians(angle))
        distances = []
        for plant in centroids:
            # the distance from P3 perpendicular to a line drawn between P1 and P2
            p3 = np.array(plant)
            dist = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
            distances.append(dist)
        
        num_bins = HIST_BINS
        n, edges = np.histogram(distances, bins=num_bins)
        empties = np.count_nonzero(n==0)
        nonzero_mean = np.mean(n[np.nonzero(n)])
        discreteness = int(empties / num_bins * nonzero_mean * 100)
        if discreteness > max_clust[0]:
            max_clust = [discreteness]
            orientation = [angle]
            hist_data = [distances]
        elif discreteness == max_clust[0]:
            max_clust.append(discreteness)
            orientation.append(angle)
            hist_data.append(distances)
        print(f'Empty bins: {empties}/{num_bins}\tAvg bin: {round(nonzero_mean,2)}\tDiscreteness: {discreteness}')
    peak_width = len(max_clust)
    if peak_width > 1:
        max_clust = np.mean(max_clust)
        orientation = np.mean(orientation)
        hist_data = np.mean(hist_data, axis=0)
    else:
        max_clust = max_clust[0]
        orientation = orientation[0]
        hist_data = hist_data[0]

    print(f'Max discreteness ({max_clust}) at {orientation}°')
    return orientation, hist_data

def find_orientation(centroids, img):
    dims = img.size

    # course scrape through 180 degrees in 10 degree steps
    course, hist = scrape(0, 180, 10, dims, centroids)

    # fine scrape around course result in 1 degree steps
    fine, hist = scrape(course-5, course+5 ,1 , dims, centroids)

    if VISUALISE:
        plot_orientation(img, fine, centroids, hist)
        
    return fine, centroids

def plot_orientation(img, angle, centroids, hist):
    # plot scrape results
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(14,6))
    # plot orientation angle
    axs[0].set_title('Angle')
    axs[0].imshow(img)
    # plot orientation line
    centre = np.array(img.size) / 2 # img_centre
    img_top = centre * np.array([1, 0])
    img_bot = centre * np.array([1, 2])
    p1 = rotate(centre, img_top, math.radians(angle))
    p2 = rotate(centre, img_bot, math.radians(angle))
    axs[0].plot([p1[0],p2[0]],[p1[1],p2[1]], color="yellow")
    for plant in centroids:
        axs[0].plot(plant[0], plant[1], marker='o', color="blue")
    # plot distances
    axs[1].set_title('Distances')
    axs[1].hist(hist, bins=HIST_BINS)
    filename = f'kornrows/{angle}_plot'
    i = 0
    while os.path.isfile(f'{filename}_{i}'):
        i += 1
    plt.savefig(f'{filename}_{i}.png', dpi=300)

def main(field):
    #parser = argparse.ArgumentParser(description="Kornrow finder")
    #parser.add_argument('--image_name', required=True, help="Name of the input image (with or without extension)")
    #args = parser.parse_args()
    #field = args.image_name
    if field.lower().endswith('.jpg'):
        field = field[:-4]
    image = field + '.JPG'
    lables = field + '.txt'
    rgb = img.open(image)
    #img_x, img_y = rgb.size
    plants = get_centroids(lables, rgb.size)
    orientation = find_orientation(plants, rgb)
    return orientation, plants
    


    '''
    plt.imshow(rgb)
    for plant in plants:
        plt.plot(plant[0], plant[1], marker='o', color="blue")
    plt.show()
    '''




'''
labels.txt in the form:
0 0.063657 0.131173 0.062500 0.052469
0 0.635995 0.628086 0.075231 0.058642
0 0.968750 0.055041 0.062500 0.060700
which is: 
class x y width height

'''
import cv2
import numpy as np
import shapely.geometry as sg
import pyclipper
from PIL import Image
from skimage.draw import polygon


def masks_to_contours(masks, se_size=10, approx_length=None):
    mask_contours = list()
    for mask in masks:
        mask_contours.append(mask_to_contours(mask, se_size, approx_length))
    return mask_contours


def mask_to_contours(mask, se_size=10, approx_length=None):
    # Mask is a PIL-image with 0's and 1's of size (width, height)
    # Or tensor with 0's and 1's of shape (1, height, width)
    if isinstance(mask, Image.Image):
        width, height = mask.size
    else:
        if len(mask.shape) > 2:
            mask = mask.squeeze(0)
        height, width = mask.shape
    mask_np = np.uint8(255 * np.array(mask))
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask_np = cv2.dilate(mask_np, SE3)
    mask_np = cv2.erode(mask_np, SE)
    mask_np = cv2.dilate(mask_np, SE)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, SE)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not approx_length:
        approx_length = 0.005 * width
    contours = [cv2.approxPolyDP(c, approx_length, False) for c in contours]
    ok_contours = []
    for contour in contours:
        if contour.shape[0] >= 3:  # Make sure it's not a line
            ok_contours.append(contour.squeeze(1))
        else:
            print("Contour is a point or line, skipping it")
    return ok_contours


def contours_to_polygons(contours, make_square=False, rotated=False, shrink_factor=-1, force_shrink_factor=False,
                         shrink_cap=2.5):
    padded_polygons = []
    for contour in contours:
        contour = np.array(contour)
        if contour.shape[0] < 3:
            print("Contour has bad shape, skipping it")
            continue
        padded_polygon = contour_to_polygon(contour,
                                            make_square=make_square,
                                            rotated=rotated,
                                            shrink_factor=shrink_factor,
                                            force_shrink_factor=force_shrink_factor,
                                            shrink_cap=shrink_cap)

        if not padded_polygon.is_valid:
            print("Found invalid polygon. Attempting fix")
            padded_polygon = padded_polygon.buffer(0)
            if padded_polygon.is_valid:
                print("Fix worked")
                padded_polygons.append(padded_polygon)
            else:
                print("Fix didn't work, skipping polygon")
        else:
            padded_polygons.append(padded_polygon)
    return padded_polygons


def contour_to_polygon(contour, make_square=False, rotated=False, shrink_factor=-1, force_shrink_factor=False,
                       shrink_cap=2.5):
    poly_np = np.zeros((contour.shape[0], 2))
    for i, point in enumerate(contour):
        x = point[0]
        y = point[1]
        poly_np[i, 0] = x
        poly_np[i, 1] = y
    poly_shape = sg.Polygon(poly_np)
    if poly_shape.area < 500:
        pass

    if force_shrink_factor:
        shrink_ratio = shrink_factor
    elif shrink_factor == -1:
        shrink_ratio = 20 / np.log(poly_shape.area)
    else:
        shrink_ratio = shrink_factor / np.log(poly_shape.area)
    if shrink_ratio > shrink_cap:
        print("Capping", shrink_ratio, "at", shrink_cap)
        shrink_ratio = shrink_cap

    distance = calc_distance(poly_shape, shrink_ratio)
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(poly_np, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon_np = np.array(padding.Execute(-distance)[0])
    padded_polygon = sg.Polygon(list(zip(padded_polygon_np[:, 0], padded_polygon_np[:, 1])))
    if make_square:
        if rotated:
            padded_polygon = padded_polygon.minimum_rotated_rectangle
        else:
            padded_polygon = padded_polygon.envelope
    return padded_polygon


def polygons_to_points(polygons):
    points = []
    for polygon in polygons:
        points.append(polygon_to_points(polygon))
    return points


def polygon_to_points(polygon):
    p = list(polygon.exterior.xy)
    p = [(int(x), int(y)) for x,y in zip(p[0], p[1])]
    return p


def polygons_to_crops(polygons, image):
    # polygons = [Shaply polygon, ...]
    # cropped_images = [{'image', 'coords'}, ...]
    cropped_images = []
    for polygon in polygons:
        cropped_images.append(polygon_to_crop(polygon, image))
    return cropped_images


def polygon_to_crop(polygon, image):
    p_np = np.array(polygon.exterior.coords).astype(int)
    min_w, max_w = np.min(p_np[:, 0]), np.max(p_np[:, 0])
    min_h, max_h = np.min(p_np[:, 1]), np.max(p_np[:, 1])

    min_w = max(min_w, 0)
    max_w = min(max_w, image.size[0])
    min_h = max(min_h, 0)
    max_h = min(max_h, image.size[1])

    mask_w = max_w - min_w
    mask_h = max_h - min_h
    small_mask = np.zeros((mask_h, mask_w), dtype="bool")
    rr, cc = polygon(p_np[:, 0] - min_w, p_np[:, 1] - min_h, (mask_w, mask_h))
    small_mask[cc, rr] = 1
    small_mask = Image.fromarray(small_mask)
    small_image = image.crop((min_w, min_h, min_w + mask_w, min_h + mask_h))
    cropped_image = Image.new('RGB', small_image.size, (255, 255, 255))
    cropped_image.paste(small_image, mask=small_mask)
    return {'image': cropped_image, 'coords': [min_w, min_h, max_w, max_h]}


def polygons_to_midlines(polygons):
    midlines = []
    for polygon in polygons:
        midlines.append(polygon_to_midline(polygon))
    return midlines


def polygon_to_midline(polygon):
    if not polygon.is_valid:
        print("Found invalid polygon. Attempting fix")
        polygon = polygon.buffer(0)
        assert polygon.is_valid and type(polygon) == sg.Polygon, "Fix didn't work. Crash time"
    coords = polygon.envelope.exterior.xy  # coordinates of the polygon bounding box
    min_x = int(min(coords[0]))
    max_x = int(max(coords[0]))
    min_y = int(min(coords[1]))
    max_y = int(max(coords[1]))
    pixel_per_dot = 50
    if (max_x - min_x) // pixel_per_dot > 1:
        dx = int((max_x - min_x) / ((max_x - min_x) // pixel_per_dot))
    else:
        dx = (max_x - min_x) // 2
    points = []
    p0 = (min_x, min_y)
    p1 = (min_x, max_y)
    points.append((p0, p1))
    for x_int in range(min_x + dx, max_x - dx + 1, dx):
        points.append(((x_int, min_y), (x_int, max_y)))
    p0 = (max_x, min_y)
    p1 = (max_x, max_y)
    points.append((p0, p1))
    midline_coords = []
    for p in points:
        p0 = p[0]
        p1 = p[1]
        line = sg.LineString([p0, p1])
        line_intersect = polygon.intersection(line)
        if not (type(line_intersect) == sg.Point or type(line_intersect) == sg.LineString):
            print("Midline intersection has wrong type. Skipping it")
            continue
        line_coords = line_intersect.xy
        if len(line_coords[1]) == 1:
            midline_coords.append((int(line_coords[0][0]), int(line_coords[1][0])))
        else:
            mid_y = int(line_coords[1][0] + (line_coords[1][1] - line_coords[1][0]) // 2)
            midline_coords.append((p0[0], mid_y))
    if len(midline_coords) > 3:
        midline_coords[0] = (min_x, midline_coords[1][1])
        midline_coords[-1] = (max_x, midline_coords[-2][1])
    return midline_coords


def points_to_mask(polygons, image_size, shrink_ratio=0.4, area_thresh=float('inf')):
    # image_size = (width, height)
    # Polygons is a list of of elements [p1, p2, p3, ...] where p_i = (x,y)
    poly_nps = []
    for poly in polygons:
        poly_np = np.zeros((len(poly), 2))
        for i, (x, y) in enumerate(poly):
            poly_np[i, 0] = x
            poly_np[i, 1] = y
        poly_nps.append(poly_np)
    mask = np.zeros((image_size[1], image_size[0]), dtype="bool")
    for poly_np in poly_nps:
        poly_shape = sg.Polygon(poly_np)
        distance = calc_distance(poly_shape, shrink_ratio)
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(poly_np, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(-distance)[0])

        padded_shapely_poly = sg.Polygon(padded_polygon)
        if padded_shapely_poly.area < area_thresh:
            rr, cc = polygon(padded_polygon[:, 0], padded_polygon[:, 1], image_size)
            mask[cc, rr] = 1
    return Image.fromarray(mask)


def get_overlapping_polygons(mask, polygons):
    overlap_p = []
    for poly in polygons:
        if mask.intersects(poly):
            overlap_p.append(poly)
    return overlap_p


def calc_distance(poly_shape, shrink_ratio):
    distance = (poly_shape.area * (1 - np.power(shrink_ratio, 2)) / poly_shape.length) * min(
        np.power(poly_shape.area / 2000, 2), 1)
    return distance

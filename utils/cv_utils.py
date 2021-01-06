import cv2
import numpy as np


def draw_text_with_backgroud(
    img, text, x, y, font_scale, thickness=1,
    font=cv2.FONT_HERSHEY_SIMPLEX, background=(0, 0, 0),
    foreground=(255, 255, 255), box_coords_1=(-5, 5), box_coords_2=(5, -5)
):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((x+box_coords_1[0], y+box_coords_1[1]), (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]))
    cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, fontScale=font_scale, color=foreground, thickness=thickness)


def draw_tracked_objects(frame, tracked_objs, tracker_type):
    for obj in tracked_objs.values():
        obj_rect = obj.rect
        obj_ctr = (
            obj.centroid
            if tracker_type == "centroid" else
            obj.state[:2]
        )

        cv2.circle(frame, obj_ctr, radius=3, color=(0, 0, 0), thickness=-1)

        if obj.direction:
            base_color = [0, 255, 0]
        else:
            base_color = [0, 0, 255]
            draw_text_with_backgroud(frame, "Wrong Direction", obj_ctr[0]-10, obj_ctr[1]-10, font_scale=0.4, thickness=1, foreground=(0, 0, 255))

        cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)

        prev_point = None
        for pt, perc, size in zip(obj.path, np.linspace(0.25, 0.75, len(obj.path)), [2]*10 + [3]*15 + [4]*15):
            if not prev_point is None:
                color = tuple(np.array(base_color)*(1-perc))
                cv2.line(frame, (prev_point[0], prev_point[1]), (pt[0], pt[1]), color, thickness=size, lineType=8)
            prev_point = pt

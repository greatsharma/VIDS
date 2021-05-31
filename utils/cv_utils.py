import cv2
import math
import numpy as np


CLASS_COLOR = {
    "pedestrian": (255, 0, 255),
    "cattles": (255, 255, 0),
    "tw": (0,140,255),
    "auto": (100, 0, 100),
    "lmv": (0, 255, 0),
    "hmv": (255, 0, 0),
}


def draw_text_with_backgroud(
    img,
    text,
    x,
    y,
    font_scale,
    thickness=1,
    font=cv2.FONT_HERSHEY_COMPLEX,
    background=(0,0,0),
    foreground=(255,255,255),
    box_coords_1=(-5, 5),
    box_coords_2=(5, -5),
):
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]

    box_coords = (
        (x + box_coords_1[0], y + box_coords_1[1]),
        (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]),
    )

    if background is not None:
        cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        fontScale=font_scale,
        color=foreground,
        thickness=thickness,
    )


def draw_tracked_objects(self, frame, tracked_objs):
    global CLASS_COLOR

    to_deregister = []

    for obj in tracked_objs.values():
        obj_rect = obj.rect

        obj_centroid = (obj_rect[0] + obj_rect[2]) // 2, (obj_rect[1] + obj_rect[3]) // 2

        obj_bottom = (
            obj.obj_bottom
            if self.tracker_type == "centroid"
            else (round(obj.state[0]), round(obj.state[2]))
        )

        (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["deregistering_line_rightdirection"]
        position1 = (obj_bottom[0] - Ax) * (By - Ay) - (obj_bottom[1] - Ay) * (Bx - Ax)

        (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["deregistering_line_wrongdirection"]
        position2 = (obj_bottom[0] - Ax) * (By - Ay) - (obj_bottom[1] - Ay) * (Bx - Ax)

        if (
            (
                obj.absent_count > self.max_absent // 2
                and (obj.state[1], obj.state[3]) == (0, 0)
            )
            or (obj.lane in ["1", "2", "5"] and obj.direction in ["right", "parked"] and position1 > 0)
            or (obj.lane in ["3", "4", "6"] and obj.direction in ["right", "parked"] and position1 < 0)
            or (obj.lane in ["1", "2", "5"] and obj.direction == "wrong" and position2 < 0)
            or (obj.lane in ["3", "4", "6"] and obj.direction == "wrong" and position2 > 0)
            or (
                obj.absent_count > 2
                and obj.continous_presence_count < self.min_continous_presence
            )
        ):
            to_deregister.append(obj.objid)
            continue

        base_color = CLASS_COLOR[obj.obj_class[0]]

        if obj.absent_count == 0:
            x, y = obj_rect[0] + 5, obj_rect[1] + 15
            if obj.direction != "wrong":
                cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)
            else:
                base_color = [0, 0, 255]
                cv2.rectangle(frame, obj_rect[:2], obj_rect[2:], base_color, 2)
        else:
            x, y = obj_bottom[0] - 10, obj_bottom[1]
            if obj.direction == "wrong":
                base_color = [0, 0, 255]

        if obj.instspeed_list[-1] is not None and obj.direction == "right" and obj.lane not in ["5", "6"] and obj.absent_count == 0:
            self.speed_detector(obj, self.frame_count)

        to_write = obj.obj_class[0]

        if obj.instspeed_list[-1] not in [0, None]:
            to_write += ", " + str(obj.instspeed_list[-1]) + " kmph"

        # if obj.avgspeed:
        #     to_write += ", " + str(obj.avgspeed) + " kmph"

        if obj.direction == "parked":
            to_write = "parked"
        elif obj.direction == "wrong":
            to_write = "wrong-way"

        path_length = len(obj.path)
        condition = path_length < 20
        
        if not condition:
            (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["mid_ref"][0]
            position1 = (obj_bottom[0] - Ax) * (By - Ay) - (obj_bottom[1] - Ay) * (Bx - Ax)
            if position1 < 0:
                condition = True

        if condition:
            # to_write = str(obj.objid) + ", " + to_write
            draw_text_with_backgroud(
                frame,
                to_write,
                x,
                y,
                font_scale=0.35,
                thickness=1,
                box_coords_1=(-4, 4),
                box_coords_2=(6, -6),
            )

        if path_length <= self.max_track_pts:
            path = obj.path
        else:
            path = obj.path[path_length - self.max_track_pts :]
            path_length = len(path)

        prev_point = None
        for pt, perc, size in zip(path, np.linspace(0.25, 0.6, path_length), [1]*10 + [2]*15 + [3]*15):
            if not prev_point is None:
                color = tuple(np.array(base_color)*(1-perc))
                cv2.line(frame, prev_point, pt, color, thickness=size, lineType=8)
            prev_point = pt

        if self.mode == "debug":
            centre = obj.eos.centre
            semi_majoraxis = obj.eos.semi_majoraxis
            semi_minoraxis = obj.eos.semi_minoraxis
            angle = obj.eos.angle
            cv2.circle(frame, centre, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(frame, obj_bottom, radius=3, color=(0, 0, 255), thickness=-1)
            cv2.ellipse(
                frame,
                center=centre,
                axes=(semi_majoraxis, semi_minoraxis),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=base_color,
                thickness=1,
            )

    for obj_id in to_deregister:
        self.tracker._deregister_object(obj_id)

import cv2
import math


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
    font=cv2.FONT_HERSHEY_SIMPLEX,
    background=(0,0,0),
    foreground=(255,255,255),
    box_coords_1=(-5, 5),
    box_coords_2=(5, -5),
):
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=thickness
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


def checkpoint(h, k, x, y, a, b, angle):
    angle = math.radians(angle)

    cosa = math.cos(angle)
    sina = math.sin(angle)

    n1 = math.pow(cosa * (x - h) + sina * (y - k), 2)
    n2 = math.pow(sina * (x - h) - cosa * (y - k), 2)

    d1 = a * a
    d2 = b * b

    return (n1 / d1) + (n2 / d2)


def draw_ellipse(obj, obj_bottom, frame, base_color, path=[], check_point=False):
    centre = tuple(round(c) for c in obj.eos.centre)
    semi_majoraxis = obj.eos.semi_majoraxis
    semi_minoraxis = obj.eos.semi_minoraxis
    angle = obj.eos.angle

    cv2.circle(frame, centre, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(frame, obj_bottom, radius=3, color=(0, 0, 255), thickness=-1)
    cv2.ellipse(frame, center=centre, axes=(semi_majoraxis, semi_minoraxis), angle=angle,
                startAngle=0, endAngle=360, color=base_color, thickness=1)

    if len(path) > 2:
        cv2.arrowedLine(frame, path[-2], path[-1], (0, 0, 0), 2)

    if check_point:
        v = checkpoint(centre[0], centre[1], obj_bottom[0], obj_bottom[1], semi_majoraxis, semi_minoraxis, angle)
        if v > 1:
            print(f"objid: {obj.objid}, v: {v}, out\n\n")
        elif v == 1:
            print(f"objid: {obj.objid}, v: {v}, on\n\n")


def draw_tracked_objects(self, frame, tracked_objs):
    global CLASS_COLOR

    to_deregister = []

    for obj in tracked_objs.values():
        obj_rect = tuple(round(v) for v in obj.rect)

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
                (obj.state[1], obj.state[3]) == (0, 0)
                and obj.absent_count > self.max_absent // 2
            )
            or (obj.direction == "wrong" and obj.absent_count > self.max_absent // 2)
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
        
        (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["mid_ref"][1]
        position1 = (obj_bottom[0] - Ax) * (By - Ay) - (obj_bottom[1] - Ay) * (Bx - Ax)

        condition = False

        if position1 < 0:
            if obj.lane in "125":
                (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["mid_ref"][0]
            else:
                (Ax, Ay), (Bx, By) = self.camera_meta[f"lane{obj.lane}"]["mid_ref"][2]

            position2 = (obj_bottom[0] - Ax) * (By - Ay) - (obj_bottom[1] - Ay) * (Bx - Ax)
            if position2 < 0:
                condition = True

        if condition:
            # to_write = str(obj.objid) + ", " + to_write
            fontscale = 0.33
            if position2 > 0:
                fontscale = 0.27

            draw_text_with_backgroud(
                frame,
                to_write,
                x,
                y,
                font_scale=fontscale,
                thickness=1,
                box_coords_1=(-2, 3),
                box_coords_2=(3, -6),
            )

        mtp = self.max_track_pts
        if position1 > 0:
            mtp = self.max_track_pts // 2
        
        path_length = len(obj.path)

        if path_length <= mtp:
            path = obj.path
        else:
            path = obj.path[path_length - mtp :]
            path_length = len(path)

        prev_point = tuple(round(p) for p in path[0])
        for pt in path[1:]:
            pt = tuple(round(p) for p in pt)
            cv2.line(frame, prev_point, pt, base_color, thickness=2)
            prev_point = pt

        # draw_ellipse(obj, obj_bottom, frame, base_color)
        
    for obj_id in to_deregister:
        self.tracker._deregister_object(obj_id)

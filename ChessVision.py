import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import math
import pytesseract
import os

# Ustawienie zmiennej œrodowiskowej TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Camera settings
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
DISPLAY_SCALE = 0.5

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Global variables
running = True
perspective_matrix = None
warped_size = 0
detection_method = None
manual_corners = []
manual_selection_active = False
detected_lines = None
proposed_corners = None
colored_frame = None
show_colored_fields = None
hough_threshold = None
min_line_length = None
max_line_gap = 10
distance_threshold = None
color_threshold = None
detected_fields = None
current_frame = None
board_labels = None  # Nowa zmienna dla oznaczeñ szachownicy
is_assigned = False
bottom_left_field = None

def validate_natural(P):
    if P == "":
        return True
    return P.isdigit() and int(P) > 0

def select_corners(event, x, y, flags, param):
    global manual_corners, manual_selection_active, perspective_matrix, warped_size
    if event == cv2.EVENT_LBUTTONDOWN and manual_selection_active and len(manual_corners) < 4:
        manual_corners.append((x / DISPLAY_SCALE, y / DISPLAY_SCALE))
        if len(manual_corners) == 4:
            manual_selection_active = False
            perspective_matrix, warped_size = get_perspective_transform(manual_corners)
            instruction_label.config(text="Perspective corrected (orthogonal view). Detect lines or select a method.")

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold.get(), 
                            minLineLength=min_line_length.get(), maxLineGap=max_line_gap)
    return lines

def group_lines_perspective(lines, distance_threshold=20, angle_threshold=20):
    if lines is None:
        return None
    
    grouped_lines = []
    lines_with_angles = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        lines_with_angles.append((x1, y1, x2, y2, angle))
    
    if not lines_with_angles:
        return None
    
    sorted_lines = sorted(lines_with_angles, key=lambda x: x[4])
    current_group = [sorted_lines[0]]
    
    for i in range(1, len(sorted_lines)):
        prev_angle = current_group[-1][4]
        curr_angle = sorted_lines[i][4]
        if abs(curr_angle - prev_angle) < angle_threshold:
            prev_mid_x = (current_group[-1][0] + current_group[-1][2]) / 2
            prev_mid_y = (current_group[-1][1] + current_group[-1][3]) / 2
            curr_mid_x = (sorted_lines[i][0] + sorted_lines[i][2]) / 2
            curr_mid_y = (sorted_lines[i][1] + sorted_lines[i][3]) / 2
            distance = math.sqrt((curr_mid_x - prev_mid_x)**2 + (curr_mid_y - prev_mid_y)**2)
            if distance < distance_threshold:
                current_group.append(sorted_lines[i])
            else:
                grouped_lines.append(average_lines(current_group))
                current_group = [sorted_lines[i]]
        else:
            grouped_lines.append(average_lines(current_group))
            current_group = [sorted_lines[i]]
    
    grouped_lines.append(average_lines(current_group))
    return np.array([[line[:4]] for line in grouped_lines], dtype=np.int32) if grouped_lines else None

def group_lines_orthogonal(lines, distance_threshold, image_size=800, min_segments=2):
    if lines is None:
        return None
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 20:
            horizontal_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 20:
            vertical_lines.append((x1, y1, x2, y2))
    
    grouped_horizontal = []
    if horizontal_lines:
        sorted_h = sorted(horizontal_lines, key=lambda x: (x[1] + x[3]) // 2)
        current_group = [sorted_h[0]]
        for i in range(1, len(sorted_h)):
            prev_y = (current_group[-1][1] + current_group[-1][3]) // 2
            curr_y = (sorted_h[i][1] + sorted_h[i][3]) // 2
            if abs(curr_y - prev_y) < distance_threshold:
                current_group.append(sorted_h[i])
            else:
                if len(current_group) >= min_segments:
                    avg_y = sum((l[1] + l[3]) // 2 for l in current_group) // len(current_group)
                    grouped_horizontal.append((0, avg_y, image_size, avg_y))
                current_group = [sorted_h[i]]
        if len(current_group) >= min_segments:
            avg_y = sum((l[1] + l[3]) // 2 for l in current_group) // len(current_group)
            grouped_horizontal.append((0, avg_y, image_size, avg_y))
    
    grouped_vertical = []
    if vertical_lines:
        sorted_v = sorted(vertical_lines, key=lambda x: (x[0] + x[2]) // 2)
        current_group = [sorted_v[0]]
        for i in range(1, len(sorted_v)):
            prev_x = (current_group[-1][0] + current_group[-1][2]) // 2
            curr_x = (sorted_v[i][0] + sorted_v[i][2]) // 2
            if abs(curr_x - prev_x) < distance_threshold:
                current_group.append(sorted_v[i])
            else:
                if len(current_group) >= min_segments:
                    avg_x = sum((l[0] + l[2]) // 2 for l in current_group) // len(current_group)
                    grouped_vertical.append((avg_x, 0, avg_x, image_size))
                current_group = [sorted_v[i]]
        if len(current_group) >= min_segments:
            avg_x = sum((l[0] + l[2]) // 2 for l in current_group) // len(current_group)
            grouped_vertical.append((avg_x, 0, avg_x, image_size))
    
    grouped_lines = grouped_horizontal + grouped_vertical
    return np.array([[line] for line in grouped_lines], dtype=np.int32) if grouped_lines else None

def average_lines(group):
    if not group:
        return None
    avg_x1 = sum(l[0] for l in group) // len(group)
    avg_y1 = sum(l[1] for l in group) // len(group)
    avg_x2 = sum(l[2] for l in group) // len(group)
    avg_y2 = sum(l[3] for l in group) // len(group)
    avg_angle = sum(l[4] for l in group) // len(group)
    return (avg_x1, avg_y1, avg_x2, avg_y2, avg_angle)

def draw_lines(frame, lines, scale=False):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if scale:
                x1, y1, x2, y2 = int(x1 * DISPLAY_SCALE), int(y1 * DISPLAY_SCALE), int(x2 * DISPLAY_SCALE), int(y2 * DISPLAY_SCALE)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

def detect_chessboard_pattern(frame, lines):
    if lines is not None:
        line_image = np.zeros_like(frame)
        draw_lines(line_image, lines, scale=False)
        gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if not ret:
        return None
    
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    horizontal_lines = sorted(set((line[0][1] + line[0][3]) // 2 for line in detected_lines if abs(line[0][1] - line[0][3]) < 20))
    vertical_lines = sorted(set((line[0][0] + line[0][2]) // 2 for line in detected_lines if abs(line[0][0] - line[0][2]) < 20))

    top_y_7x7 = min(c[0][1] for c in corners)
    bottom_y_7x7 = max(c[0][1] for c in corners)
    left_x_7x7 = min(c[0][0] for c in corners)
    right_x_7x7 = max(c[0][0] for c in corners)

    outer_top_y = max(y for y in horizontal_lines if y < top_y_7x7) if any(y < top_y_7x7 for y in horizontal_lines) else 0
    outer_bottom_y = min(y for y in horizontal_lines if y > bottom_y_7x7) if any(y > bottom_y_7x7 for y in horizontal_lines) else warped_size
    outer_left_x = max(x for x in vertical_lines if x < left_x_7x7) if any(x < left_x_7x7 for x in vertical_lines) else 0
    outer_right_x = min(x for x in vertical_lines if x > right_x_7x7) if any(x > right_x_7x7 for x in vertical_lines) else warped_size

    outer_corners = [
        (outer_left_x, outer_top_y),
        (outer_right_x, outer_top_y),
        (outer_right_x, outer_bottom_y),
        (outer_left_x, outer_bottom_y)
    ]

    return corners, outer_corners

def get_perspective_transform(corners):
    size = 800
    dst_points = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    src_points = np.float32(corners)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix, size

def draw_grid(frame, size):
    step = size // 8
    for i in range(9):
        cv2.line(frame, (0, i * step), (size, i * step), (0, 255, 0), 1)
        cv2.line(frame, (i * step, 0), (i * step, size), (0, 255, 0), 1)

def draw_proposed_corners(frame, corners_data):
    points, outer_corners = corners_data
    for point in points:
        x, y = int(point[0][0]), int(point[0][1])
        cv2.circle(frame, (x, y), 5, (255, 0, 0), 2)
    for i in range(len(outer_corners)):
        x1, y1 = int(outer_corners[i][0]), int(outer_corners[i][1])
        x2, y2 = int(outer_corners[(i + 1) % 4][0]), int(outer_corners[(i + 1) % 4][1])
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def start_manual_perspective():
    global manual_selection_active, manual_corners, perspective_matrix, detection_method, detected_lines, proposed_corners, colored_frame
    manual_corners = []
    perspective_matrix = None
    detection_method = None
    detected_lines = None
    proposed_corners = None
    colored_frame = None
    manual_selection_active = True
    instruction_label.config(text="Click 4 corners (top-left, top-right, bottom-right, bottom-left)")

def reset_perspective():
    global perspective_matrix, manual_corners, manual_selection_active, detection_method, warped_size, detected_lines, proposed_corners, colored_frame, detected_fields
    perspective_matrix = None
    warped_size = 0
    manual_corners = []
    manual_selection_active = False
    detection_method = None
    detected_lines = None
    proposed_corners = None
    colored_frame = None
    detected_fields = None
    instruction_label.config(text="Select 'Manual Perspective' or detect lines (perspective view)")
    fen_text.delete(1.0, tk.END)
    fen_text.insert(tk.END, "Waiting for chessboard detection...")

def detect_lines_and_fields_method():
    global detected_lines, current_frame, warped_size, detected_fields, colored_frame
    frame_to_use = cv2.warpPerspective(current_frame, perspective_matrix, (warped_size, warped_size)) if perspective_matrix is not None else current_frame
    raw_lines = detect_lines(frame_to_use)
    if perspective_matrix is not None:
        detected_lines = group_lines_orthogonal(raw_lines, distance_threshold=distance_threshold.get())
        if detected_lines is not None:
            gray_frame = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2GRAY)
            horizontal_lines = sorted(set((line[0][1] + line[0][3]) // 2 for line in detected_lines if abs(line[0][1] - line[0][3]) < 20))
            vertical_lines = sorted(set((line[0][0] + line[0][2]) // 2 for line in detected_lines if abs(line[0][0] - line[0][2]) < 20))
            horizontal_lines = [0] + horizontal_lines + [warped_size]
            vertical_lines = [0] + vertical_lines + [warped_size]
            detected_fields = []
            for i in range(len(horizontal_lines) - 1):
                for j in range(len(vertical_lines) - 1):
                    x_start = vertical_lines[j]
                    y_start = horizontal_lines[i]
                    x_end = vertical_lines[j + 1]
                    y_end = horizontal_lines[i + 1]
                    region = gray_frame[y_start:y_end, x_start:x_end]
                    avg_color = np.median(region) if region.size > 0 else 0
                    color = (255, 255, 255) if avg_color > color_threshold.get() else (0, 0, 0)
                    detected_fields.append((x_start, y_start, x_end, y_end, color, avg_color))
            colored_frame = None
            instruction_label.config(text="Lines and fields detected (red lines, colored fields). Use Detect methods or reset.")
    else:
        detected_lines = group_lines_perspective(raw_lines)
        if detected_lines is not None:
            instruction_label.config(text="Lines detected (red). Use Detect with Pattern/Angles or reset.")
        else:
            instruction_label.config(text="No lines detected. Try adjusting the image or reset.")


def remove_largest_black_block(binary):
    """
    Find the largest connected black region in a binary image by pixel mass and color it white.
    Args:
        binary: Binary image (0 for black, 255 for white)
    Returns:
        Modified binary image with the largest black region turned white
    """
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        mask = np.zeros_like(binary, dtype=np.uint8)
        mask[labels == largest_label] = 255
        binary = cv2.bitwise_or(binary, mask)
    
    return binary

def remove_largest_black_block(binary):
    """
    Find the largest connected black region in a binary image by pixel mass and color it white.
    Args:
        binary: Binary image (0 for black, 255 for white)
    Returns:
        Modified binary image with the largest black region turned white
    """
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        mask = np.zeros_like(binary, dtype=np.uint8)
        mask[labels == largest_label] = 255
        binary = cv2.bitwise_or(binary, mask)
    
    return binary

def detect_with_pattern_method():
    global current_frame, detected_lines, proposed_corners, detected_fields, board_labels, is_assigned, bottom_left_field
    if detected_lines is not None and detected_fields is not None:
        print("Starting Detect with Pattern...")
        # Apply perspective transformation if needed
        frame_to_use = cv2.warpPerspective(current_frame, perspective_matrix, (warped_size, warped_size)) if perspective_matrix is not None else current_frame
        pattern_frame = frame_to_use.copy()
        print("Drawing fields...")
        for field in detected_fields:
            x_start, y_start, x_end, y_end, color, _ = field
            cv2.rectangle(pattern_frame, (x_start, y_start), (x_end, y_end), color, -1)
        print("Drawing lines...")
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(pattern_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        
        result = detect_chessboard_pattern(pattern_frame, None)
        if result is not None:
            proposed_corners = result
            instruction_label.config(text="Chessboard proposed (blue dots and frame). Confirm or reset.")
            print("Chessboard detected, starting text recognition...")
            
            threshold_value = color_threshold.get()
            outer_corners = proposed_corners[1]
            min_x, max_x = min(c[0] for c in outer_corners), max(c[0] for c in outer_corners)
            min_y, max_y = min(c[1] for c in outer_corners), max(c[1] for c in outer_corners)
            step_x, step_y = (max_x - min_x) / 8, (max_y - min_y) / 8
            
            left_bottom_region = pattern_frame[int(max_y - step_y):int(max_y), int(min_x):int(min_x + step_x)]
            gray_region = cv2.cvtColor(left_bottom_region, cv2.COLOR_BGR2GRAY)
            is_black = np.median(gray_region) < threshold_value
            
            board_labels = {"top": "", "bottom": "", "left": "", "right": ""}
            
            # Define character sets and sequences based on left bottom corner color
            if is_black:
                top_bottom_chars = "ABCDEFGH"  # A-H for top and bottom
                left_right_chars = "12345678"  # 1-8 for left and right
                top_bottom_seq_1 = list("ABCDEFGH")  # A-H (ascending)
                top_bottom_seq_2 = list("HGFEDCBA")  # H-A (descending)
                left_right_seq_1 = list("12345678")  # 1-8 (ascending)
                left_right_seq_2 = list("87654321")  # 8-1 (descending)
                top_bottom_seq_1_name = "A-H (ascending)"
                top_bottom_seq_2_name = "H-A (descending)"
                left_right_seq_1_name = "1-8 (ascending)"
                left_right_seq_2_name = "8-1 (descending)"
            else:
                top_bottom_chars = "12345678"  # 1-8 for top and bottom
                left_right_chars = "ABCDEFGH"  # A-H for left and right
                top_bottom_seq_1 = list("12345678")  # 1-8 (ascending)
                top_bottom_seq_2 = list("87654321")  # 8-1 (descending)
                left_right_seq_1 = list("ABCDEFGH")  # A-H (ascending)
                left_right_seq_2 = list("HGFEDCBA")  # H-A (descending)
                top_bottom_seq_1_name = "1-8 (ascending)"
                top_bottom_seq_2_name = "8-1 (descending)"
                left_right_seq_1_name = "A-H (ascending)"
                left_right_seq_2_name = "H-A (descending)"
            
            # Initialize scores for ascending and descending sequences in a dictionary
            scores = {
                "top_bottom_ascending": 0.0,
                "top_bottom_descending": 0.0,
                "left_right_ascending": 0.0,
                "left_right_descending": 0.0
            }
            
            def recognize_char(region, name, section_idx, allowed_chars, expected_seq_1, expected_seq_2, seq_1_name, seq_2_name, scores):
                if region.size == 0 or min(region.shape[:2]) < 10:
                    return ""
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
                binary = remove_largest_black_block(binary)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                binary = cv2.dilate(binary, kernel, iterations=2)
                cv2.imwrite(f"debug_{name}_{section_idx}_binary.png", binary)
                
                # Config with confidence threshold
                config = f'--psm 10 --oem 1 -c tessedit_char_whitelist={allowed_chars} -c textord_min_confidence=0.3 -c textord_min_xheight=10'
                best_text = ""
                best_conf = -1
                best_angle = 0
                best_seq_name = ""
                matches = []
                # Test different rotations
                for angle in [0, 90, 180, 270]:
                    rotated = binary.copy()
                    if angle != 0:
                        M = cv2.getRotationMatrix2D((binary.shape[1]//2, binary.shape[0]//2), angle, 1)
                        rotated = cv2.warpAffine(binary, M, (binary.shape[1], binary.shape[0]))
                    
                    # Use image_to_data to get confidence scores
                    data = pytesseract.image_to_data(rotated, config=config, output_type=pytesseract.Output.DICT)
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip().upper()
                        # Take only the first character if multiple are detected
                        if len(text) > 1:
                            text = text[0]
                        # Filter to allowed characters
                        text = ''.join(c for c in text if c in allowed_chars)
                        conf = float(data['conf'][i])
                        if text and conf >= 0.3:
                            # Check if the detected character matches the expected sequence
                            if text == expected_seq_1[section_idx]:
                                matches.append((text, conf, angle, seq_1_name))
                                if conf > best_conf:
                                    best_text = text
                                    best_conf = conf
                                    best_angle = angle
                                    best_seq_name = seq_1_name
                            elif text == expected_seq_2[section_idx]:
                                matches.append((text, conf, angle, seq_2_name))
                                if conf > best_conf:
                                    best_text = text
                                    best_conf = conf
                                    best_angle = angle
                                    best_seq_name = seq_2_name
                
                # Log all matches that fit the expected sequences and update scores
                for text, conf, angle, seq_name in matches:
                    print(f"Detected in {name} (section {section_idx}): '{text}' with confidence {conf:.2f} at angle {angle} deg, matched to sequence {seq_name}")
                    # Update scores based on the sequence
                    if name in ["top", "bottom"]:
                        if seq_name.endswith("(ascending)"):
                            scores["top_bottom_ascending"] += conf
                        else:
                            scores["top_bottom_descending"] += conf
                    else:  # left or right
                        if seq_name.endswith("(ascending)"):
                            scores["left_right_ascending"] += conf
                        else:
                            scores["left_right_descending"] += conf
                
                return best_text
            
            # Processing regions with restricted character sets
            top_region = frame_to_use[0:int(min_y), int(min_x):int(max_x)] if min_y > 0 else None
            bottom_region = frame_to_use[int(max_y):warped_size, int(min_x):int(max_x)] if max_y < warped_size else None
            left_region = frame_to_use[int(min_y):int(max_y), 0:int(min_x)] if min_x > 0 else None
            right_region = frame_to_use[int(min_y):int(max_y), int(max_x):warped_size] if max_x < warped_size else None
            
            if top_region is not None:
                # Divide top region into 8 equal parts horizontally
                region_width = top_region.shape[1]
                section_width = region_width // 8
                top_sections = []
                for i in range(8):
                    x_start = i * section_width
                    x_end = (i + 1) * section_width if i < 7 else region_width
                    section = top_region[:, x_start:x_end]
                    char = recognize_char(section, "top", i, top_bottom_chars, top_bottom_seq_1, top_bottom_seq_2, top_bottom_seq_1_name, top_bottom_seq_2_name, scores)
                    top_sections.append(char)
                board_labels["top"] = "".join(top_sections)
            
            if bottom_region is not None:
                # Divide bottom region into 8 equal parts horizontally
                region_width = bottom_region.shape[1]
                section_width = region_width // 8
                bottom_sections = []
                for i in range(8):
                    x_start = i * section_width
                    x_end = (i + 1) * section_width if i < 7 else region_width
                    section = bottom_region[:, x_start:x_end]
                    char = recognize_char(section, "bottom", i, top_bottom_chars, top_bottom_seq_1, top_bottom_seq_2, top_bottom_seq_1_name, top_bottom_seq_2_name, scores)
                    bottom_sections.append(char)
                board_labels["bottom"] = "".join(bottom_sections)
            
            if left_region is not None:
                # Divide left region into 8 equal parts vertically
                region_height = left_region.shape[0]
                section_height = region_height // 8
                left_sections = []
                for i in range(8):
                    y_start = i * section_height
                    y_end = (i + 1) * section_height if i < 7 else region_height
                    section = left_region[y_start:y_end, :]
                    section_idx = 7 - i  # Reverse order for left side
                    char = recognize_char(section, "left", section_idx, left_right_chars, left_right_seq_1, left_right_seq_2, left_right_seq_1_name, left_right_seq_2_name, scores)
                    left_sections.append(char)
                board_labels["left"] = "".join(left_sections)
            
            if right_region is not None:
                # Divide right region into 8 equal parts vertically
                region_height = right_region.shape[0]
                section_height = region_height // 8
                right_sections = []
                for i in range(8):
                    y_start = i * section_height
                    y_end = (i + 1) * section_height if i < 7 else region_height
                    section = right_region[y_start:y_end, :]
                    section_idx = 7 - i  # Reverse order for right side
                    char = recognize_char(section, "right", section_idx, left_right_chars, left_right_seq_1, left_right_seq_2, left_right_seq_1_name, left_right_seq_2_name, scores)
                    right_sections.append(char)
                board_labels["right"] = "".join(right_sections)
            
            # Determine the orientation of each group based on scores
            top_bottom_orientation = "ascending" if scores["top_bottom_ascending"] >= scores["top_bottom_descending"] else "descending"
            left_right_orientation = "ascending" if scores["left_right_ascending"] >= scores["left_right_descending"] else "descending"
            
            # Print the scores and orientations
            print(f"Top/Bottom scores: ascending={scores['top_bottom_ascending']:.2f}, descending={scores['top_bottom_descending']:.2f}, orientation={top_bottom_orientation}")
            print(f"Left/Right scores: ascending={scores['left_right_ascending']:.2f}, descending={scores['left_right_descending']:.2f}, orientation={left_right_orientation}")
            
            # Determine verdict
            if is_black:
                if top_bottom_orientation == "ascending" and left_right_orientation == "ascending":
                    verdict = "A1"
                elif top_bottom_orientation == "descending" and left_right_orientation == "descending":
                    verdict = "H8"
                else:
                    verdict = "unknown"
            else:
                if top_bottom_orientation == "ascending" and left_right_orientation == "descending":
                    verdict = "H1"
                elif top_bottom_orientation == "descending" and left_right_orientation == "ascending":
                    verdict = "A8"
                else:
                    verdict = "unknown"
            
            # Set global variables for decision and field
            is_assigned = (verdict != "unknown")
            bottom_left_field = verdict if is_assigned else None
            
            # Update the text field next to "Confirm Chessboard" button
            if is_assigned:
                instruction_label.config(text=f"Found chessboard and assigned fields: {verdict}")
            else:
                instruction_label.config(text="Found chessboard but fields not assigned")
            
            print(f"Left bottom color: {'black' if is_black else 'white'}")
            print(f"Detected labels: {board_labels}")
            if verdict == "unknown":
                print("Based on the recognized text, the numbering of the fields could not be determined.")
            else:
                print(f"Verdict: Left bottom corner is {verdict}")
        else:
            proposed_corners = None
            instruction_label.config(text="Pattern method failed to propose chessboard.")
    else:
        instruction_label.config(text="No lines or fields detected. Detect lines first.")



def confirm_chessboard_pattern():
    global detection_method, proposed_corners, bottom_left_field, is_assigned
    if proposed_corners is None:
        instruction_label.config(text="No proposed chessboard to confirm.")
        return
    
    # If fields were not assigned, default to A1 orientation
    if not is_assigned or bottom_left_field is None:
        instruction_label.config(text="Fields not assigned, defaulting to A1 orientation.")
        bottom_left_field = "A1"
    
    detection_method = "pattern"
    instruction_label.config(text=f"Chessboard confirmed (green frame and lines) with bottom-left field: {bottom_left_field}")
    update_fen()

def reset_detection():
    global detection_method, detected_lines, proposed_corners, colored_frame, detected_fields, board_labels
    detection_method = None
    detected_lines = None
    proposed_corners = None
    colored_frame = None
    detected_fields = None
    board_labels = None
    instruction_label.config(text="Detect lines or select a method" + (" (orthogonal view)" if perspective_matrix is not None else " (perspective view)"))
    fen_text.delete(1.0, tk.END)
    fen_text.insert(tk.END, "Waiting for chessboard detection...")

def exit_program():
    global running
    running = False

def update_fen():
    fen_text.delete(1.0, tk.END)
    fen_text.insert(tk.END, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Set up GUI window
root = tk.Tk()
root.title("ChessVision Control")
root.geometry("600x600")

show_colored_fields = tk.BooleanVar(value=True)
hough_threshold = tk.IntVar(value=120)
min_line_length = tk.IntVar(value=50)
distance_threshold = tk.IntVar(value=10)
color_threshold = tk.IntVar(value=90)

vcmd = (root.register(validate_natural), '%P')

instruction_label = ttk.Label(root, text="Select 'Manual Perspective' or detect lines (perspective view)")
instruction_label.pack(pady=5)

manual_perspective_button = ttk.Button(root, text="Manual Perspective", command=start_manual_perspective)
manual_perspective_button.pack(pady=5)

reset_perspective_button = ttk.Button(root, text="Reset Perspective", command=reset_perspective)
reset_perspective_button.pack(pady=5)

detect_lines_button = ttk.Button(root, text="Detect Lines and Fields", command=detect_lines_and_fields_method)
detect_lines_button.pack(pady=5)

detect_pattern_button = ttk.Button(root, text="Detect with Pattern", command=detect_with_pattern_method)
detect_pattern_button.pack(pady=5)

confirm_chessboard_pattern_button = ttk.Button(root, text="Confirm Chessboard (Pattern)", command=confirm_chessboard_pattern)
confirm_chessboard_pattern_button.pack(pady=5)

reset_detection_button = ttk.Button(root, text="Reset Detection", command=reset_detection)
reset_detection_button.pack(pady=5)

param_frame = ttk.Frame(root)
param_frame.pack(pady=10)

ttk.Label(param_frame, text="Hough Threshold:").grid(row=0, column=0, padx=5, pady=5)
hough_slider = ttk.Scale(param_frame, from_=50, to_=200, variable=hough_threshold, orient=tk.HORIZONTAL, 
                         command=lambda x: hough_threshold.set(int(float(x))))
hough_slider.grid(row=0, column=1, padx=5, pady=5)
hough_entry = ttk.Entry(param_frame, textvariable=hough_threshold, width=5, validate="key", validatecommand=vcmd)
hough_entry.grid(row=0, column=2, padx=5, pady=5)

ttk.Label(param_frame, text="Min Line Length:").grid(row=1, column=0, padx=5, pady=5)
min_length_slider = ttk.Scale(param_frame, from_=20, to_=200, variable=min_line_length, orient=tk.HORIZONTAL, 
                              command=lambda x: min_line_length.set(int(float(x))))
min_length_slider.grid(row=1, column=1, padx=5, pady=5)
min_length_entry = ttk.Entry(param_frame, textvariable=min_line_length, width=5, validate="key", validatecommand=vcmd)
min_length_entry.grid(row=1, column=2, padx=5, pady=5)

ttk.Label(param_frame, text="Distance Threshold:").grid(row=2, column=0, padx=5, pady=5)
distance_slider = ttk.Scale(param_frame, from_=5, to_=50, variable=distance_threshold, orient=tk.HORIZONTAL, 
                            command=lambda x: distance_threshold.set(int(float(x))))
distance_slider.grid(row=2, column=1, padx=5, pady=5)
distance_entry = ttk.Entry(param_frame, textvariable=distance_threshold, width=5, validate="key", validatecommand=vcmd)
distance_entry.grid(row=2, column=2, padx=5, pady=5)

ttk.Label(param_frame, text="Color Threshold:").grid(row=3, column=0, padx=5, pady=5)
color_slider = ttk.Scale(param_frame, from_=0, to_=255, variable=color_threshold, orient=tk.HORIZONTAL, 
                         command=lambda x: color_threshold.set(int(float(x))))
color_slider.grid(row=3, column=1, padx=5, pady=5)
color_entry = ttk.Entry(param_frame, textvariable=color_threshold, width=5, validate="key", validatecommand=vcmd)
color_entry.grid(row=3, column=2, padx=5, pady=5)

color_checkbox = ttk.Checkbutton(root, text="Show Colored Fields", variable=show_colored_fields)
color_checkbox.pack(pady=5)

exit_button = ttk.Button(root, text="Exit", command=exit_program)
exit_button.pack(pady=5)

fen_label = ttk.Label(root, text="Current Position (FEN):")
fen_label.pack(pady=5)
fen_text = tk.Text(root, height=2, width=40)
fen_text.pack(pady=5)
fen_text.insert(tk.END, "Waiting for chessboard detection...")

cv2.namedWindow('ChessVision')
cv2.setMouseCallback('ChessVision', select_corners)

# Main loop
while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break
    
    current_frame = frame.copy()
    display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    
    if manual_selection_active:
        for corner in manual_corners:
            cv2.circle(display_frame, (int(corner[0] * DISPLAY_SCALE), int(corner[1] * DISPLAY_SCALE)), 5, (0, 0, 255), -1)
    
    elif detected_lines is not None and proposed_corners is None and perspective_matrix is None:
        draw_lines(display_frame, detected_lines, scale=True)
    
    elif perspective_matrix is not None:
        warped_frame = cv2.warpPerspective(current_frame, perspective_matrix, (warped_size, warped_size))
        
        if show_colored_fields.get() and detected_fields is not None:
            colored_frame = warped_frame.copy()
            for field in detected_fields:
                x_start, y_start, x_end, y_end, color, avg_color = field
                cv2.rectangle(colored_frame, (x_start, y_start), (x_end, y_end), color, -1)
                if detection_method is None:
                    text = f"{int(avg_color)}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    text_x = x_start + ((x_end - x_start) - text_size[0]) // 2
                    text_y = y_start + ((y_end - y_start) + text_size[1]) // 2
                    cv2.putText(colored_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            warped_frame = colored_frame
        
        if detected_lines is not None:
            draw_lines(warped_frame, detected_lines)
        
        if proposed_corners is not None and detection_method is None:
            draw_proposed_corners(warped_frame, proposed_corners)
        
        elif detection_method == "pattern" and proposed_corners is not None:
            outer_corners = proposed_corners[1]
            for i in range(len(outer_corners)):
                x1, y1 = int(outer_corners[i][0]), int(outer_corners[i][1])
                x2, y2 = int(outer_corners[(i + 1) % 4][0]), int(outer_corners[(i + 1) % 4][1])
                cv2.line(warped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            min_x = min(c[0] for c in outer_corners)
            max_x = max(c[0] for c in outer_corners)
            min_y = min(c[1] for c in outer_corners)
            max_y = max(c[1] for c in outer_corners)
            
            if detected_lines is not None:
                for line in detected_lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 20:
                        x1 = max(min_x, min(max_x, x1))
                        x2 = max(min_x, min(max_x, x2))
                        y1 = y2 = max(min_y, min(max_y, y1))
                    elif abs(x1 - x2) < 20:
                        y1 = max(min_y, min(max_y, y1))
                        y2 = max(min_y, min(max_y, y2))
                        x1 = x2 = max(min_x, min(max_x, x1))
                    cv2.line(warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
            step_x = (max_x - min_x) / 8
            step_y = (max_y - min_y) / 8
            
            # Determine field labels based on bottom_left_field
            if bottom_left_field == "A1":
                # Standard orientation: A1 at (min_x, max_y)
                for i in range(8):
                    for j in range(8):
                        col_letter = chr(ord('A') + i)  # A to H (min_x to max_x)
                        row_number = str(8 - j)  # 1 to 8 (max_y to min_y)
                        label = f"{col_letter}{row_number}"
                        x = int(min_x + i * step_x + step_x / 2)
                        y = int(min_y + j * step_y + step_y / 2)
                        cv2.putText(warped_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            elif bottom_left_field == "H8":
                # Reversed orientation: H8 at (min_x, max_y)
                for i in range(8):
                    for j in range(8):
                        col_letter = chr(ord('H') - i)  # H to A (min_x to max_x)
                        row_number = str(j + 1)  # 8 to 1 (max_y to min_y)
                        label = f"{col_letter}{row_number}"
                        x = int(min_x + i * step_x + step_x / 2)
                        y = int(min_y + j * step_y + step_y / 2)
                        cv2.putText(warped_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            elif bottom_left_field == "H1":
                # Rotated 90 degrees clockwise: H1 at (min_x, max_y)
                for i in range(8):
                    for j in range(8):
                        col_letter = chr(ord('A') + j)  # A to H (max_y to min_y)
                        row_number = str(i + 1)  # 1 to 8 (min_x to max_x)
                        label = f"{col_letter}{row_number}"
                        x = int(min_x + i * step_x + step_x / 2)
                        y = int(min_y + j * step_y + step_y / 2)
                        cv2.putText(warped_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            elif bottom_left_field == "A8":
                # Rotated 90 degrees counterclockwise: A8 at (min_x, max_y)
                for i in range(8):
                    for j in range(8):
                        col_letter = chr(ord('H') - j)  # H to A (max_y to min_y)
                        row_number = str(8 - i)  # 8 to 1 (min_x to max_x)
                        label = f"{col_letter}{row_number}"
                        x = int(min_x + i * step_x + step_x / 2)
                        y = int(min_y + j * step_y + step_y / 2)
                        cv2.putText(warped_frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        display_frame = cv2.resize(warped_frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
    
    cv2.imshow('ChessVision', display_frame)
    root.update()
    
    if not cv2.getWindowProperty('ChessVision', cv2.WND_PROP_VISIBLE):
        break

cap.release()
cv2.destroyAllWindows()
root.destroy()
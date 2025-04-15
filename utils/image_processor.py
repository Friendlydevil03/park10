import cv2
import numpy as np


def process_parking_spaces(img_pro, img, pos_list, threshold, debug=False, space_groups=None):
    """Process and mark parking spaces in the image - with group support"""
    space_counter = 0

    # Create a copy of img only if needed for drawing
    if len(pos_list) > 0:
        img_display = img  # Use direct reference to avoid copy unless needed
    else:
        return img, 0, 0, 0  # Return early if no positions

    # Precompute font and colors to avoid recreation
    font = cv2.FONT_HERSHEY_SIMPLEX
    green_color = (0, 255, 0)  # Free space
    red_color = (0, 0, 255)  # Occupied space
    yellow_color = (255, 255, 0)  # Labels
    blue_color = (255, 165, 0)  # Group boundary color (orange)

    # Initialize group info and flattened groups list for easy lookup
    if space_groups is None:
        space_groups = {}

    # Create flat list for quick check if a space is in any group
    grouped_spaces = []
    for group_spaces in space_groups.values():
        grouped_spaces.extend(group_spaces)

    # Add debug info
    if debug:
        img_height, img_width = img.shape[:2]
        cv2.putText(img_display, f"Image size: {img_width}x{img_height}", (10, 20),
                    font, 0.5, yellow_color, 1)

    # Process all spaces individually (including those in groups)
    for i, pos in enumerate(pos_list):
        # Ensure pos is a tuple of length 4
        if not isinstance(pos, tuple) or len(pos) != 4:
            continue

        # Unpack and ensure all coordinates are integers
        x, y, w, h = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])

        # Ensure coordinates are within image bounds
        if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
            # Add box number and coordinates in debug mode
            if debug:
                coord_text = f"Box {i}: ({x},{y})"
                cv2.putText(img_display, coord_text, (x, y - 5),
                            font, 0.4, yellow_color, 1)

            # Extract crop for this parking space
            try:
                img_crop = img_pro[y:y + h, x:x + w]

                # Optimize counting - use sum instead of countNonZero for better performance
                count = np.sum(img_crop > 0)
            except Exception as e:
                print(f"Error processing space {i} at ({x},{y},{w},{h}): {str(e)}")
                continue

            if count < threshold:
                color = green_color  # Green for free
                space_counter += 1
            else:
                color = red_color  # Red for occupied

            # Check if this space belongs to a group
            is_in_group = i in grouped_spaces

            # Use thinner lines for spaces in groups
            line_thickness = 1 if is_in_group else 2

            # Draw rectangle and count for all spaces
            cv2.rectangle(img_display, (x, y), (x + w, y + h), color, line_thickness)

            # Draw ID number for each space
            if not is_in_group or debug:
                cv2.putText(img_display, str(i), (x + 5, y + 15),
                            font, 0.5, yellow_color, 1)

            # Draw count value for all spaces
            cv2.putText(img_display, str(count), (x, y + h - 3), font,
                        0.4, color, 1)

    # Second pass: Draw group boundaries
    for group_id, space_indices in space_groups.items():
        if not space_indices:
            continue

        try:
            # Calculate the group bounding box
            valid_indices = [i for i in space_indices if i < len(pos_list)]
            if not valid_indices:
                continue

            # Ensure all coordinates are integers when calculating min/max
            min_x = min(int(pos_list[i][0]) for i in valid_indices)
            min_y = min(int(pos_list[i][1]) for i in valid_indices)
            max_x = max(int(pos_list[i][0]) + int(pos_list[i][2]) for i in valid_indices)
            max_y = max(int(pos_list[i][1]) + int(pos_list[i][3]) for i in valid_indices)

            # Ensure coordinates are within image bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(img_pro.shape[1] - 1, max_x)
            max_y = min(img_pro.shape[0] - 1, max_y)

            # Count free spaces in this group
            group_free_count = 0
            group_total = 0

            # Count free spaces in this group
            for i in space_indices:
                if i < len(pos_list):
                    x, y, w, h = [int(coord) for coord in pos_list[i]]
                    # Ensure coordinates are valid
                    if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
                        try:
                            img_crop = img_pro[y:y + h, x:x + w]
                            count = np.sum(img_crop > 0)
                            group_total += 1
                            if count < threshold:
                                group_free_count += 1
                        except Exception as e:
                            print(f"Error processing group space {i} at ({x},{y},{w},{h}): {str(e)}")

            # Draw group boundary
            cv2.rectangle(img_display, (min_x - 3, min_y - 3), (max_x + 3, max_y + 3), blue_color, 2)

            # Add group label with free/total count
            group_label = group_id.split('_')[-1] if '_' in group_id else group_id
            cv2.putText(img_display, f"G{group_label}: {group_free_count}/{group_total}",
                        (min_x, min_y - 5), font, 0.6, blue_color, 2)
        except Exception as e:
            print(f"Error drawing group {group_id}: {str(e)}")

    # Display total count stats
    free_spaces = space_counter
    total_spaces = len(pos_list)
    occupied_spaces = total_spaces - free_spaces

    return img_display, free_spaces, occupied_spaces, total_spaces


def detect_vehicles_traditional(current_frame, prev_frame, line_height, min_contour_width, min_contour_height, offset,
                                matches, vehicles_count):
    """
    Detect vehicles using traditional computer vision - optimized version
    """
    # Only create a copy of the frame if we need to draw on it
    display_frame = current_frame.copy()

    # Calculate absolute difference between frames
    d = cv2.absdiff(prev_frame, current_frame)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    # Apply blur and threshold
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Apply dilation and morphology operations
    # Optimize by combining operations when possible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(cv2.dilate(th, np.ones((3, 3))), cv2.MORPH_CLOSE, kernel)

    # Find contours - use EXTERNAL type for faster processing
    contours, h = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make a copy of matches only if needed (if we have contours)
    if not contours:
        # Draw detection line
        cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Total Vehicle Detected: {vehicles_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)
        return display_frame, matches, vehicles_count

    matches_copy = matches.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Process each contour - reduce the number processed if there are too many
    max_contours = 50  # Maximum contours to process for performance
    for (i, c) in enumerate(contours[:max_contours]):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        # Draw rectangle around vehicle
        cv2.rectangle(display_frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        # Calculate centroid
        cx = x + w // 2
        cy = y + h // 2
        centroid = (cx, cy)

        # Add centroid to matches list
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        # Check if centroid is near the line
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            # Keep centroids that haven't crossed the line
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    return display_frame, new_matches, new_vehicles_count


def process_ml_detections(frame, detections, line_height, offset, matches, vehicles_count, class_names):
    """Process detections from ML model - optimized version"""
    display_frame = frame.copy()

    # Draw detection line
    cv2.line(display_frame, (0, line_height), (display_frame.shape[1], line_height), (0, 255, 0), 2)

    # Make a deep copy of matches list
    matches_copy = matches.copy() if matches is not None else []

    # Handle case where detections might be None
    if detections is None:
        detections = []

    # Process only a limited number of detections for performance
    max_detections = 30
    for i, detection in enumerate(detections[:max_detections]):
        # Ensure detection has the expected format
        if len(detection) < 3:  # We need at least box, score, label
            continue

        box, score, label = detection[:3]  # Unpack the first 3 elements

        # Ensure box is valid
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centroid = (cx, cy)

        # Only add label if score is high enough (optimization)
        if score > 0.6:
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            cv2.putText(display_frame, f"{class_name}: {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add centroid
        matches_copy.append(centroid)

        # Draw centroid
        cv2.circle(display_frame, centroid, 5, (0, 0, 255), -1)

    # Count vehicles crossing the line
    new_vehicles_count = vehicles_count
    new_matches = []

    for (x, y) in matches_copy:
        if line_height - offset < y < line_height + offset:
            new_vehicles_count += 1
        else:
            new_matches.append((x, y))

    # Display vehicle count
    cv2.putText(display_frame, f"Total Vehicle Detected: {new_vehicles_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 170, 0), 2)

    # Return all required values
    return display_frame, new_matches, new_vehicles_count

import cv2
import numpy as np
import threading
import time
import os
import pickle
from datetime import datetime


def get_centroid(x, y, w, h):
    """Calculate centroid of a rectangle"""
    return x + w // 2, y + h // 2


class ParkingManager:
    """Core parking management functionality separate from UI"""

    DEFAULT_CONFIDENCE = 0.6
    DEFAULT_THRESHOLD = 500
    MIN_CONTOUR_SIZE = 40
    DEFAULT_OFFSET = 10
    DEFAULT_LINE_HEIGHT = 400

    def __init__(self, config_dir="config", log_dir="logs"):
        # Initialize directories
        self.config_dir = config_dir
        self.log_dir = log_dir
        self._ensure_directories_exist()

        # Initialize tracking variables
        self.posList = []
        self.total_spaces = 0
        self.free_spaces = 0
        self.occupied_spaces = 0
        self.vehicle_counter = 0
        self.matches = []

        # Detection parameters
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.detection_mode = "parking"
        self.line_height = self.DEFAULT_LINE_HEIGHT
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET

        # Video/image references
        self.video_reference_map = {}
        self.reference_dimensions = {}
        self.current_reference_image = None
        self.current_video = None

        # ML detection
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = self.DEFAULT_CONFIDENCE

        # Thread safety
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.log_data = []

        # For the parking allocation system
        self.parking_visualizer = None
        self.parking_data = {}

        # For simultaneous detection
        self.simultaneous_mode = False
        self.vehicle_detection_result = None
        self.vehicle_detection_thread = None
        self.parking_detection_result = None
        self.parking_detection_thread = None

    def _ensure_directories_exist(self):
        """Ensure necessary directories exist"""
        for directory in [self.config_dir, self.log_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    # Update the save_parking_positions method
    def save_parking_positions(self, reference_image):
        """Save parking positions to file, ensuring clean save"""
        try:
            # Create a clean file name without extension for consistency
            base_name = os.path.splitext(os.path.basename(reference_image))[0]
            pos_file = os.path.join(self.config_dir, f'CarParkPos_{base_name}')

            # Clear any existing file to prevent append issues
            if os.path.exists(pos_file):
                os.remove(pos_file)

            # Now save the current positions to a clean file
            with open(pos_file, 'wb') as f:
                pickle.dump(self.posList, f)
                print(f"Saved {len(self.posList)} parking positions to {pos_file}")

            return True
        except Exception as e:
            print(f"Error saving parking positions: {str(e)}")
            return False

    # Update the load_parking_positions method
    def load_parking_positions(self, reference_image):
        """Load parking positions from file with extensive validation"""
        try:
            # Create a clean file name without extension
            base_name = os.path.splitext(os.path.basename(reference_image))[0]
            pos_file = os.path.join(self.config_dir, f'CarParkPos_{base_name}')

            # Start with empty lists
            self.posList = []

            if os.path.exists(pos_file):
                with open(pos_file, 'rb') as f:
                    loaded_data = pickle.load(f)

                    # Check if loaded data is a list
                    if not isinstance(loaded_data, list):
                        raise TypeError(f"Expected a list of positions, got {type(loaded_data)}")

                    # Validate each position entry
                    for pos in loaded_data:
                        # Check if position is a tuple of 4 numbers
                        if (isinstance(pos, tuple) and len(pos) == 4 and
                                all(isinstance(coord, (int, float)) for coord in pos)):
                            # Verify coordinates are positive
                            x, y, w, h = pos
                            if x >= 0 and y >= 0 and w > 0 and h > 0:
                                self.posList.append(pos)
                            else:
                                print(f"Skipped invalid position with negative values: {pos}")

                    # Remove any duplicates
                    seen = set()
                    unique_positions = []
                    for pos in self.posList:
                        pos_tuple = tuple(map(int, pos))  # Convert to integers for comparison
                        if pos_tuple not in seen:
                            seen.add(pos_tuple)
                            unique_positions.append(pos)

                    self.posList = unique_positions

                    print(f"Loaded {len(self.posList)} valid parking positions from {pos_file}")

                    # Update counters
                    self.total_spaces = len(self.posList)
                    self.free_spaces = 0
                    self.occupied_spaces = self.total_spaces
                    return True
            else:
                print(f"No parking positions file found at {pos_file}")
                self.total_spaces = 0
                self.free_spaces = 0
                self.occupied_spaces = 0
                return False
        except Exception as e:
            print(f"Error loading parking positions: {str(e)}")
            # Reset to prevent inconsistent state
            self.posList = []
            self.total_spaces = 0
            self.free_spaces = 0
            self.occupied_spaces = 0
            return False

    def clear_parking_positions(self):
        """Clear all parking positions and related data"""
        with self.data_lock:  # Use the existing lock for thread safety
            self.posList = []
            self.total_spaces = 0
            self.free_spaces = 0
            self.occupied_spaces = 0
            self.parking_data = {}  # Clear parking data dictionary
            print("All parking positions cleared")
        return True

    def check_parking_space(self, img_pro, img):
        """Process frame to check parking spaces"""
        space_counter = 0
        for i, pos in enumerate(self.posList):
            # Skip invalid position formats - this is the key fix
            if not isinstance(pos, tuple) or len(pos) != 4:
                continue

            # Now we know pos is a valid 4-value tuple
            x, y, w, h = pos

            # Ensure coordinates are within image bounds
            if (y >= 0 and y + h < img_pro.shape[0] and
                    x >= 0 and x + w < img_pro.shape[1]):

                # Get crop of parking space
                img_crop = img_pro[y:y + h, x:x + w]
                count = cv2.countNonZero(img_crop)

                if count < self.parking_threshold:
                    color = (0, 255, 0)  # Green for free
                    space_counter += 1
                else:
                    color = (0, 0, 255)  # Red for occupied

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Add count text
                text_scale = 0.6
                text_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(
                    str(count), cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
                )
                text_x = x + (w - text_width) // 2
                text_y = y + h - 5
                cv2.putText(img, str(count), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

        # Update counters
        self.free_spaces = space_counter
        self.occupied_spaces = self.total_spaces - self.free_spaces

        return img

    def update_individual_slots_in_groups(self, img_pro):
        """
        Update status of individual slots within groups
        This allows showing individual slot status in the parking allocation view
        """
        if not hasattr(self, 'parking_data'):
            return

        # First pass: Update individual slot statuses
        for i, pos in enumerate(self.posList):
            # Skip invalid position formats
            if not isinstance(pos, tuple) or len(pos) != 4:
                continue

            # Now we know pos is a valid 4-value tuple
            x, y, w, h = pos

            # Generate section and space ID
            section = "A" if x < img_pro.shape[1] / 2 else "B"
            section += "1" if y < img_pro.shape[0] / 2 else "2"
            space_id = f"S{i + 1}-{section}"

            # Ensure coordinates are within image bounds
            if (y >= 0 and y + h < img_pro.shape[0] and x >= 0 and x + w < img_pro.shape[1]):
                # Get crop of parking space
                img_crop = img_pro[y:y + h, x:x + w]
                count = cv2.countNonZero(img_crop)
                is_occupied = count >= self.parking_threshold

                # Update or create the entry in parking_data
                if space_id not in self.parking_data:
                    self.parking_data[space_id] = {
                        'position': (x, y, w, h),
                        'occupied': is_occupied,
                        'vehicle_id': None,
                        'last_state_change': datetime.now(),
                        'distance_to_entrance': x + y,
                        'section': section,
                        'in_group': False,  # Not part of a group by default
                        'group_id': None  # No group by default
                    }
                else:
                    # Only update occupied status if not manually set
                    if not self.parking_data[space_id].get('manually_set', False):
                        self.parking_data[space_id]['occupied'] = is_occupied

        # Second pass: Update group information
        for group_id, data in list(self.parking_data.items()):
            if data.get('is_group', False) and 'member_spaces' in data:
                member_spaces = data['member_spaces']

                # For each member space, mark it as part of a group
                for i in member_spaces:
                    if i < len(self.posList):
                        # Generate ID for this individual space
                        x, y, w, h = self.posList[i]
                        section = "A" if x < img_pro.shape[1] / 2 else "B"
                        section += "1" if y < img_pro.shape[0] / 2 else "2"
                        space_id = f"S{i + 1}-{section}"

                        # Update the space to know it belongs to a group
                        if space_id in self.parking_data:
                            self.parking_data[space_id]['in_group'] = True
                            self.parking_data[space_id]['group_id'] = group_id

    def update_allocation_status(self, img_pro, img):
        """Update both the original parking status and the allocation system"""
        # First, process with the existing method to determine which spaces are free/occupied
        img = self.check_parking_space(img_pro, img)

        # Update individual slots and mark those in groups
        self.update_individual_slots_in_groups(img_pro)

        # Then update group status
        self.process_group_status(img_pro, img)

        return img

    def detect_vehicles(self, frame1, frame2):
        """Process frames to detect and count vehicles"""
        # Get difference between frames
        d = cv2.absdiff(frame1, frame2)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detection line
        line_y = self.line_height
        if line_y >= frame1.shape[0]:
            line_y = frame1.shape[0] - 50
        cv2.line(frame1, (0, line_y), (frame1.shape[1], line_y), (0, 255, 0), 2)

        # Process contours
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= self.min_contour_width) and (h >= self.min_contour_height)

            if not contour_valid:
                continue

            cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            centroid = get_centroid(x, y, w, h)
            self.matches.append(centroid)
            cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

        # Check for vehicles crossing the line
        new_matches = []
        for (x, y) in self.matches:
            if (line_y - self.offset) < y < (line_y + self.offset):
                self.vehicle_counter += 1
            else:
                new_matches.append((x, y))

        self.matches = new_matches

        # Display count
        cv2.putText(frame1, f"Vehicle Count: {self.vehicle_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        return frame1

    def scale_positions(self, orig_width, orig_height, new_width, new_height):
        """Scale parking positions based on current video dimensions"""
        # Calculate scale factors
        width_scale = new_width / orig_width
        height_scale = new_height / orig_height

        # Scale all positions
        scaled_positions = []
        for x, y, w, h in self.posList:
            new_x = int(x * width_scale)
            new_y = int(y * height_scale)
            new_w = int(w * width_scale)
            new_h = int(h * height_scale)
            scaled_positions.append((new_x, new_y, new_w, new_h))

        self.posList = scaled_positions

    def cleanup(self):
        """Clean up resources"""
        with self._cleanup_lock:
            if hasattr(self, 'ml_detector') and self.ml_detector:
                del self.ml_detector

            # Clean up any other resources here
            import gc
            gc.collect()

    # New methods for simultaneous detection

    def process_frame_simultaneous(self, current_frame, prev_frame=None):
        """Process a frame using both parking and vehicle detection simultaneously"""
        if self.simultaneous_mode:
            # Create copies of the frame for each detection method
            parking_frame = current_frame.copy()
            vehicle_frame = current_frame.copy() if prev_frame is not None else None

            # Start threads for parallel processing
            self.parking_detection_thread = threading.Thread(
                target=self._process_parking_detection,
                args=(parking_frame,)
            )
            self.parking_detection_thread.start()

            if prev_frame is not None and vehicle_frame is not None:
                self.vehicle_detection_thread = threading.Thread(
                    target=self._process_vehicle_detection,
                    args=(vehicle_frame, prev_frame)
                )
                self.vehicle_detection_thread.start()

            # Wait for threads to complete
            if self.parking_detection_thread:
                self.parking_detection_thread.join()

            if prev_frame is not None and self.vehicle_detection_thread:
                self.vehicle_detection_thread.join()

            # Merge results into one frame
            result_frame = current_frame.copy()

            # If we have parking results, draw parking spaces
            if self.parking_detection_result is not None:
                for i, (x, y, w, h, is_free) in enumerate(self.parking_detection_result):
                    color = (0, 255, 0) if is_free else (0, 0, 255)  # Green for free, Red for occupied
                    cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

                    # Add space ID
                    cv2.putText(result_frame, f"S{i}", (x + 5, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # If we have vehicle detection results, draw vehicles and count
            if self.vehicle_detection_result is not None and prev_frame is not None:
                # Draw detection line
                line_y = self.line_height
                if line_y >= result_frame.shape[0]:
                    line_y = result_frame.shape[0] - 50
                cv2.line(result_frame, (0, line_y), (result_frame.shape[1], line_y), (0, 255, 0), 2)

                # Draw vehicle count
                cv2.putText(result_frame, f"Vehicle Count: {self.vehicle_counter}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

                # Draw rectangles for detected vehicles
                for rect in self.vehicle_detection_result:
                    x, y, w, h = rect
                    cv2.rectangle(result_frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

            # Add parking space information
            cv2.putText(result_frame, f"Free: {self.free_spaces}/{self.total_spaces}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return result_frame
        else:
            # Use standard detection based on current mode
            if self.detection_mode == "parking":
                # Preprocess the frame for parking detection
                imgGray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
                imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY_INV, 25, 16)
                imgProcessed = cv2.medianBlur(imgThreshold, 5)
                kernel = np.ones((3, 3), np.uint8)
                imgProcessed = cv2.dilate(imgProcessed, kernel, iterations=1)

                return self.check_parking_space(imgProcessed, current_frame.copy())
            elif prev_frame is not None:
                return self.detect_vehicles(current_frame.copy(), prev_frame)
            else:
                return current_frame.copy()

    def _process_parking_detection(self, frame):
        """Process parking detection in a separate thread"""
        # Preprocess the frame
        imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgProcessed = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgProcessed = cv2.dilate(imgProcessed, kernel, iterations=1)

        # Process each parking space
        parking_results = []
        space_counter = 0

        with self.data_lock:
            for i, (x, y, w, h) in enumerate(self.posList):
                # Ensure coordinates are within image bounds
                if (y >= 0 and y + h < imgProcessed.shape[0] and x >= 0 and x + w < imgProcessed.shape[1]):
                    # Get crop of parking space
                    img_crop = imgProcessed[y:y + h, x:x + w]
                    count = cv2.countNonZero(img_crop)
                    is_free = count < self.parking_threshold

                    if is_free:
                        space_counter += 1

                    # Store results
                    parking_results.append((x, y, w, h, is_free))

            self.free_spaces = space_counter
            self.total_spaces = len(self.posList)
            self.occupied_spaces = self.total_spaces - self.free_spaces
            self.parking_detection_result = parking_results

    def _process_vehicle_detection(self, current_frame, prev_frame):
        """Process vehicle detection in a separate thread"""
        # Get difference between frames
        d = cv2.absdiff(prev_frame, current_frame)
        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours and update vehicle detection
        vehicle_results = []
        line_y = self.line_height

        # Make a copy of matches to work with
        with self.data_lock:
            matches_copy = self.matches.copy()

        # Process each contour
        for (i, c) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(c)
            contour_valid = (w >= self.min_contour_width) and (h >= self.min_contour_height)

            if not contour_valid:
                continue

            # Save for drawing
            vehicle_results.append((x, y, w, h))

            # Add centroid
            centroid = get_centroid(x, y, w, h)
            matches_copy.append(centroid)

        # Check for vehicles crossing the line
        new_matches = []
        new_counter = self.vehicle_counter

        for (x, y) in matches_copy:
            if (line_y - self.offset) < y < (line_y + self.offset):
                new_counter += 1
            else:
                new_matches.append((x, y))

        # Update vehicle data
        with self.data_lock:
            self.matches = new_matches
            self.vehicle_counter = new_counter
            self.vehicle_detection_result = vehicle_results

    # Add these methods to the ParkingManager class

    def process_group_status(self, img_pro, img):
        """Process the status of grouped parking spaces"""
        if not hasattr(self, 'parking_data'):
            return

        # Find all entries that are groups
        for space_id, data in list(self.parking_data.items()):
            if data.get('is_group', False) and 'member_spaces' in data:
                member_spaces = data['member_spaces']
                total_members = len(member_spaces)

                if total_members == 0:
                    continue

                # Count occupied spaces in this group
                occupied_count = 0
                for i in member_spaces:
                    if i < len(self.posList):
                        # Get data for this space
                        x, y, w, h = self.posList[i]

                        # Check if space is within image bounds
                        if (y >= 0 and y + h < img_pro.shape[0] and
                                x >= 0 and x + w < img_pro.shape[1]):

                            # Get crop of parking space
                            img_crop = img_pro[y:y + h, x:x + w]
                            count = cv2.countNonZero(img_crop)

                            if count >= self.parking_threshold:
                                occupied_count += 1

                # Determine if group is occupied (more than 50% of spaces occupied)
                is_group_occupied = occupied_count > (total_members / 2)

                # Update group status
                data['occupied'] = is_group_occupied

                # Draw group boundary on the image
                x, y, w, h = data['position']
                color = (0, 0, 255) if is_group_occupied else (0, 255, 0)  # Red if occupied, green if free
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), color, 2)

                # Add group label
                cv2.putText(img, f"{space_id}: {occupied_count}/{total_members}",
                            (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def sync_group_data(self, space_groups):
        """
        Synchronize group data from setup tab to parking manager

        Args:
            space_groups: Dictionary mapping group_id -> list of space indices
        """
        if not hasattr(self, 'parking_data'):
            self.parking_data = {}

        # First, clear all existing group flags
        for space_id, data in self.parking_data.items():
            if not data.get('is_group', False):  # Skip group entries themselves
                data['in_group'] = False
                data['group_id'] = None

        # Remove existing group entries
        group_keys = [k for k, v in self.parking_data.items() if v.get('is_group', False)]
        for key in group_keys:
            self.parking_data.pop(key, None)

        # Add new group entries and update space memberships
        for group_id, space_indices in space_groups.items():
            if not space_indices:
                continue

            valid_indices = [i for i in space_indices if i < len(self.posList)]
            if not valid_indices:
                continue

            # Calculate group bounds
            min_x = min(self.posList[i][0] for i in valid_indices)
            min_y = min(self.posList[i][1] for i in valid_indices)
            max_x = max(self.posList[i][0] + self.posList[i][2] for i in valid_indices)
            max_y = max(self.posList[i][1] + self.posList[i][3] for i in valid_indices)

            # Add group entry
            self.parking_data[group_id] = {
                'position': (min_x, min_y, max_x - min_x, max_y - min_y),
                'is_group': True,
                'occupied': False,  # Will be calculated based on member spaces
                'member_spaces': space_indices.copy(),
                'vehicle_id': None,
                'last_state_change': datetime.now(),
                'section': 'G'  # Special section for groups
            }

            # Mark individual spaces as part of this group
            for i in space_indices:
                if i < len(self.posList):
                    x, y, w, h = self.posList[i]
                    # Calculate section based on position
                    section = "A" if x < self.posList[0][0] + self.posList[0][2] * len(self.posList) / 2 else "B"
                    section += "1" if y < self.posList[0][1] + self.posList[0][3] * len(self.posList) / 2 else "2"

                    space_id = f"S{i + 1}-{section}"

                    if space_id in self.parking_data:
                        self.parking_data[space_id]['in_group'] = True
                        self.parking_data[space_id]['group_id'] = group_id
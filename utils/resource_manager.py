import os
import pickle
from datetime import datetime


def ensure_directories_exist(directories):
    """Ensure necessary directories exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_parking_positions(config_dir, reference_image):
    """Load parking positions from file with improved error handling"""
    try:
        pos_file = os.path.join(config_dir, f'CarParkPos_{os.path.splitext(reference_image)[0]}')

        if os.path.exists(pos_file):
            with open(pos_file, 'rb') as f:
                positions = pickle.load(f)

                # Validate the positions to ensure each entry is either a 4-value tuple or a dictionary
                valid_positions = []
                for pos in positions:
                    if isinstance(pos, tuple) and len(pos) == 4:
                        valid_positions.append(pos)
                    elif isinstance(pos, dict):
                        valid_positions.append(pos)
                    else:
                        print(f"Warning: Skipping invalid position format: {pos}")

                if len(valid_positions) < len(positions):
                    print(f"Warning: Filtered out {len(positions) - len(valid_positions)} invalid positions")

                return valid_positions
        else:
            return []

    except Exception as e:
        print(f"Error loading parking positions: {str(e)}")
        return []

def save_parking_positions(positions, config_dir, reference_image):
    """
    Save parking positions to a file

    Args:
        positions: List of (x, y, w, h) tuples
        config_dir: Directory to save the file in
        reference_image: Name of the reference image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import os
        import pickle

        # Create a clean file name without extension
        base_name = os.path.splitext(os.path.basename(reference_image))[0]
        pos_file = os.path.join(config_dir, f'CarParkPos_{base_name}')

        # Ensure config directory exists
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        # Delete the file if it exists to ensure a clean write
        if os.path.exists(pos_file):
            os.remove(pos_file)

        # Save positions to file
        with open(pos_file, 'wb') as f:
            pickle.dump(positions, f)

        print(f"Saved {len(positions)} parking positions to {pos_file}")
        return True
    except Exception as e:
        print(f"Error saving parking positions: {str(e)}")
        return False

def save_positions_with_groups(positions, groups, config_dir, reference_image):
    """
    Save parking positions and group data separately to avoid breaking video processing

    Args:
        positions: List of (x, y, w, h) tuples - only the regular parking spaces
        groups: List of group metadata dictionaries
        config_dir: Directory to save the file in
        reference_image: Name of the reference image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import os
        import pickle

        # Save the regular positions using the existing function
        success = save_parking_positions(positions, config_dir, reference_image)

        # Also save the groups data separately
        if groups:
            # Create a clean file name without extension
            base_name = os.path.splitext(os.path.basename(reference_image))[0]
            group_file = os.path.join(config_dir, f'Groups_{base_name}')

            # Ensure config directory exists
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)

            # Delete the file if it exists to ensure a clean write
            if os.path.exists(group_file):
                os.remove(group_file)

            # Save groups to file
            with open(group_file, 'wb') as f:
                pickle.dump(groups, f)

            print(f"Saved {len(groups)} group(s) to {group_file}")

        return success
    except Exception as e:
        print(f"Error saving positions with groups: {str(e)}")
        return False


def save_log(log_data, log_dir):
    """Save log data to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_log_{timestamp}.txt")

        with open(filename, 'w') as f:
            for entry in log_data:
                f.write(entry + "\n")

        return filename
    except Exception as e:
        print(f"Error saving log: {str(e)}")
        return None


def export_statistics(stats_data, log_dir):
    """Export statistics to CSV"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_stats_{timestamp}.csv")

        with open(filename, 'w') as f:
            f.write("Timestamp,Total Spaces,Free Spaces,Occupied Spaces,Vehicles Counted\n")

            for row in stats_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

        return filename
    except Exception as e:
        print(f"Error exporting statistics: {str(e)}")
        return None
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import random
from datetime import datetime
import traceback


class ParkingAllocationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Load components from other modules
        self.visualizer = self.app.parking_visualizer
        self.allocation_engine = self.app.allocation_engine

        # Set up UI state variables
        self.show_visualization = tk.BooleanVar(value=True)
        self.highlight_free_spaces = tk.BooleanVar(value=True)
        self.auto_allocation_enabled = tk.BooleanVar(value=False)
        self.preferred_section = tk.StringVar(value="Any")
        self.load_balancing_weight = tk.DoubleVar(value=0.3)
        self.vehicle_size = tk.IntVar(value=1)

        # Queue for thread-safe UI updates
        self.update_queue = queue.Queue()

        # Variables for vehicle simulation
        self.next_vehicle_id = 1
        self.allocated_vehicles = {}

        # Setup UI components
        self.setup_ui()

        # Start update thread
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def setup_ui(self):
        """Set up the UI components for the parking allocation tab"""
        # Configure grid layout
        self.parent.grid_columnconfigure(0, weight=3)  # Visualization area
        self.parent.grid_columnconfigure(1, weight=1)  # Control panel
        self.parent.grid_rowconfigure(0, weight=1)

        # Left side - Visualization area
        self.viz_frame = ttk.Frame(self.parent)
        self.viz_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)

        # Canvas for visualization
        self.canvas_frame = ttk.Frame(self.viz_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        # Initial visualization with explicit DPI setting
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)  # Explicitly set DPI
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Right side - Control panel
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Control panel sections
        self._add_visualization_controls()
        self._add_separator()
        self._add_allocation_controls()
        self._add_separator()
        self._add_stats_section()
        self._add_separator()
        self._add_simulation_controls()

    def _add_visualization_controls(self):
        # (Keep the existing implementation)
        """Add visualization controls to the control panel"""
        viz_frame = ttk.LabelFrame(self.control_frame, text="Visualization")
        viz_frame.pack(fill="x", padx=5, pady=5)

        # Toggle visualization view
        viz_check = ttk.Checkbutton(viz_frame, text="Show visualization",
                                    variable=self.show_visualization,
                                    command=self.update_visualization)
        viz_check.pack(anchor="w", padx=5, pady=3)

        # Highlight free spaces
        highlight_check = ttk.Checkbutton(viz_frame, text="Highlight free spaces",
                                          variable=self.highlight_free_spaces,
                                          command=self.update_visualization)
        highlight_check.pack(anchor="w", padx=5, pady=3)

        # Update button
        update_btn = ttk.Button(viz_frame, text="Refresh Visualization",
                                command=self.update_visualization)
        update_btn.pack(fill="x", padx=5, pady=5)

    def _add_allocation_controls(self):
        """Add allocation controls to the control panel"""
        alloc_frame = ttk.LabelFrame(self.control_frame, text="Space Allocation")
        alloc_frame.pack(fill="x", padx=5, pady=5)

        # Auto-allocation toggle
        auto_check = ttk.Checkbutton(alloc_frame, text="Enable auto allocation",
                                     variable=self.auto_allocation_enabled)
        auto_check.pack(anchor="w", padx=5, pady=3)

        # Section preference
        section_frame = ttk.Frame(alloc_frame)
        section_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(section_frame, text="Preferred section:").pack(side="left")
        section_combo = ttk.Combobox(section_frame, textvariable=self.preferred_section,
                                     values=["Any", "A", "B", "C", "D"])
        section_combo.pack(side="left", padx=5)

        # Vehicle size
        size_frame = ttk.Frame(alloc_frame)
        size_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(size_frame, text="Vehicle size:").pack(side="left")
        size_combo = ttk.Combobox(size_frame, textvariable=self.vehicle_size,
                                  values=[1, 2, 3])
        size_combo.pack(side="left", padx=5)

        # Load balancing weight
        lb_frame = ttk.Frame(alloc_frame)
        lb_frame.pack(fill="x", padx=5, pady=3)

        ttk.Label(lb_frame, text="Load balancing:").pack(side="left")
        lb_scale = ttk.Scale(lb_frame, from_=0.0, to=1.0, orient="horizontal",
                             variable=self.load_balancing_weight)
        lb_scale.pack(side="left", fill="x", expand=True, padx=5)

        # Allocate button
        allocate_btn = ttk.Button(alloc_frame, text="Allocate Vehicle",
                                  command=self.allocate_new_vehicle)
        allocate_btn.pack(fill="x", padx=5, pady=5)

    def _add_stats_section(self):
        """Add statistics section to the control panel"""
        stats_frame = ttk.LabelFrame(self.control_frame, text="Parking Statistics")
        stats_frame.pack(fill="x", padx=5, pady=5)

        # Stats display
        self.stats_text = tk.Text(stats_frame, height=6, width=30, wrap="word")
        self.stats_text.pack(fill="both", padx=5, pady=5)
        self.stats_text.insert("end", "Loading statistics...")
        self.stats_text.config(state="disabled")

    def _add_simulation_controls(self):
        """Add simulation controls to the control panel"""
        sim_frame = ttk.LabelFrame(self.control_frame, text="Simulation")
        sim_frame.pack(fill="x", padx=5, pady=5)

        # Simulation buttons
        add_vehicle_btn = ttk.Button(sim_frame, text="Add Random Vehicle",
                                     command=self.add_random_vehicle)
        add_vehicle_btn.pack(fill="x", padx=5, pady=3)

        remove_vehicle_btn = ttk.Button(sim_frame, text="Remove Random Vehicle",
                                        command=self.remove_random_vehicle)
        remove_vehicle_btn.pack(fill="x", padx=5, pady=3)

        reset_btn = ttk.Button(sim_frame, text="Reset Simulation",
                               command=self.reset_simulation)
        reset_btn.pack(fill="x", padx=5, pady=3)

    def _add_separator(self):
        """Add a separator line to the control panel"""
        ttk.Separator(self.control_frame, orient="horizontal").pack(fill="x", padx=5, pady=10)

    def update_visualization(self):
        """Update the parking visualization with simplified error handling"""
        if not self.show_visualization.get():
            return

        try:
            # Check if we need to recreate the figure
            canvas_exists = hasattr(self, 'canvas') and self.canvas and self.canvas.get_tk_widget().winfo_exists()
            figure_exists = hasattr(self, 'fig') and self.fig is not None
            ax_exists = hasattr(self, 'ax') and self.ax is not None

            # Create or recreate visualization components if needed
            if not (canvas_exists and figure_exists and ax_exists):
                if hasattr(self, 'canvas') and self.canvas:
                    try:
                        self.canvas.get_tk_widget().destroy()
                    except Exception as e:
                        print(f"Error destroying canvas: {e}")
                        pass

                self.fig = plt.Figure(figsize=(10, 6), dpi=100)
                self.ax = self.fig.add_subplot(111)
                self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
                self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

            # Clear the figure for fresh drawing
            self.ax.clear()

            # Get parking data
            parking_data = {}
            if hasattr(self.app, 'parking_manager') and hasattr(self.app.parking_manager, 'parking_data'):
                parking_data = self.app.parking_manager.parking_data.copy()

            # Update allocated vehicles
            for vehicle_id, space_id in self.allocated_vehicles.items():
                if space_id in parking_data:
                    parking_data[space_id]['occupied'] = True
                    parking_data[space_id]['vehicle_id'] = vehicle_id

            # If no posList or parking data, show message
            if (not hasattr(self.app, 'posList') or not self.app.posList or
                    not parking_data):
                self.ax.text(0.5, 0.5, "No parking data available",
                             ha='center', va='center', fontsize=14)
                self.canvas.draw()
                return

            # Calculate statistics
            individual_spaces = {id: data for id, data in parking_data.items()
                                 if not data.get('is_group', False)}
            group_spaces = {id: data for id, data in parking_data.items()
                            if data.get('is_group', False)}

            free_count = sum(1 for data in individual_spaces.values()
                             if not data.get('occupied', True))
            total = len(individual_spaces)

            # Set up grid layout
            space_w = 80
            space_h = 50
            margin = 20
            cols = max(3, min(8, int(np.sqrt(total * 2))))  # More reasonable column count

            # Calculate rows needed for individual spaces and groups
            individual_rows = (len(individual_spaces) + cols - 1) // cols
            group_rows = (len(group_spaces) + (cols // 2) - 1) // (cols // 2) if group_spaces else 0

            # Set plot area - make sure it's large enough
            plot_width = cols * space_w + 2 * margin
            plot_height = (individual_rows + group_rows + 3) * space_h + 3 * margin

            self.ax.set_xlim(0, plot_width)
            self.ax.set_ylim(0, plot_height)

            # Draw section headers
            current_y = plot_height - margin

            # Draw individual spaces section header
            self.ax.text(margin, current_y, "INDIVIDUAL SPACES",
                         fontsize=14, fontweight='bold')
            current_y -= space_h

            # Draw individual spaces
            space_i = 0
            for idx, pos in enumerate(self.app.posList):
                if not isinstance(pos, tuple) or len(pos) != 4:
                    continue

                x, y, w, h = pos

                # Create space ID
                section = "A" if x < getattr(self.app, 'image_width', 1000) / 2 else "B"
                section += "1" if y < getattr(self.app, 'image_height', 1000) / 2 else "2"
                space_id = f"S{idx + 1}-{section}"

                # Skip if part of a group
                is_in_group = False
                if space_id in individual_spaces:
                    is_in_group = individual_spaces[space_id].get('in_group', False)

                if is_in_group:
                    continue

                # Calculate position in grid
                row = space_i // cols
                col = space_i % cols

                x_pos = margin + col * space_w
                y_pos = current_y - (row + 1) * space_h

                # Get occupancy
                is_occupied = True
                vehicle_id = None
                if space_id in parking_data:
                    is_occupied = parking_data[space_id].get('occupied', True)
                    vehicle_id = parking_data[space_id].get('vehicle_id')

                # Set color
                color = 'red' if is_occupied else 'green'

                # Draw rectangle
                rect = plt.Rectangle((x_pos, y_pos), space_w - 10, space_h - 5,
                                     linewidth=2, edgecolor='black',
                                     facecolor=color, alpha=0.6)
                self.ax.add_patch(rect)

                # Add space ID
                self.ax.text(x_pos + 5, y_pos + space_h - 15, space_id,
                             fontsize=8, weight='bold', color='white')

                # Add vehicle ID if occupied
                if is_occupied and vehicle_id:
                    self.ax.text(x_pos + 5, y_pos + 10, f"V: {vehicle_id}",
                                 fontsize=8, color='white')

                space_i += 1

            # Update current y position after individual spaces
            current_y -= (individual_rows + 1) * space_h

            # Draw separator
            if group_spaces:
                self.ax.axhline(y=current_y + space_h / 2, color='gray',
                                linestyle='-', linewidth=1)

                # Draw group section header
                current_y -= space_h
                self.ax.text(margin, current_y, "GROUP SPACES",
                             fontsize=14, fontweight='bold')
                current_y -= space_h

                # Draw groups - with larger boxes
                group_i = 0
                group_cols = cols // 2  # Fewer columns for groups
                group_w = space_w * 1.8
                group_h = space_h * 1.5

                for group_id, group_data in group_spaces.items():
                    # Calculate position
                    row = group_i // group_cols
                    col = group_i % group_cols

                    x_pos = margin + col * group_w * 1.1  # Add extra spacing between groups
                    y_pos = current_y - (row + 1) * group_h

                    # Get group details
                    is_occupied = group_data.get('occupied', False)
                    member_spaces = group_data.get('member_spaces', [])

                    # Count occupied members
                    occupied_count = 0
                    for member_idx in member_spaces:
                        if member_idx < len(self.app.posList):
                            member_x, member_y, member_w, member_h = self.app.posList[member_idx]
                            member_section = "A" if member_x < getattr(self.app, 'image_width', 1000) / 2 else "B"
                            member_section += "1" if member_y < getattr(self.app, 'image_height', 1000) / 2 else "2"
                            member_id = f"S{member_idx + 1}-{member_section}"

                            if member_id in parking_data and parking_data[member_id].get('occupied', False):
                                occupied_count += 1

                    # Determine color
                    if is_occupied or occupied_count == len(member_spaces):
                        color = 'red'  # Fully occupied
                    elif occupied_count > 0:
                        color = 'orange'  # Partially occupied
                    else:
                        color = 'green'  # Free

                    # Draw group box
                    rect = plt.Rectangle((x_pos, y_pos), group_w - 10, group_h - 5,
                                         linewidth=3, edgecolor='blue',
                                         facecolor=color, alpha=0.5)
                    self.ax.add_patch(rect)

                    # Add group title
                    self.ax.text(x_pos + 5, y_pos + group_h - 20, f"{group_id}",
                                 fontsize=12, fontweight='bold', color='black')

                    # Add occupancy info
                    self.ax.text(x_pos + 5, y_pos + group_h - 40,
                                 f"({occupied_count}/{len(member_spaces)} occupied)",
                                 fontsize=9, color='black')

                    # Add small indicators for member spaces
                    mini_w = (group_w - 30) / min(4, max(1, len(member_spaces)))
                    mini_h = mini_w * 0.6

                    for idx, member_idx in enumerate(member_spaces):
                        if member_idx < len(self.app.posList):
                            # Calculate position for mini-space
                            mini_row = idx // 4
                            mini_col = idx % 4

                            mini_x = x_pos + 5 + mini_col * mini_w + 2
                            mini_y = y_pos + 5 + mini_row * mini_h + 2

                            # Check if occupied
                            member_x, member_y, member_w, member_h = self.app.posList[member_idx]
                            member_section = "A" if member_x < getattr(self.app, 'image_width', 1000) / 2 else "B"
                            member_section += "1" if member_y < getattr(self.app, 'image_height', 1000) / 2 else "2"
                            member_id = f"S{member_idx + 1}-{member_section}"

                            is_mini_occupied = False
                            if member_id in parking_data:
                                is_mini_occupied = parking_data[member_id].get('occupied', False)

                            # Draw mini space
                            mini_color = 'red' if is_mini_occupied else 'green'
                            mini_rect = plt.Rectangle((mini_x, mini_y),
                                                      mini_w - 4, mini_h - 4,
                                                      linewidth=1, edgecolor='black',
                                                      facecolor=mini_color, alpha=0.8)
                            self.ax.add_patch(mini_rect)

                            # Number the mini space
                            self.ax.text(mini_x + 2, mini_y + 2, f"{member_idx + 1}",
                                         fontsize=6, color='white')

                    group_i += 1

            # Add legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.6),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.6),
                plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.6),
            ]
            legend_labels = ['Free', 'Occupied', 'Partially Occupied']

            self.ax.legend(legend_elements, legend_labels, loc='upper right')

            # Add title with stats
            self.ax.set_title(f"Parking Status: {free_count}/{total} Available, {len(group_spaces)} Groups")

            # Remove axis ticks
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            # Add timestamp
            current_time = datetime.now().strftime('%H:%M:%S')
            self.ax.text(margin, margin / 2, f"Updated: {current_time}", fontsize=8)

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            print(f"Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()

            # Simple error display
            try:
                self.ax.clear()
                self.ax.text(0.5, 0.5, f"Visualization Error: {str(e)}",
                             ha='center', va='center', fontsize=12, color='red')
                self.canvas.draw()
            except:
                pass

    def are_groups_opposite(self, group1, group2):
        """
        Determine if two groups are in opposite alignments
        This is a simplified version - you may need to adjust based on your specific criteria
        """
        try:
            # Get member spaces positions
            members1 = group1.get('member_spaces', [])
            members2 = group2.get('member_spaces', [])

            if not members1 or not members2 or not hasattr(self.app, 'posList'):
                return False

            # Get positions
            pos1 = []
            pos2 = []

            for idx in members1:
                if 0 <= idx < len(self.app.posList):
                    pos1.append(self.app.posList[idx])

            for idx in members2:
                if 0 <= idx < len(self.app.posList):
                    pos2.append(self.app.posList[idx])

            if not pos1 or not pos2:
                return False

            # Calculate bounding boxes
            min_x1 = min(p[0] for p in pos1)
            min_y1 = min(p[1] for p in pos1)
            max_x1 = max(p[0] + p[2] for p in pos1)
            max_y1 = max(p[1] + p[3] for p in pos1)

            min_x2 = min(p[0] for p in pos2)
            min_y2 = min(p[1] for p in pos2)
            max_x2 = max(p[0] + p[2] for p in pos2)
            max_y2 = max(p[1] + p[3] for p in pos2)

            # Group dimensions
            width1 = max_x1 - min_x1
            height1 = max_y1 - min_y1
            width2 = max_x2 - min_x2
            height2 = max_y2 - min_y2

            # Check if one is vertical and one is horizontal
            group1_is_vertical = height1 > width1
            group2_is_vertical = height2 > width2

            # They are opposite if one is vertical and one is horizontal
            return group1_is_vertical != group2_is_vertical

        except Exception as e:
            print(f"Error checking opposite groups: {str(e)}")
            return False

    # Keep all other existing methods as they were

    def update_statistics(self):
        """Update the statistics display"""
        try:
            # Get current parking data
            parking_data = {}
            if hasattr(self.app, 'parking_manager'):
                if hasattr(self.app.parking_manager, 'parking_data'):
                    parking_data = self.app.parking_manager.parking_data.copy()

            # Calculate statistics
            total_spaces = len(parking_data) if parking_data else 0
            free_spaces = sum(
                1 for data in parking_data.values() if data.get('occupied') == False) if parking_data else 0
            occupied_spaces = total_spaces - free_spaces

            # Calculate occupancy rate safely
            try:
                occupancy_rate = (occupied_spaces / total_spaces) * 100 if total_spaces > 0 else 0
            except (ZeroDivisionError, TypeError):
                occupancy_rate = 0
                print("Warning: Error calculating occupancy rate - using 0")

            # Count allocations
            allocations = len(self.allocated_vehicles)

            # Format statistics text
            stats_text = (
                f"Total Spaces: {total_spaces}\n"
                f"Free Spaces: {free_spaces}\n"
                f"Occupied Spaces: {occupied_spaces}\n"
                f"Occupancy Rate: {occupancy_rate:.1f}%\n"
                f"Active Allocations: {allocations}\n"
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )

            # Update text widget
            self.stats_text.config(state="normal")
            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("end", stats_text)
            self.stats_text.config(state="disabled")

        except Exception as e:
            print(f"Error updating statistics: {str(e)}")
            # Log the full stack trace for debugging
            import traceback
            traceback.print_exc()

    def allocate_new_vehicle(self):
        """Allocate a new vehicle to a parking space"""
        # Run in a separate thread to avoid blocking UI
        threading.Thread(target=self._allocate_vehicle_thread, daemon=True).start()

    def _allocate_vehicle_thread(self):
        """Thread-safe vehicle allocation"""
        try:
            # Get parking data from parking manager
            parking_data = {}
            if hasattr(self.app, 'parking_manager'):
                with threading.Lock():  # Use lock for thread safety
                    parking_data = self.app.parking_manager.parking_data.copy()

            if not parking_data:
                self.queue_function(lambda: messagebox.showerror(
                    "Error", "No parking data available. Setup parking spaces first."))
                return

            # Check if there are free spaces
            free_spaces = {space_id: data for space_id, data in parking_data.items()
                           if not data.get('occupied', True)}

            if not free_spaces:
                self.queue_function(lambda: messagebox.showinfo(
                    "No Free Spaces", "No free parking spaces available."))
                return

            # Get vehicle parameters (capture current values)
            vehicle_id = f"V{self.next_vehicle_id}"
            vehicle_size = self.vehicle_size.get()
            preferred_section = None if self.preferred_section.get() == "Any" else self.preferred_section.get()
            weight = self.load_balancing_weight.get()

            # Set load balancing weight
            if hasattr(self.allocation_engine, 'load_balancing_weight'):
                self.allocation_engine.load_balancing_weight = weight

            # Perform allocation
            best_space_id, score = self.allocation_engine.allocate_parking(
                parking_data, vehicle_size, preferred_section)

            if best_space_id:
                # Update parking data
                if best_space_id in parking_data:
                    # Update data in thread-safe manner
                    with threading.Lock():
                        if hasattr(self.app, 'parking_manager'):
                            self.app.parking_manager.parking_data[best_space_id]['occupied'] = True
                            self.app.parking_manager.parking_data[best_space_id]['vehicle_id'] = vehicle_id

                    # Store allocation
                    self.allocated_vehicles[vehicle_id] = best_space_id
                    self.next_vehicle_id += 1  # Increment counter

                    # Schedule UI updates in main thread - fixed queue_function calls
                    info_msg = f"Vehicle {vehicle_id} allocated to space {best_space_id}\nAllocation score: {score:.2f}"
                    self.queue_function(lambda: messagebox.showinfo("Allocation Success", info_msg))

                    # Update UI
                    self.queue_function(self.update_visualization)
                    self.queue_function(self.update_statistics)
                else:
                    error_msg = f"Space {best_space_id} not found in parking data."
                    self.queue_function(lambda: messagebox.showerror("Allocation Error", error_msg))
            else:
                self.queue_function(lambda: messagebox.showinfo(
                    "Allocation Failed", "Could not find a suitable parking space."))

        except Exception as e:
            error_msg = f"Allocation error: {str(e)}"
            self.queue_function(lambda: messagebox.showerror("Error", error_msg))

    def _perform_allocation(self, vehicle_size, preferred_section, weight):
        """Worker function to perform allocation without freezing UI"""
        try:
            # Get parking data
            parking_data = self.app.parking_manager.parking_data if hasattr(self.app, 'parking_manager') else {}

            # Check if there are free spaces
            free_spaces = {space_id: data for space_id, data in parking_data.items()
                           if not data.get('occupied', True)}

            if not free_spaces:
                # Use queue_function to show message in main thread
                self.queue_function(
                    lambda: messagebox.showinfo("No Free Spaces", "No free parking spaces available.")
                )
                return

            # Get vehicle parameters - use the passed parameters, not UI variables
            vehicle_id = f"V{self.next_vehicle_id}"

            # Set load balancing weight safely
            if hasattr(self.allocation_engine, 'load_balancing_weight'):
                self.allocation_engine.load_balancing_weight = weight

            # Perform allocation
            best_space_id, score = self.allocation_engine.allocate_parking(
                parking_data, vehicle_size, preferred_section)

            # Schedule UI updates and messages in the main thread
            if best_space_id:
                # Update data structures first
                if best_space_id in parking_data:
                    parking_data[best_space_id]['occupied'] = True
                    parking_data[best_space_id]['vehicle_id'] = vehicle_id

                    # Store allocation safely with mutex if needed
                    self.allocated_vehicles[vehicle_id] = best_space_id
                    # Update ID counter safely
                    self.next_vehicle_id += 1

                    # Schedule UI updates for main thread
                    self.queue_function(
                        lambda: messagebox.showinfo("Allocation Success",
                                                    f"Vehicle {vehicle_id} allocated to space {best_space_id}\n"
                                                    f"Allocation score: {score:.2f}")
                    )
                    self.queue_function(self.update_visualization)
                    self.queue_function(self.update_statistics)
                else:
                    self.queue_function(
                        lambda: messagebox.showerror("Allocation Error",
                                                     f"Space {best_space_id} not found in parking data.")
                    )
            else:
                self.queue_function(
                    lambda: messagebox.showinfo("Allocation Failed",
                                                "Could not find a suitable parking space.")
                )

        except Exception as e:
            # Safely show error in main thread
            error_msg = str(e)
            self.queue_function(
                lambda: messagebox.showerror("Error", f"Allocation error: {error_msg}")
            )

    def add_random_vehicle(self):
        """Add a random vehicle for simulation purposes"""
        # Generate random vehicle parameters
        vehicle_size = random.randint(1, 3)
        self.vehicle_size.set(vehicle_size)

        # Random section preference (80% chance of "Any")
        if random.random() < 0.8:
            self.preferred_section.set("Any")
        else:
            self.preferred_section.set(random.choice(["A", "B", "C", "D"]))

        # Allocate the vehicle
        self.allocate_new_vehicle()

    def remove_random_vehicle(self):
        """Remove a randomly selected vehicle"""
        # Run in a separate thread to avoid blocking UI
        threading.Thread(target=self._remove_vehicle_thread, daemon=True).start()

    def _remove_vehicle_thread(self):
        """Thread-safe vehicle removal"""
        try:
            if not self.allocated_vehicles:
                self.queue_function(lambda: messagebox.showinfo(
                    "No Vehicles", "No vehicles are currently allocated."))
                return

            # Select a random vehicle
            vehicle_id = random.choice(list(self.allocated_vehicles.keys()))
            space_id = self.allocated_vehicles[vehicle_id]

            # Update parking data in thread-safe manner
            with threading.Lock():
                if hasattr(self.app, 'parking_manager'):
                    if space_id in self.app.parking_manager.parking_data:
                        self.app.parking_manager.parking_data[space_id]['occupied'] = False
                        self.app.parking_manager.parking_data[space_id]['vehicle_id'] = None

            # Remove from allocated vehicles
            del self.allocated_vehicles[vehicle_id]

            # Update UI in main thread - fixed queue_function calls
            self.queue_function(self.update_visualization)
            self.queue_function(self.update_statistics)

            message = f"Vehicle {vehicle_id} removed from space {space_id}."
            self.queue_function(lambda: messagebox.showinfo("Vehicle Removed", message))

        except Exception as e:
            error_msg = f"Error removing vehicle: {str(e)}"
            self.queue_function(lambda: messagebox.showerror("Error", error_msg))

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        confirm = messagebox.askyesno("Confirm Reset", "Are you sure you want to reset the simulation?")
        if not confirm:
            return

        # Reset parking data
        parking_data = self.app.parking_manager.parking_data if hasattr(self.app, 'parking_manager') else {}
        for space_id in parking_data:
            parking_data[space_id]['occupied'] = False
            parking_data[space_id]['vehicle_id'] = None

        # Clear allocated vehicles
        self.allocated_vehicles = {}
        self.next_vehicle_id = 1

        # Update UI
        self.update_visualization()
        self.update_statistics()

        messagebox.showinfo("Simulation Reset", "Simulation has been reset.")

    def update_loop(self):
        """Background thread for periodic updates with improved error handling"""
        update_interval = 5  # Update visualization every 5 seconds
        visualization_counter = 0

        while self.running:
            # Process any UI update events in queue
            try:
                while not self.update_queue.empty():
                    try:
                        func, args = self.update_queue.get_nowait()
                        if callable(func):
                            try:
                                func(*args)  # This passes the unpacked tuple as arguments
                            except Exception as e:
                                print(f"Error executing queued function: {str(e)}")
                    except queue.Empty:
                        break
                    except Exception as e:
                        print(f"Error processing queue item: {str(e)}")
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
                pass

            try:
                # Perform auto-allocation if enabled
                if hasattr(self, 'auto_allocation_enabled') and self.auto_allocation_enabled.get():
                    # Add a vehicle occasionally
                    if random.random() < 0.1:  # 10% chance each cycle
                        self.queue_function(self.add_random_vehicle)  # No args needed

                    # Remove a vehicle occasionally
                    if hasattr(self, 'allocated_vehicles') and self.allocated_vehicles and random.random() < 0.05:
                        self.queue_function(self.remove_random_vehicle)  # No args needed

                # Update visualization only every X seconds
                visualization_counter += 1
                if visualization_counter >= update_interval:
                    # Schedule a visualization update
                    self.queue_function(self.update_visualization)  # No args needed
                    # Statistics every 10 seconds
                    if visualization_counter >= update_interval * 2:
                        self.queue_function(self.update_statistics)  # No args needed
                        visualization_counter = 0
            except Exception as e:
                print(f"Error in update loop scheduling: {str(e)}")
                visualization_counter = 0  # Reset counter on error

            # Sleep between updates (1 second to keep other operations responsive)
            time.sleep(1)

    def queue_function(self, func, *args):
        """Queue a function to be executed in the main thread with error checking"""
        try:
            if not hasattr(self, 'update_queue'):
                self.update_queue = queue.Queue()
            if callable(func):
                # If no args provided, use an empty tuple instead
                if not args:
                    args = ()
                self.update_queue.put((func, args))
            else:
                print(f"Warning: Attempted to queue non-callable object: {func}")
        except Exception as e:
            print(f"Error queuing function: {str(e)}")

    def on_tab_selected(self):
        """Called when this tab is selected"""
        print("Parking allocation tab selected")
        self.ensure_parking_data()
        self.update_visualization()
        self.update_statistics()

    def ensure_parking_data(self):
        """Ensure we have proper parking data to visualize"""
        if hasattr(self.app, 'parking_manager') and not hasattr(self.app.parking_manager, 'parking_data'):
            self.app.parking_manager.parking_data = {}

            # Initialize from positions
            if hasattr(self.app, 'posList') and self.app.posList:
                for i, (x, y, w, h) in enumerate(self.app.posList):
                    space_id = f"S{i + 1}"
                    section = "A" if x < self.app.image_width / 2 else "B"
                    section += "1" if y < self.app.image_height / 2 else "2"
                    full_space_id = f"{space_id}-{section}"

                    self.app.parking_manager.parking_data[full_space_id] = {
                        'position': (x, y, w, h),
                        'occupied': True,  # Default to occupied
                        'vehicle_id': None,
                        'last_state_change': datetime.now(),
                        'distance_to_entrance': x + y,
                        'section': section
                    }

    def cleanup(self):
        """Clean up resources before closing"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

    # Add this method to your allocation class (if it exists)

    def allocate_group(self, vehicle_id, vehicle_size):
        """Allocate a vehicle to a group of spaces"""
        # Get all free groups
        free_groups = {space_id: data for space_id, data in self.app.parking_manager.parking_data.items()
                       if data.get('is_group', False) and not data['occupied']}

        if not free_groups:
            return None, 0  # No free group

        # Find the best group based on size matching
        best_group = None
        best_score = 0

        for group_id, data in free_groups.items():
            # Get number of spaces in this group
            member_count = len(data.get('member_spaces', []))

            # Score based on size matching (higher is better)
            size_match = 1 - abs(member_count - vehicle_size) / max(member_count, vehicle_size)

            # Consider distance to entrance
            distance_score = 1 / (1 + data.get('distance_to_entrance', 0) / 1000)

            # Combine scores
            score = (size_match * 0.7) + (distance_score * 0.3)

            if score > best_score:
                best_score = score
                best_group = group_id

        return best_group, best_score

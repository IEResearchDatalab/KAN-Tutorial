# Interactive 1D B-Spline Function Visualizer
#
# This script helps to understand the fundamentals of 1D B-spline functions
# by providing an interactive plot. You can drag the control points vertically
# to change the spline's shape and edit the knot vector in the text box.
#
# Requirements:
# pip install numpy matplotlib
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# --- Core B-Spline Calculation ---

def cox_de_boor(u, i, p, knots):
    """
    Computes the i-th B-spline basis function of degree p at parameter u.
    (This function remains unchanged)
    """
    # Base case for recursion: degree 0
    if p == 0:
        is_in_span = knots[i] <= u < knots[i + 1]
        is_endpoint_case = (u == knots[-1]) and (knots[i+1] == knots[-1])
        if knots[i] < knots[i+1] and (is_in_span or is_endpoint_case):
            return 1.0
        else:
            return 0.0
    # Recursive Step
    term1 = 0.0
    denominator1 = knots[i + p] - knots[i]
    if denominator1 > 0:
        term1 = ((u - knots[i]) / denominator1) * cox_de_boor(u, i, p - 1, knots)
    term2 = 0.0
    denominator2 = knots[i + p + 1] - knots[i + 1]
    if denominator2 > 0:
        term2 = ((knots[i + p + 1] - u) / denominator2) * cox_de_boor(u, i + 1, p - 1, knots)
    return term1 + term2


def bspline_function(y_vals, degree, num_eval_points=200, custom_knots=None):
    """
    Calculates the points on a 1D B-spline function S(u).

    The function is a weighted sum of the control point y-values.
    Formula: S(u) = sum_{i=0 to n} y_i * N_i,p(u)

    Args:
        y_vals (np.array): An array of control point y-values, shape (n+1,).
        degree (int): The degree 'p' of the B-spline.
        num_eval_points (int): The number of points to calculate for a smooth plot.
        custom_knots (np.array, optional): A user-provided knot vector. If None,
                                           a clamped knot vector is generated.
    Returns:
        tuple: A tuple containing:
            - np.array: The parameter values 'u' for the x-axis.
            - np.array: The calculated S(u) values for the y-axis.
            - np.array: The knot vector used for the calculation.
    """
    num_control_points = len(y_vals)
    n = num_control_points - 1
    
    if custom_knots is not None:
        knots = custom_knots
    else:
        # Generate a clamped (open) knot vector
        num_knots = n + degree + 2
        knots = np.zeros(num_knots)
        internal_knot_vals = np.arange(1, n - degree + 1)
        knots[degree+1 : n+1] = internal_knot_vals
        knots[n+1:] = n - degree + 1
        # Normalize to [0, 1] range
        if np.max(knots) > 0:
            knots /= np.max(knots)

    # The valid parameter range for 'u'
    u_min = knots[degree]
    u_max = knots[n + 1]
    u_values = np.linspace(u_min, u_max, num_eval_points)

    curve_values = np.zeros(num_eval_points)
    for j, u in enumerate(u_values):
        value = 0.0
        for i in range(num_control_points):
            basis_val = cox_de_boor(u, i, degree, knots)
            value += y_vals[i] * basis_val
        curve_values[j] = value

    return u_values, curve_values, knots


# --- Interactive Plotting Class ---

class BSplineInteractor:
    def __init__(self, ax, initial_y_vals, degree):
        self.ax = ax
        self.control_point_y_vals = np.array(initial_y_vals, dtype=float)
        self.degree = degree
        
        self.knots = []
        self.control_point_x_pos = [] # Will be determined by knot range
        self.use_user_knots = False
        self.user_knots = None
        
        self.epsilon = 0.5 # Click tolerance for y-axis
        self._drag_index = None

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def draw(self):
        self.ax.clear()
        
        # Determine which knots to use
        knots_to_use = self.user_knots if self.use_user_knots else None
        
        # Recalculate the B-spline curve
        u_vals, curve_vals, self.knots = bspline_function(
            self.control_point_y_vals, self.degree, custom_knots=knots_to_use
        )

        # The x-positions of control points are spaced over the valid parameter range
        u_min, u_max = self.knots[self.degree], self.knots[len(self.control_point_y_vals)]
        self.control_point_x_pos = np.linspace(u_min, u_max, len(self.control_point_y_vals))

        # Draw the B-spline function
        self.ax.plot(u_vals, curve_vals, 'b-', lw=2, label='B-spline Function S(u)')

        # Draw the control polygon (y-values vs their determined x-positions)
        self.ax.plot(self.control_point_x_pos, self.control_point_y_vals, 'r:o', lw=1, markersize=8, label='Control Points')

        # Visualize Knot Vector on the x-axis
        unique_knots = np.unique(self.knots)
        self.ax.plot(unique_knots, np.zeros_like(unique_knots), 'gx', markersize=10, mew=2, label='Knot Positions')

        # Set plot appearance
        self.ax.set_title(f"Interactive 1D B-spline (Degree p={self.degree})")
        self.ax.set_xlabel("Parameter u")
        self.ax.set_ylabel("S(u)")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_ylim(-5, 30)
        self.ax.figure.canvas.draw_idle()

    def submit_knots(self, text):
        try:
            new_knots = np.array([float(k.strip()) for k in text.strip('[]').split(',')])
            num_cp = len(self.control_point_y_vals)
            expected_len = num_cp + self.degree + 1
            
            if len(new_knots) != expected_len:
                print(f"Error: Invalid knot vector length. Expected {expected_len}, got {len(new_knots)}.")
                return
            if not np.all(np.diff(new_knots) >= -1e-9): # Allow for small float inaccuracies
                print("Error: Knot vector must be non-decreasing.")
                return
            
            self.user_knots = new_knots
            self.use_user_knots = True
            self.draw()
            print("Knot vector updated successfully.")
        except ValueError:
            print("Error: Could not parse knot vector.")

    def get_index_under_point(self, event):
        if event.xdata is None or event.ydata is None: return None
        # Check distance in both axes, but primarily x
        x_dist = np.abs(self.control_point_x_pos - event.xdata)
        y_dist = np.abs(self.control_point_y_vals - event.ydata)
        # Convert x-distance to data units to make it comparable to y
        x_tolerance = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.05
        
        for i, (xd, yd) in enumerate(zip(x_dist, y_dist)):
            if xd < x_tolerance and yd < self.epsilon:
                return i
        return None

    def on_press(self, event):
        if event.inaxes != self.ax: return
        self._drag_index = self.get_index_under_point(event)

    def on_release(self, event):
        self._drag_index = None

    def on_motion(self, event):
        if self._drag_index is None or event.inaxes != self.ax: return
        # Update the y-value of the dragged control point (x is fixed)
        self.control_point_y_vals[self._drag_index] = event.ydata
        self.draw()

# --- Main Execution ---
if __name__ == '__main__':
    SPLINE_DEGREE = 3
    INITIAL_CONTROL_POINT_Y_VALS = [5, 20, 25, 15, 8, 5]

    fig, ax = plt.subplots(figsize=(12, 8))
    # Make space at the bottom for the TextBox
    plt.subplots_adjust(bottom=0.2)

    interactor = BSplineInteractor(ax, INITIAL_CONTROL_POINT_Y_VALS, SPLINE_DEGREE)
    
    # Create the knot vector text box
    ax_box = plt.axes([0.15, 0.05, 0.75, 0.075])
    # Initial draw to generate the first knot vector
    interactor.draw() 
    knot_str = ", ".join([f"{k:.2f}" for k in interactor.knots])
    text_box = TextBox(ax_box, "Knot Vector", initial=knot_str)
    text_box.on_submit(interactor.submit_knots)

    plt.show()


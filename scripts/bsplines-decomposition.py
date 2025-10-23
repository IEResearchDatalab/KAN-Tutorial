# B-Spline Basis Function Visualizer
#
# This script provides an interactive plot to deconstruct a single
# B-spline basis function and show how all basis functions sum up
# to form the final spline curve.
#
# Requirements:
# pip install numpy matplotlib
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# --- Core B-Spline Calculation ---

def cox_de_boor(u, i, p, knots):
    """
    Computes the i-th B-spline basis function of degree p at parameter u.
    """
    if p == 0:
        is_in_span = knots[i] <= u < knots[i + 1]
        is_endpoint_case = (u == knots[-1]) and (knots[i+1] == knots[-1])
        if knots[i] < knots[i+1] and (is_in_span or is_endpoint_case):
            return 1.0
        else:
            return 0.0
    term1, term2 = 0.0, 0.0
    denominator1 = knots[i + p] - knots[i]
    if denominator1 > 0:
        term1 = ((u - knots[i]) / denominator1) * cox_de_boor(u, i, p - 1, knots)
    denominator2 = knots[i + p + 1] - knots[i + 1]
    if denominator2 > 0:
        term2 = ((knots[i + p + 1] - u) / denominator2) * cox_de_boor(u, i + 1, p - 1, knots)
    return term1 + term2

def cox_de_boor_tree(u, i, p, knots):
    """
    Computes the i-th B-spline basis function and returns the full recursion tree.
    """
    node = {'name': f'N({i},{p})', 'value': 0.0, 'children': [], 'weights': [0.0, 0.0]}
    if p == 0:
        node['value'] = cox_de_boor(u, i, 0, knots)
        return node
    
    child1_node = cox_de_boor_tree(u, i, p - 1, knots)
    child2_node = cox_de_boor_tree(u, i + 1, p - 1, knots)
    
    term1_val, w1 = 0.0, 0.0
    denominator1 = knots[i + p] - knots[i]
    if denominator1 > 0:
        w1 = (u - knots[i]) / denominator1
        term1_val = w1 * child1_node['value']
    
    term2_val, w2 = 0.0, 0.0
    denominator2 = knots[i + p + 1] - knots[i + 1]
    if denominator2 > 0:
        w2 = (knots[i + p + 1] - u) / denominator2
        term2_val = w2 * child2_node['value']

    node['value'] = term1_val + term2_val
    node['children'] = [child1_node, child2_node]
    node['weights'] = [w1, w2]
    return node

def draw_tree(ax, node, x, y, width, height_step):
    """Recursively draws the computation tree."""
    node_text = f"{node['name']}\n{node['value']:.3f}"
    ax.text(x, y, node_text, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", fc="#cceeff", ec="b", lw=1))

    if not node['children']: return

    x_child1, x_child2 = x - width / 4, x + width / 4
    y_child = y - height_step

    ax.plot([x, x_child1], [y, y_child], 'k-'); ax.plot([x, x_child2], [y, y_child], 'k-')
    ax.text((x + x_child1) / 2, (y + y_child) / 2 + 0.05, f'× {node["weights"][0]:.2f}', ha='center', va='center', color='green', fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.text((x + x_child2) / 2, (y + y_child) / 2 + 0.05, f'× {node["weights"][1]:.2f}', ha='center', va='center', color='red', fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.7))

    draw_tree(ax, node['children'][0], x_child1, y_child, width / 2, height_step)
    draw_tree(ax, node['children'][1], x_child2, y_child, width / 2, height_step)

# --- Interactive Plotting Class ---

class BasisVisualizer:
    def __init__(self, fig, max_degree, num_control_points):
        self.fig = fig
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 3])
        self.ax_spline = fig.add_subplot(gs[0])
        self.ax_func = fig.add_subplot(gs[1], sharex=self.ax_spline)
        self.ax_tree = fig.add_subplot(gs[2])

        self.max_degree, self.num_control_points = max_degree, num_control_points
        self.n = self.num_control_points - 1

        self.i_index, self.p_degree, self.u_val = 0, 3, 0.5
        self.knots = []
        self.control_points = np.array([5.0, -5.0, 10.0, 2.0, 15.0, 8.0])
        self.generate_knots()

        # Spline plot elements
        self.spline_line, = self.ax_spline.plot([], [], 'b-', lw=3, label='Final Spline S(u)')
        self.control_point_markers, = self.ax_spline.plot([], [], 'ro', markersize=8)
        self.basis_lines = [self.ax_spline.plot([], [], alpha=0.5, lw=1)[0] for _ in range(num_control_points)]
        self.v_line_spline = self.ax_spline.axvline(self.u_val, color='gray', linestyle='--')
        self.summation_text = self.ax_spline.text(0.02, 0.98, '', transform=self.ax_spline.transAxes,
                                                  ha='left', va='top', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
        self.ax_spline.set_title('Final B-Spline Curve and All Basis Functions')
        self.ax_spline.set_ylabel('S(u)')
        self.ax_spline.grid(True)
        self.ax_spline.legend(loc='upper right', fontsize='small')

        # Basis function plot elements
        self.line_main, = self.ax_func.plot([], [], 'b-', lw=3, label=r'$N_{i,p}(u)$')
        self.line_child1, = self.ax_func.plot([], [], 'g:', lw=1.5, label=r'$N_{i,p-1}(u)$')
        self.line_child2, = self.ax_func.plot([], [], 'r:', lw=1.5, label=r'$N_{i+1,p-1}(u)$')
        self.v_line_func = self.ax_func.axvline(self.u_val, color='gray', linestyle='--')
        self.ax_func.set_ylim(-0.1, 1.2)
        self.ax_func.set_xlabel("Parameter u")
        self.ax_func.set_ylabel("Value")
        self.ax_func.grid(True)
        self.ax_func.legend(loc='upper left', fontsize='small')

        # Control point dragging
        self._drag_data = {'index': -1, 'press_y': 0}
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def generate_knots(self):
        num_knots = self.n + self.max_degree + 2
        self.knots = np.zeros(num_knots)
        internal_knots = np.arange(1, self.n - self.max_degree + 1)
        self.knots[self.max_degree+1 : self.n+1] = internal_knots
        self.knots[self.n+1:] = self.n - self.max_degree + 1
        if np.max(self.knots) > 0:
            self.knots /= np.max(self.knots)

    def update(self, val=None):
        u_values = np.linspace(self.knots[0], self.knots[-1], 500)
        all_basis_vals_plot = np.array([[cox_de_boor(u, i, self.p_degree, self.knots) for u in u_values] for i in range(self.num_control_points)])
        
        # --- Update Top Plot (Spline View) ---
        spline_vals = np.dot(self.control_points, all_basis_vals_plot)
        self.spline_line.set_data(u_values, spline_vals)
        cp_x_pos = np.linspace(self.knots[self.p_degree], self.knots[self.n+1], self.num_control_points)
        self.control_point_markers.set_data(cp_x_pos, self.control_points)
        for i, line in enumerate(self.basis_lines):
             line.set_data(u_values, all_basis_vals_plot[i, :] * self.control_points[i]) # Show weighted basis
        self.v_line_spline.set_xdata([self.u_val])
        
        # Update summation text
        basis_vals_at_u = [cox_de_boor(self.u_val, i, self.p_degree, self.knots) for i in range(self.num_control_points)]
        final_spline_val = np.dot(self.control_points, basis_vals_at_u)
        
        sum_str1 = f"$S(u={self.u_val:.2f}) = \sum C_i \cdot N_{{i,p}}(u)$\n"
        terms = [f"{cp:.1f} $\cdot$ {bval:.2f}" for cp, bval in zip(self.control_points, basis_vals_at_u) if abs(bval) > 1e-9]
        if not terms: # Handle case where all basis vals are zero
            term_str = "0.0"
        elif len(terms) > 3:
            term_str = " + ".join(terms[:3]) + " + \n      " + " + ".join(terms[3:])
        else:
            term_str = " + ".join(terms)
            
        sum_str2 = " = " + term_str
        sum_str3 = f"\n = {final_spline_val:.3f}"
        self.summation_text.set_text(sum_str1 + sum_str2 + sum_str3)

        self.ax_spline.relim(); self.ax_spline.autoscale_view()

        # --- Update Middle Plot (Function View) ---
        N_child1 = [cox_de_boor(u, self.i_index, self.p_degree - 1, self.knots) if self.p_degree > 0 else 0 for u in u_values]
        N_child2 = [cox_de_boor(u, self.i_index + 1, self.p_degree - 1, self.knots) if self.p_degree > 0 else 0 for u in u_values]
        self.line_main.set_data(u_values, all_basis_vals_plot[self.i_index])
        self.line_child1.set_data(u_values, N_child1)
        self.line_child2.set_data(u_values, N_child2)
        self.v_line_func.set_xdata([self.u_val])
        self.ax_func.set_title(f'Decomposition of Basis Function $N_{{{self.i_index},{self.p_degree}}}(u)$')
        self.ax_func.set_xlim(self.knots[0], self.knots[-1])
        
        # --- Update Bottom Plot (Tree View) ---
        self.ax_tree.clear()
        self.ax_tree.set_title(f'Recursion Tree for $N_{{{self.i_index},{self.p_degree}}}$ at u = {self.u_val:.3f}')
        self.ax_tree.set_xticks([]); self.ax_tree.set_yticks([])
        tree = cox_de_boor_tree(self.u_val, self.i_index, self.p_degree, self.knots)
        # Use a fixed height step to prevent the tree from getting too tight
        draw_tree(self.ax_tree, tree, x=0.5, y=0.9, width=1.0, height_step=0.3)
        self.ax_tree.set_ylim(0, 1)

        self.fig.canvas.draw_idle()

    def submit_knots(self, text):
        try:
            new_knots = np.array([float(k.strip()) for k in text.strip('[]').split(',')])
            expected_len = self.n + self.max_degree + 2
            if len(new_knots) != expected_len:
                print(f"Error: Invalid knot vector length. Expected {expected_len}, got {len(new_knots)}.")
                return
            if not np.all(np.diff(new_knots) >= -1e-9):
                print("Error: Knot vector must be non-decreasing.")
                return
            self.knots = new_knots
            self.update()
            print("Knot vector updated successfully.")
        except ValueError:
            print("Error: Could not parse knot vector.")


    def on_press(self, event):
        if event.inaxes != self.ax_spline: return
        x, y = event.xdata, event.ydata
        dists = np.sqrt((self.control_point_markers.get_xdata() - x)**2 + (self.control_point_markers.get_ydata() - y)**2)
        if np.min(dists) < 0.5: # Click tolerance
            self._drag_data['index'] = np.argmin(dists)
            self._drag_data['press_y'] = y
    
    def on_motion(self, event):
        if self._drag_data['index'] == -1 or event.inaxes != self.ax_spline: return
        y = event.ydata
        self.control_points[self._drag_data['index']] = y
        self.update()
        
    def on_release(self, event):
        self._drag_data['index'] = -1

# --- Main Execution ---
if __name__ == '__main__':
    MAX_DEGREE = 3
    NUM_CONTROL_POINTS = 6

    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(bottom=0.25, hspace=0.6)

    visualizer = BasisVisualizer(fig, MAX_DEGREE, NUM_CONTROL_POINTS)

    ax_u = plt.axes([0.25, 0.15, 0.65, 0.02])
    ax_degree = plt.axes([0.25, 0.12, 0.65, 0.02])
    ax_index = plt.axes([0.25, 0.09, 0.65, 0.02])
    ax_knots = plt.axes([0.15, 0.02, 0.75, 0.04])

    slider_u = Slider(ax_u, 'u value', 0.0, 1.0, valinit=0.5)
    slider_degree = Slider(ax_degree, 'Degree (p)', 0, MAX_DEGREE, valinit=3, valstep=1)
    slider_index = Slider(ax_index, 'Index (i)', 0, NUM_CONTROL_POINTS - 1, valinit=0, valstep=1)

    def on_slider_change(val):
        visualizer.p_degree = int(slider_degree.val)
        visualizer.i_index = int(slider_index.val)
        visualizer.u_val = slider_u.val
        visualizer.update()
    
    slider_u.on_changed(on_slider_change); slider_degree.on_changed(on_slider_change); slider_index.on_changed(on_slider_change)
    knot_str = ", ".join([f"{k:.2f}" for k in visualizer.knots])
    text_box_knots = TextBox(ax_knots, "Knot Vector", initial=knot_str)
    text_box_knots.on_submit(visualizer.submit_knots)
    
    visualizer.update()
    plt.show()


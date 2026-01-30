# Develpoed  by Dr.Wahab , 1.30.2026
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = 'opensees_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# --- CONSISTENT UNITS: N, mm ---
H_col = 1800.0          # mm
L_fnd = 1200.0          # mm
P_axial = 500000.0      # N (500 kN)

# --- Nodes ---
ops.node(1, 0.0, 0.0)           
ops.node(2, 0.0, L_fnd)         
ops.node(3, 0.0, L_fnd + H_col) 
ops.fix(1, 1, 1, 1) 

# Store original node coordinates
node_coords = {}
for node_tag in [1, 2, 3]:
    node_coords[node_tag] = [ops.nodeCoord(node_tag, 1), ops.nodeCoord(node_tag, 2)]

# --- Materials (N, mm units) - IMPROVED FOR BETTER YIELDING ---
# Concrete Core (confined): f'c = 30 MPa = 30 N/mm²
# Using more ductile parameters
ops.uniaxialMaterial('Concrete02', 1, 
                     -40.0,      # fpc (confined strength - slightly higher)
                     -0.002,     # epsc0
                     -8.0,       # fpcu (residual strength for better post-peak)
                     -0.015,     # epsU (larger ultimate strain for ductility)
                     0.1,        # lambda
                     2.5,        # ft (small tensile strength)
                     500.0)      # Ets (tension softening)

# Concrete Cover (unconfined) - softer to promote yielding
ops.uniaxialMaterial('Concrete02', 3,
                     -30.0,      # fpc
                     -0.002,     # epsc0
                     0.0,        # fpcu (no residual for cover)
                     -0.005,     # epsU (earlier spalling)
                     0.1,        # lambda
                     2.5,        # ft
                     500.0)      # Ets

# Steel: REDUCED yield strength for easier yielding
# Using Fy=300 MPa with better calibrated hardening
ops.uniaxialMaterial('Steel02', 2, 
                     300.0,      # Fy (N/mm²)
                     200000.0,   # E (N/mm²)
                     0.01,       # b (reduced strain hardening for better hysteresis)
                     18.0,       # R0 (controls transition from elastic to plastic)
                     0.925,      # cR1
                     0.15)       # cR2

# --- Section: 400x400 Column (mm) - MORE REALISTIC REINFORCEMENT ---
b = 400.0
h = 400.0
cover = 40.0
core_b = b - 2*cover
core_h = h - 2*cover

ops.section('Fiber', 1)

# Core concrete (confined) - finer discretization
ops.patch('rect', 1, 25, 25, -core_b/2, -core_h/2, core_b/2, core_h/2)

# Cover concrete (4 patches)
ops.patch('rect', 3, 25, 5, -b/2, core_h/2, b/2, h/2)      # Top
ops.patch('rect', 3, 25, 5, -b/2, -h/2, b/2, -core_h/2)    # Bottom
ops.patch('rect', 3, 5, 25, -b/2, -core_h/2, -core_b/2, core_h/2)  # Left
ops.patch('rect', 3, 5, 25, core_b/2, -core_h/2, b/2, core_h/2)    # Right

# Reinforcement: 8-D20 bars (MORE REALISTIC LAYOUT)
# Bar diameter and area
bar_diameter = 20.0  # mm
A_bar = np.pi * (bar_diameter/2)**2  # ~314 mm²
rebar_dist = 160.0  # distance from center to rebar

# Corner bars (4 bars)
ops.fiber(-rebar_dist, -rebar_dist, A_bar, 2)  # Bottom-left
ops.fiber(rebar_dist, -rebar_dist, A_bar, 2)   # Bottom-right
ops.fiber(rebar_dist, rebar_dist, A_bar, 2)    # Top-right
ops.fiber(-rebar_dist, rebar_dist, A_bar, 2)   # Top-left

# Intermediate bars along edges (4 bars)
ops.fiber(0, -rebar_dist, A_bar, 2)   # Bottom-middle
ops.fiber(rebar_dist, 0, A_bar, 2)    # Right-middle
ops.fiber(0, rebar_dist, A_bar, 2)    # Top-middle
ops.fiber(-rebar_dist, 0, A_bar, 2)   # Left-middle

print(f"\nReinforcement: 8-D{bar_diameter} bars")
print(f"Reinforcement ratio: {8*A_bar/(b*h)*100:.2f}%")

# --- Calculate Plastic Hinge Length ---
fy = 300.0  # MPa
db = bar_diameter  # mm
Lp = 0.08 * H_col + 0.022 * fy * db  # Plastic hinge length
print(f"Plastic Hinge Length (Lp): {Lp:.1f} mm ({Lp/H_col*100:.1f}% of column height)")

# --- Additional nodes for plastic hinge modeling ---
ops.node(4, 0.0, L_fnd + Lp)

# --- Elements ---
ops.geomTransf('PDelta', 1)

# Foundation (Elastic)
E_conc = 25000.0  # MPa = 25000 N/mm²
I_fnd = 1.08e10   # mm⁴
A_fnd = 360000.0  # mm²
ops.element('elasticBeamColumn', 1, 1, 2, A_fnd, E_conc, I_fnd, 1)

# Column - Plastic Hinge Region (Force-Based with MANY integration points)
ops.beamIntegration('Lobatto', 1, 1, 5)  # 5 points in plastic hinge
ops.element('forceBeamColumn', 2, 2, 4, 1, 1)

# Column - Elastic Region (above plastic hinge)
I_col = (b * h**3) / 12  # mm⁴
A_col = b * h  # mm²
ops.element('elasticBeamColumn', 3, 4, 3, A_col, E_conc, I_col, 1)

print(f"Column divided into:")
print(f"  - Plastic hinge region: {Lp:.1f} mm (Node 2 to 4) - Fiber element")
print(f"  - Elastic region: {H_col-Lp:.1f} mm (Node 4 to 3) - Elastic element")

# Update node coordinates dictionary
node_coords[4] = [0.0, L_fnd + Lp]

# ============================================================================
# FUNCTION TO PLOT MODEL GEOMETRY
# ============================================================================
def plot_model_geometry():
    """Plot the undeformed model showing nodes, elements, and cross-section"""
    fig = plt.figure(figsize=(16, 6))
    
    # --- Plot 1: Model Elevation ---
    ax1 = fig.add_subplot(131)
    
    # Draw foundation element
    node1_coord = node_coords[1]
    node2_coord = node_coords[2]
    ax1.plot([node1_coord[0], node2_coord[0]], [node1_coord[1], node2_coord[1]], 
             'b-', linewidth=8, label='Foundation', alpha=0.7)
    
    # Draw plastic hinge region
    node4_coord = node_coords[4]
    ax1.plot([node2_coord[0], node4_coord[0]], [node2_coord[1], node4_coord[1]], 
             'r-', linewidth=6, label='Plastic Hinge Region', alpha=0.8)
    
    # Draw elastic column region
    node3_coord = node_coords[3]
    ax1.plot([node4_coord[0], node3_coord[0]], [node4_coord[1], node3_coord[1]], 
             'orange', linewidth=6, label='Elastic Column', alpha=0.7)
    
    # Draw nodes
    for node_tag, coord in node_coords.items():
        ax1.plot(coord[0], coord[1], 'ko', markersize=12)
        ax1.text(coord[0]+100, coord[1], f'Node {node_tag}', fontsize=9, fontweight='bold')
    
    # Highlight plastic hinge region
    ax1.axhspan(L_fnd, L_fnd + Lp, alpha=0.2, color='red', label='Plastic Hinge Zone')
    
    # Draw boundary condition
    support_size = 100
    ax1.plot([-support_size, support_size], [0, 0], 'k-', linewidth=4)
    for i in range(-3, 4):
        ax1.plot([i*support_size/3, (i-0.5)*support_size/3], [0, -support_size/2], 'k-', linewidth=2)
    
    # Add dimensions
    ax1.annotate('', xy=(300, L_fnd), xytext=(300, 0),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text(400, L_fnd/2, f'{L_fnd} mm', fontsize=10, color='green', fontweight='bold')
    
    ax1.annotate('', xy=(300, L_fnd+Lp), xytext=(300, L_fnd),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(400, L_fnd+Lp/2, f'Lp={Lp:.0f} mm', fontsize=10, color='red', fontweight='bold')
    
    ax1.annotate('', xy=(300, L_fnd+H_col), xytext=(300, L_fnd+Lp),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax1.text(400, L_fnd+Lp+(H_col-Lp)/2, f'{H_col-Lp:.0f} mm', fontsize=10, color='orange', fontweight='bold')
    
    # Add axial load
    ax1.arrow(0, L_fnd+H_col+200, 0, -150, head_width=100, head_length=50, 
             fc='purple', ec='purple', linewidth=2)
    ax1.text(150, L_fnd+H_col+200, f'P = {P_axial/1000} kN', 
            fontsize=10, color='purple', fontweight='bold')
    
    ax1.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Elevation with Plastic Hinge', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='datalim')
    
    # --- Plot 2: Column Cross-Section ---
    ax2 = fig.add_subplot(132)
    
    # Draw cover concrete (outer rectangle)
    cover_rect = Rectangle((-b/2, -h/2), b, h, linewidth=2, 
                          edgecolor='blue', facecolor='lightblue', alpha=0.5, label='Cover Concrete')
    ax2.add_patch(cover_rect)
    
    # Draw core concrete (inner rectangle)
    core_rect = Rectangle((-core_b/2, -core_h/2), core_b, core_h, linewidth=2, 
                         edgecolor='darkblue', facecolor='skyblue', alpha=0.7, label='Core Concrete')
    ax2.add_patch(core_rect)
    
    # Draw reinforcement bars
    bar_positions = [
        # Corner bars
        (-rebar_dist, -rebar_dist), (rebar_dist, -rebar_dist),
        (rebar_dist, rebar_dist), (-rebar_dist, rebar_dist),
        # Intermediate bars
        (0, -rebar_dist), (rebar_dist, 0), 
        (0, rebar_dist), (-rebar_dist, 0)
    ]
    
    bar_radius = bar_diameter / 2
    for pos in bar_positions:
        bar = Circle(pos, bar_radius, color='red', alpha=0.8)
        ax2.add_patch(bar)
    ax2.plot([], [], 'ro', markersize=10, label=f'Rebar D{bar_diameter} (8 bars)')
    
    # Add dimensions
    ax2.plot([-b/2, b/2], [-h/2-60, -h/2-60], 'k-', linewidth=1)
    ax2.plot([-b/2, -b/2], [-h/2-70, -h/2-50], 'k-', linewidth=1)
    ax2.plot([b/2, b/2], [-h/2-70, -h/2-50], 'k-', linewidth=1)
    ax2.text(0, -h/2-90, f'{b} mm', ha='center', fontsize=10, fontweight='bold')
    
    ax2.plot([-b/2-60, -b/2-60], [-h/2, h/2], 'k-', linewidth=1)
    ax2.plot([-b/2-70, -b/2-50], [-h/2, -h/2], 'k-', linewidth=1)
    ax2.plot([-b/2-70, -b/2-50], [h/2, h/2], 'k-', linewidth=1)
    ax2.text(-b/2-100, 0, f'{h} mm', rotation=90, va='center', fontsize=10, fontweight='bold')
    
    # Add cover dimension
    ax2.plot([core_b/2, b/2], [h/2+20, h/2+20], 'g-', linewidth=1)
    ax2.text((core_b/2+b/2)/2, h/2+35, f'cover={cover} mm', ha='center', 
            fontsize=9, color='green', fontweight='bold')
    
    ax2.set_xlabel('Width (mm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Height (mm)', fontsize=11, fontweight='bold')
    ax2.set_title('Column Cross-Section (400x400 mm)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlim(-300, 300)
    ax2.set_ylim(-300, 300)
    
    # --- Plot 3: Material Information ---
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    rho = 8*A_bar/(b*h)*100
    
    info_text = f"""
    MODEL INFORMATION
    {'='*40}
    
    GEOMETRY:
    - Column Height: {H_col} mm
    - Foundation: {L_fnd} mm
    - Section: {b}x{h} mm
    - Cover: {cover} mm
    
    PLASTIC HINGE:
    - Length (Lp): {Lp:.1f} mm
    - Lp/H: {Lp/H_col*100:.1f}%
    
    MATERIALS:
    - Concrete (Core): 40 MPa
    - Concrete (Cover): 30 MPa
    - Steel: Fy = 300 MPa
    - Reinforcement: 8-D{bar_diameter}
    - Rho: {rho:.2f}%
    
    LOADING:
    - Axial: {P_axial/1000} kN
    - Lateral: Cyclic
    
    ELEMENTS:
    - Foundation (elastic)
    - Plastic hinge (fiber, 5 pts)
    - Elastic column
    """
    
    ax3.text(0.1, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'model_geometry.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Model geometry plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)

# Plot the model before analysis
plot_model_geometry()

# --- Gravity Analysis ---
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(3, 0.0, -P_axial, 0.0)

ops.constraints('Plain')
ops.numberer('RCM')
ops.system('BandGeneral')
ops.test('NormDispIncr', 1.0e-6, 25)
ops.algorithm('Newton')
ops.integrator('LoadControl', 0.1)
ops.analysis('Static')
ops.analyze(10)
ops.loadConst('-time', 0.0)

print(f"\nGravity applied: Axial load = {P_axial/1000} kN")

# --- Cyclic Loading Protocol ---
drift_peaks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]  # % drift
cycles_per_peak = 3

# Generate cyclic displacement history
disp_history = []
for drift_pct in drift_peaks:
    target_disp = (drift_pct / 100.0) * H_col
    for cycle in range(cycles_per_peak):
        disp_history.append(target_disp)
        disp_history.append(0.0)
        disp_history.append(-target_disp)
        disp_history.append(0.0)

print(f"\nTotal displacement steps in cyclic protocol: {len(disp_history)}")
print(f"Drift levels: {drift_peaks}%")
print(f"Cycles per level: {cycles_per_peak}")

# --- Cyclic Analysis ---
ops.pattern('Plain', 2, 1)
ops.load(3, 1.0, 0.0, 0.0)

ops.test('NormDispIncr', 1.0e-6, 100)
ops.algorithm('Newton')
ops.analysis('Static')

# Data storage
displacements = []
base_shear = []
deformed_shapes = []
step_count = 0

print("\nRunning Cyclic Analysis...")

# Start from zero
current_disp = 0.0

# Store snapshots at specific drift levels for deformation plots
snapshot_drifts = [0.5, 1.0, 2.0, 3.0]
snapshot_indices = []

for i, target_disp in enumerate(disp_history):
    disp_incr = target_disp - current_disp
    num_substeps = max(10, int(abs(disp_incr) / 0.5))
    step_size = disp_incr / num_substeps
    
    ops.integrator('DisplacementControl', 3, 1, step_size)
    
    for j in range(num_substeps):
        ok = ops.analyze(1)
        
        if ok != 0:
            ops.algorithm('ModifiedNewton', '-initial')
            ok = ops.analyze(1)
            ops.algorithm('Newton')
        
        if ok != 0:
            ops.algorithm('NewtonLineSearch')
            ok = ops.analyze(1)
            ops.algorithm('Newton')
        
        if ok != 0:
            ops.test('NormDispIncr', 1.0e-5, 200)
            ops.algorithm('ModifiedNewton', '-initial')
            ok = ops.analyze(1)
            ops.test('NormDispIncr', 1.0e-6, 100)
            ops.algorithm('Newton')
        
        if ok != 0:
            print(f"  WARNING: Convergence failed at step {i}, substep {j}")
            break
        
        # Record data
        disp = ops.nodeDisp(3, 1)
        displacements.append(disp)
        
        ops.reactions()
        base_shear.append(-ops.nodeReaction(1, 1) / 1000.0)
        
        # Store deformed shape at key drifts
        current_drift = abs(disp / H_col * 100)
        for snapshot_drift in snapshot_drifts:
            if abs(current_drift - snapshot_drift) < 0.05:
                # Check if this drift level already captured
                already_captured = False
                for captured_drift, _, _ in deformed_shapes:
                    if captured_drift == snapshot_drift:
                        already_captured = True
                        break
                
                if not already_captured:
                    node_disp_data = {
                        1: [ops.nodeDisp(1, 1), ops.nodeDisp(1, 2)],
                        2: [ops.nodeDisp(2, 1), ops.nodeDisp(2, 2)],
                        3: [ops.nodeDisp(3, 1), ops.nodeDisp(3, 2)],
                        4: [ops.nodeDisp(4, 1), ops.nodeDisp(4, 2)]
                    }
                    deformed_shapes.append((snapshot_drift, node_disp_data, disp))
                    snapshot_indices.append(len(displacements)-1)
        
        step_count += 1
    
    if ok != 0:
        break
    
    current_disp = target_disp
    
    if (i+1) % 12 == 0:
        drift_level_idx = min(i // 12, len(drift_peaks)-1)
        print(f"  Completed drift level {drift_peaks[drift_level_idx]}%")

print(f"\nCyclic Analysis Finished!")
print(f"Total analysis steps: {step_count}")

# ============================================================================
# FUNCTION TO PLOT DEFORMED SHAPES
# ============================================================================
def plot_deformed_shapes():
    """Plot deformed shapes at different drift levels"""
    n_shapes = len(deformed_shapes)
    if n_shapes == 0:
        print("WARNING: No deformed shapes captured!")
        return
        
    fig, axes = plt.subplots(1, n_shapes, figsize=(5*n_shapes, 6))
    
    if n_shapes == 1:
        axes = [axes]
    
    scale_factor = 5  # Exaggeration for visibility
    
    for idx, (drift, node_disps, actual_disp) in enumerate(deformed_shapes):
        ax = axes[idx]
        
        # Plot original shape
        # Foundation
        node1 = node_coords[1]
        node2 = node_coords[2]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 
               'b--', linewidth=2, alpha=0.5, label='Original')
        
        # Plastic hinge region
        node4 = node_coords[4]
        ax.plot([node2[0], node4[0]], [node2[1], node4[1]], 
               'r--', linewidth=2, alpha=0.5)
        
        # Elastic column
        node3 = node_coords[3]
        ax.plot([node4[0], node3[0]], [node4[1], node3[1]], 
               'orange', linestyle='--', linewidth=2, alpha=0.5)
        
        # Plot deformed shape
        # Foundation
        node1_def = [node1[0] + node_disps[1][0]*scale_factor, 
                    node1[1] + node_disps[1][1]*scale_factor]
        node2_def = [node2[0] + node_disps[2][0]*scale_factor, 
                    node2[1] + node_disps[2][1]*scale_factor]
        ax.plot([node1_def[0], node2_def[0]], [node1_def[1], node2_def[1]], 
               'b-', linewidth=3, alpha=0.8, label='Foundation')
        
        # Plastic hinge region
        node4_def = [node4[0] + node_disps[4][0]*scale_factor, 
                    node4[1] + node_disps[4][1]*scale_factor]
        ax.plot([node2_def[0], node4_def[0]], [node2_def[1], node4_def[1]], 
               'r-', linewidth=3, alpha=0.8, label='Plastic Hinge')
        
        # Elastic column
        node3_def = [node3[0] + node_disps[3][0]*scale_factor, 
                    node3[1] + node_disps[3][1]*scale_factor]
        ax.plot([node4_def[0], node3_def[0]], [node4_def[1], node3_def[1]], 
               'orange', linewidth=3, alpha=0.8, label='Elastic Column')
        
        # Plot nodes
        for node_tag, disp in node_disps.items():
            orig = node_coords[node_tag]
            def_pos = [orig[0] + disp[0]*scale_factor, orig[1] + disp[1]*scale_factor]
            ax.plot(def_pos[0], def_pos[1], 'ko', markersize=10)
            ax.plot(orig[0], orig[1], 'bo', markersize=8, alpha=0.5)
        
        # Highlight plastic hinge zone
        ax.axhspan(L_fnd, L_fnd + Lp, alpha=0.15, color='red')
        
        # Add drift arrow
        top_node_orig = node_coords[3]
        top_node_disp = node_disps[3]
        top_node_def = [top_node_orig[0] + top_node_disp[0]*scale_factor, 
                       top_node_orig[1] + top_node_disp[1]*scale_factor]
        
        ax.annotate('', xy=(top_node_def[0], top_node_def[1]), 
                   xytext=(top_node_orig[0], top_node_orig[1]),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=3))
        
        ax.text(top_node_orig[0]+200, top_node_orig[1]+200, 
               f'Delta = {actual_disp:.1f} mm\n({drift:.1f}% drift)', 
               fontsize=10, color='purple', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
        ax.set_title(f'Deformed @ {drift}% Drift (Scale: {scale_factor}x)', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
        
        # Draw boundary condition
        support_size = 100
        ax.plot([-support_size, support_size], [0, 0], 'k-', linewidth=4)
        for i in range(-3, 4):
            ax.plot([i*support_size/3, (i-0.5)*support_size/3], [0, -support_size/2], 'k-', linewidth=2)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'deformed_shapes.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Deformed shapes plot saved to: {output_path}")
    
    # Close the figure
    plt.close(fig)

# Plot deformed shapes
if deformed_shapes:
    plot_deformed_shapes()

# ============================================================================
# HYSTERESIS AND HISTORY PLOTS
# ============================================================================
print("\nGenerating hysteresis plots...")
fig = plt.figure(figsize=(16, 10))

# 1. Hysteresis Loop
ax1 = plt.subplot(2, 2, (1, 2))
ax1.plot(displacements, base_shear, 'b-', linewidth=1.5, alpha=0.8)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Mark snapshot points
if snapshot_indices:
    snapshot_disps = [displacements[i] for i in snapshot_indices]
    snapshot_forces = [base_shear[i] for i in snapshot_indices]
    ax1.plot(snapshot_disps, snapshot_forces, 'ro', markersize=10, 
            label='Snapshot Points', zorder=5)

ax1.set_xlabel('Lateral Displacement (mm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Base Shear (kN)', fontsize=12, fontweight='bold')
ax1.set_title('Hysteresis Loops - Cyclic Loading', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add drift markers
drift_markers = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
for drift in drift_markers:
    pos_disp = (drift / 100.0) * H_col
    neg_disp = -pos_disp
    ax1.axvline(x=pos_disp, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.axvline(x=neg_disp, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.text(pos_disp, ax1.get_ylim()[1]*0.95, f'{drift}%', 
             ha='center', fontsize=8, color='gray')

# 2. Displacement History
ax2 = plt.subplot(2, 2, 3)
steps = range(len(displacements))
ax2.plot(steps, displacements, 'r-', linewidth=1)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Mark snapshots
if snapshot_indices:
    ax2.plot(snapshot_indices, [displacements[i] for i in snapshot_indices], 
            'go', markersize=8, label='Snapshots')

ax2.set_xlabel('Analysis Step', fontsize=11)
ax2.set_ylabel('Displacement (mm)', fontsize=11)
ax2.set_title('Displacement History', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Base Shear History
ax3 = plt.subplot(2, 2, 4)
ax3.plot(steps, base_shear, 'g-', linewidth=1)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

if snapshot_indices:
    ax3.plot(snapshot_indices, [base_shear[i] for i in snapshot_indices], 
            'ro', markersize=8, label='Snapshots')

ax3.set_xlabel('Analysis Step', fontsize=11)
ax3.set_ylabel('Base Shear (kN)', fontsize=11)
ax3.set_title('Base Shear History', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, 'hysteresis_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Hysteresis analysis plot saved to: {output_path}")
plt.close(fig)

# ============================================================================
# ENVELOPE CURVE
# ============================================================================
print("Generating envelope curve...")
fig2, ax = plt.subplots(figsize=(10, 6))

# Extract envelopes
pos_disps = []
pos_shears = []
neg_disps = []
neg_shears = []

for d, v in zip(displacements, base_shear):
    if d >= 0:
        if not pos_disps or d > pos_disps[-1]:
            pos_disps.append(d)
            pos_shears.append(v)
    else:
        if not neg_disps or d < neg_disps[-1]:
            neg_disps.append(d)
            neg_shears.append(v)

ax.plot(displacements, base_shear, 'b-', linewidth=0.5, alpha=0.3, label='Hysteresis')
ax.plot(pos_disps, pos_shears, 'r-', linewidth=3, label='Positive Envelope', 
        marker='o', markersize=4, markevery=50)
ax.plot(neg_disps, neg_shears, 'g-', linewidth=3, label='Negative Envelope', 
        marker='s', markersize=4, markevery=50)

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Lateral Displacement (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Base Shear (kN)', fontsize=12, fontweight='bold')
ax.set_title('Backbone Curve (Envelope)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, 'envelope_curve.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Envelope curve plot saved to: {output_path}")
plt.close(fig2)

# ============================================================================
# ENERGY DISSIPATION
# ============================================================================
print("Generating energy dissipation plot...")
energy_dissipated = []
cumulative_energy = 0.0

for i in range(1, len(displacements)):
    dE = 0.5 * (base_shear[i] + base_shear[i-1]) * (displacements[i] - displacements[i-1])
    cumulative_energy += abs(dE)
    energy_dissipated.append(cumulative_energy)

fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(displacements)), [e/1e6 for e in energy_dissipated], 
        'purple', linewidth=2)
ax.set_xlabel('Analysis Step', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Energy Dissipated (kN*m)', fontsize=12, fontweight='bold')
ax.set_title('Energy Dissipation History', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, 'energy_dissipation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Energy dissipation plot saved to: {output_path}")
plt.close(fig3)

# ============================================================================
# DRIFT VS BASE SHEAR PLOT
# ============================================================================
print("Generating drift vs base shear plot...")
drift_ratios = [d / H_col * 100 for d in displacements]

fig4, ax = plt.subplots(figsize=(10, 6))
ax.plot(drift_ratios, base_shear, 'b-', linewidth=1.5, alpha=0.8)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Drift Ratio (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Base Shear (kN)', fontsize=12, fontweight='bold')
ax.set_title('Base Shear vs Drift Ratio', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, 'drift_vs_base_shear.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] Drift vs base shear plot saved to: {output_path}")
plt.close(fig4)

# ============================================================================
# SUMMARY OUTPUT AND TEXT FILE
# ============================================================================
print(f"\n{'='*60}")
print(f"ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Plastic hinge length: {Lp:.1f} mm ({Lp/H_col*100:.1f}% of column)")
print(f"Total energy dissipated: {cumulative_energy/1e6:.2f} kN*m")
print(f"Peak positive displacement: {max(displacements):.2f} mm ({max(displacements)/H_col*100:.2f}%)")
print(f"Peak negative displacement: {min(displacements):.2f} mm ({min(displacements)/H_col*100:.2f}%)")
print(f"Peak positive base shear: {max(base_shear):.1f} kN")
print(f"Peak negative base shear: {min(base_shear):.1f} kN")
print(f"{'='*60}")

# Save summary to text file
summary_path = os.path.join(output_dir, 'analysis_summary.txt')
with open(summary_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("OPENSEES CYCLIC ANALYSIS SUMMARY\n")
    f.write("WITH PLASTIC HINGE MODELING\n")
    f.write("="*60 + "\n\n")
    
    f.write("MODEL PARAMETERS:\n")
    f.write(f"  Column Height: {H_col} mm\n")
    f.write(f"  Foundation Length: {L_fnd} mm\n")
    f.write(f"  Axial Load: {P_axial/1000} kN\n")
    f.write(f"  Column Section: {b}x{h} mm\n")
    f.write(f"  Cover: {cover} mm\n")
    f.write(f"  Reinforcement: 8-D{bar_diameter}\n")
    f.write(f"  Rho: {8*A_bar/(b*h)*100:.2f}%\n\n")
    
    f.write("PLASTIC HINGE:\n")
    f.write(f"  Length (Lp): {Lp:.1f} mm\n")
    f.write(f"  Lp/H ratio: {Lp/H_col*100:.1f}%\n\n")
    
    f.write("CYCLIC LOADING PROTOCOL:\n")
    f.write(f"  Drift levels: {drift_peaks}%\n")
    f.write(f"  Cycles per level: {cycles_per_peak}\n")
    f.write(f"  Total analysis steps: {step_count}\n\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Total energy dissipated: {cumulative_energy/1e6:.2f} kN*m\n")
    f.write(f"  Peak positive displacement: {max(displacements):.2f} mm ({max(displacements)/H_col*100:.2f}%)\n")
    f.write(f"  Peak negative displacement: {min(displacements):.2f} mm ({min(displacements)/H_col*100:.2f}%)\n")
    f.write(f"  Peak positive base shear: {max(base_shear):.1f} kN\n")
    f.write(f"  Peak negative base shear: {min(base_shear):.1f} kN\n")

print(f"\n[OK] Analysis summary saved to: {summary_path}")
print(f"\n{'='*60}")
print(f"ALL OUTPUTS SAVED TO: {os.path.abspath(output_dir)}")
print(f"{'='*60}")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_feature_based_nn():
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define the positions and labels of the boxes
    feature_names = ['Spectral Centroid', 'Spectral Bandwidth', 'Spectral Rolloff', 'Zero Crossing Rate', 'RMS Energy']
    input_positions = [(0.1, y) for y in range(9, 4, -1)]
    dense_positions = [(1.5, y) for y in range(9, 4, -1)]

    boxes = []
    arrows = []

    # Input layers
    for i, (feature, pos) in enumerate(zip(feature_names, input_positions)):
        boxes.append({'label': f'Input\n{feature}', 'xy': pos, 'width': 0.5, 'height': 0.5, 'color': 'lightblue'})
        arrows.append((pos[0] + 0.5, pos[1] + 0.25, 1.0, pos[1] + 0.25))
    
    # Reshape and Dense layers
    for i, pos in enumerate(dense_positions):
        boxes.append({'label': 'Reshape\n(400,)', 'xy': (1.5, pos[1]), 'width': 0.5, 'height': 0.5, 'color': 'lightgrey'})
        boxes.append({'label': 'Dense\n64 units', 'xy': (2.5, pos[1]), 'width': 0.5, 'height': 0.5, 'color': 'lightgreen'})
        arrows.append((2.0, pos[1] + 0.25, 2.5, pos[1] + 0.25))
    
    # Concatenate layer
    boxes.append({'label': 'Concatenate', 'xy': (3.5, 6.5), 'width': 0.5, 'height': 0.5, 'color': 'lightgrey'})
    for pos in dense_positions:
        arrows.append((3.0, pos[1] + 0.25, 3.5, 6.75))
    
    # Output layer
    boxes.append({'label': 'Output\nLayer', 'xy': (4.5, 6.5), 'width': 0.5, 'height': 0.5, 'color': 'lightblue'})
    arrows.append((4.0, 6.75, 4.5, 6.75))

    # Plot the boxes
    for box in boxes:
        ax.add_patch(Rectangle(box['xy'], box['width'], box['height'], color=box['color'], ec='black'))
        rx, ry = box['xy']
        cx = rx + box['width'] / 2.0
        cy = ry + box['height'] / 2.0
        ax.annotate(box['label'], (cx, cy), color='black', weight='bold', fontsize=10, ha='center', va='center')

    # Draw arrows
    arrowprops = dict(facecolor='black', arrowstyle='->')
    for arrow in arrows:
        ax.annotate('', xy=(arrow[2], arrow[3]), xytext=(arrow[0], arrow[1]), arrowprops=arrowprops)

    # Set limits and hide axes
    ax.set_xlim(0, 6)
    ax.set_ylim(4, 10)
    ax.axis('off')

    plt.show()

plot_feature_based_nn()

from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import cv2 
class ImageLabeler:
    def __init__(self, title, image):
        self.title = title
        self.positive_keypoints = []
        self.negative_keypoints = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.imshow(image)

        def onclick(event):
            # Ignore clicks outside the axes
            if event.inaxes != self.ax:
                return
            if event.button is MouseButton.LEFT:
                # Left click ignored (per your message)
                print("Ignored, please use the right click.")
            elif event.button is MouseButton.RIGHT:
                ix, iy = event.xdata, event.ydata
                print(f'negative x = {ix:.2f}, y = {iy:.2f}')
                self.ax.scatter(ix, iy, c='b', marker='x')
                self.negative_keypoints.append([float(ix), float(iy)])
                # Only redraw this figure:
                self.fig.canvas.draw_idle()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', onclick)

    def show(self, block=True):
        plt.show(block=block)

    def save(self, filename, dpi=150):
        # Ensure the latest artists are rendered
        self.fig.tight_layout()
        self.fig.savefig(filename, dpi=dpi)
        print(f"Image saved as {filename}")

    def get_keypoints(self):
        return self.negative_keypoints
        # return self.positive_keypoints, self.negative_keypoints
    
def main():
    # Example usage
    image = plt.imread('example_image.jpg')  # Replace with your image path
    labeler = ImageLabeler("Image Labeler", image)
    labeler.show()
    positive, negative = labeler.get_keypoints()
    print("Positive Keypoints:", positive)
    print("Negative Keypoints:", negative)
    
if __name__ == "__main__":
    main()
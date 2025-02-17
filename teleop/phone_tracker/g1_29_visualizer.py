import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import numpy as np

class G1_29_Visualizer:

	def __init__(self, reduced_robot):
		# Initialize the Meshcat visualizer for visualization
		self.reduced_robot = reduced_robot
		self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
		self.vis.initViewer(open=True) 
		self.vis.loadViewerModel("pinocchio") 
		self.vis.displayFrames(True, frame_ids=[101, 102], axis_length = 0.15, axis_width = 5)
		self.vis.display(pin.neutral(self.reduced_robot.model))
		self.vis.viewer["/Background"].set_property("top_color", [0.9, 0.2, 0.3, 1.0])

		# Enable the display of end effector target frames with short axis lengths and greater width.
		frame_viz_names = ['L_ee_target', 'R_ee_target']
		FRAME_AXIS_POSITIONS = (
			np.array([[0, 0, 0], [1, 0, 0],
						[0, 0, 0], [0, 1, 0],
						[0, 0, 0], [0, 0, 1]]).astype(np.float32).T
		)
		FRAME_AXIS_COLORS = (
			np.array([[1, 0, 0], [1, 0.6, 0],
						[0, 1, 0], [0.6, 1, 0],
						[0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
		)
		axis_length = 0.1
		axis_width = 10
		for frame_viz_name in frame_viz_names:
			self.vis.viewer[frame_viz_name].set_object(
				mg.LineSegments(
					mg.PointsGeometry(
						position=axis_length * FRAME_AXIS_POSITIONS,
						color=FRAME_AXIS_COLORS,
					),
					mg.LineBasicMaterial(
						linewidth=axis_width,
						vertexColors=True,
					),
				)
			)

	def update(self, q, left_wrist, right_wrist):
		self.vis.viewer['L_ee_target'].set_transform(left_wrist) 
		self.vis.viewer['R_ee_target'].set_transform(right_wrist)
		self.vis.display(q)
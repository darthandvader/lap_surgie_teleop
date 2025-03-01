import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib
import meshcat.transformations as tf
matplotlib.use('Agg')

touch_sensing_geometry_names = ['right_index_1_0', 'left_index_1_0',
                                'right_index_2_0', 'left_index_2_0',
                                'right_index_tip_0', 'left_index_tip_0',
                                'right_little_1_0', 'left_little_1_0',
                                'right_little_2_0', 'left_little_2_0',
                                'right_little_tip_0', 'left_little_tip_0',
                                'right_middle_1_0', 'left_middle_1_0',
                                'right_middle_2_0', 'left_middle_2_0',
                                'right_middle_tip_0', 'left_middle_tip_0',                                        
                                'right_ring_1_0', 'left_ring_1_0',
                                'right_ring_2_0', 'left_ring_2_0',
                                'right_ring_tip_0', 'left_ring_tip_0',
                                'right_thumb_1_0', 'left_thumb_1_0',
                                'right_thumb_2_0', 'left_thumb_2_0',
                                'right_thumb_3_0', 'left_thumb_3_0',
                                'right_thumb_4_0', 'left_thumb_4_0',
                                'right_thumb_tip_0', 'left_thumb_tip_0',
                                'right_base_link_0', 'left_base_link_0',
                                ]
    
joint_sensing_names = ['right_index_1_joint', 'left_index_1_joint',
                        'right_little_1_joint', 'left_little_1_joint',
                        'right_middle_1_joint', 'left_middle_1_joint',
                        'right_ring_1_joint', 'left_ring_1_joint',
                        'right_thumb_1_joint', 'left_thumb_1_joint',
                        'right_thumb_2_joint', 'left_thumb_2_joint',
                        ]

def compute_billboard_transform(camera_position, target, up_vector):
    '''
    
    '''
    # Compute the forward vector (from camera to target)
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    
    # Compute the right vector
    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)
    
    # Recompute the up vector to ensure orthogonality
    up = np.cross(forward, right)
    
    # Construct the rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = up
    rotation_matrix[:3, 2] = -forward
    
    # Construct the translation matrix (position the plane at the target)
    translation_matrix = tf.translation_matrix(target)
    
    # Combine rotation and translation
    billboard_transform = np.dot(translation_matrix, rotation_matrix)
    
    return billboard_transform


class TouchSensorArray:

    def __init__(self, name, shape, Y, Z, translation, flip0=True, flip1=True):
        self.name = name
        self.shape = shape
        self.Y = Y
        self.Z = Z
        self.translation = translation
        self.data = np.zeros(shape)
        self.range = [0, 500] # 4095

        ys = np.linspace(0, self.Y, self.shape[0])
        zs = np.linspace(0, self.Z, self.shape[1])
        ys, zs = np.meshgrid(zs, ys)
        ys, zs = ys.flatten(), zs.flatten()
        self.points = np.vstack((np.zeros_like(ys), zs, ys)).T + self.translation
        self.flip0, self.flip1 = flip0, flip1
        

    def set_data(self, data):
        # assert(data.shape == self.shape, f"Data shape mismatch: {import plotly.graph_objects as godata.shape} != {self.shape}, for {self.name}")
        self.data = np.clip(data, self.range[0], self.range[1])
        if self.flip0:
            self.data = np.flip(self.data, axis=0)
        if self.flip1:
            self.data = np.flip(self.data, axis=1)
        
    
    def get_data_as_colored_points(self):

        colors = np.zeros((self.data.size, 4))
        colors[:, -1] = 0.03 + (self.data.flatten() - self.range[0]) / (self.range[1] - self.range[0])
        colors[:, -1] = np.clip(colors[:, -1], 0, 1)
        colors[:, 0] = (self.data.flatten() - self.range[0]) / (self.range[1] - self.range[0])

        return self.points, colors

right_hand_sensors = {
            'palm' : TouchSensorArray('palm', (14, 8), 0.08, 0.03, np.array([0., 0.005, 0.02]), flip0=False,),
            'thumb_palm' : TouchSensorArray('thumb_palm', (8, 12), 0.02, 0.03, np.array([0, 0.135, -0.02]), flip1=False),
            'thumb_middle': TouchSensorArray('thumb_middle', (3, 3), 0.02, 0.02, np.array([0, 0.135, 0.03]), flip0=False,),
            'thumb_top' : TouchSensorArray('thumb_top', (8, 12), 0.02, 0.03, np.array([0, 0.135, 0.06]), flip0=False,),
            'thumb_tip': TouchSensorArray('thumb_tip', (3, 3), 0.02, 0.005, np.array([0, 0.135, 0.095]), flip0=False,),
            'index_palm' : TouchSensorArray('thumb_palm', (8, 10), 0.02, 0.03, np.array([0, 0.085, 0.066]), flip0=False,),
            'index_top' : TouchSensorArray('thumb_top', (8, 12), 0.02, 0.03, np.array([0, 0.085, 0.13]), flip0=False,),
            'index_tip': TouchSensorArray('thumb_tip', (3, 3), 0.02, 0.005, np.array([0, 0.085, 0.163]), flip0=False,),
            'middle_palm' : TouchSensorArray('thumb_palm', (8, 10), 0.02, 0.03, np.array([0, 0.05, 0.067]), flip0=False,),
            'middle_top' : TouchSensorArray('thumb_top', (8, 12), 0.02, 0.03, np.array([0, 0.05, 0.138]), flip0=False,),
            'middle_tip': TouchSensorArray('thumb_tip', (3, 3), 0.02, 0.005, np.array([0, 0.05, 0.17]), flip0=False,),
            'ring_palm' : TouchSensorArray('thumb_palm', (8, 10), 0.02, 0.03, np.array([0, 0.015, 0.065]), flip0=False,),
            'ring_top' : TouchSensorArray('thumb_top', (8, 12), 0.02, 0.03, np.array([0, 0.015, 0.13]), flip0=False,),
            'ring_tip': TouchSensorArray('thumb_tip', (3, 3), 0.02, 0.005, np.array([0, 0.015, 0.165]), flip0=False,),
            'little_palm' : TouchSensorArray('thumb_palm', (8, 10), 0.02, 0.03, np.array([0, -0.015, 0.062]), flip0=False,),
            'little_top' : TouchSensorArray('thumb_top', (8, 12), 0.02, 0.03, np.array([0, -0.015, 0.11]), flip0=False,),
            'little_tip': TouchSensorArray('thumb_tip', (3, 3), 0.02, 0.005, np.array([0, -0.015, 0.145]), flip0=False,),
            }
left_hand_sensors = {
            'palm' : TouchSensorArray('palm', (14, 8), -0.08, 0.03, np.array([0., -0.005, 0.02])),
            'thumb_palm' : TouchSensorArray('thumb_palm', (8, 12), -0.02, 0.03, np.array([0, -0.135, -0.02]), flip0=False, flip1=False),
            'thumb_middle': TouchSensorArray('thumb_middle', (3, 3), -0.02, 0.02, np.array([0, -0.135, 0.03])),
            'thumb_top' : TouchSensorArray('thumb_top', (8, 12), -0.02, 0.03, np.array([0, -0.135, 0.06])),
            'thumb_tip': TouchSensorArray('thumb_tip', (3, 3), -0.02, 0.005, np.array([0, -0.135, 0.095])),
            'index_palm' : TouchSensorArray('thumb_palm', (8, 10), -0.02, 0.03, np.array([0, -0.085, 0.066])),
            'index_top' : TouchSensorArray('thumb_top', (8, 12), -0.02, 0.03, np.array([0, -0.085, 0.13])),
            'index_tip': TouchSensorArray('thumb_tip', (3, 3), -0.02, 0.005, np.array([0, -0.085, 0.163])),
            'middle_palm' : TouchSensorArray('thumb_palm', (8, 10), -0.02, 0.03, np.array([0, -0.05, 0.067])),
            'middle_top' : TouchSensorArray('thumb_top', (8, 12), -0.02, 0.03, np.array([0, -0.05, 0.138])),
            'middle_tip': TouchSensorArray('thumb_tip', (3, 3), -0.02, 0.005, np.array([0, -0.05, 0.17])),
            'ring_palm' : TouchSensorArray('thumb_palm', (8, 10), -0.02, 0.03, np.array([0, -0.015, 0.065])),
            'ring_top' : TouchSensorArray('thumb_top', (8, 12), -0.02, 0.03, np.array([0, -0.015, 0.13])),
            'ring_tip': TouchSensorArray('thumb_tip', (3, 3), -0.02, 0.005, np.array([0, -0.015, 0.165])),
            'little_palm' : TouchSensorArray('thumb_palm', (8, 10), -0.02, 0.03, np.array([0, 0.015, 0.062])),
            'little_top' : TouchSensorArray('thumb_top', (8, 12), -0.02, 0.03, np.array([0, 0.015, 0.11])),
            'little_tip': TouchSensorArray('thumb_tip', (3, 3), -0.02, 0.005, np.array([0, 0.015, 0.145])),
            }



img_right = Image.open('../../assets/g1/hand.png')
img_left = img_right.transpose(Image.FLIP_LEFT_RIGHT)

def sensor_data_to_image(data, side='right'):
    '''
    Give hand data, return an touch sensing image
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    term_map = {'fingerone':'little',
            'fingertwo':'ring',
            'fingerthree':'middle',
            'fingerfour':'index',
            'fingerfive':'thumb',
            'palm':'palm'}  

    sensors = right_hand_sensors if side == 'right' else left_hand_sensors

    all_points = []
    all_colors = []
    for key in data['touch'].keys():
        if key == 'palm_touch':
            sensor_key = 'palm'
        else:
            sensor_key = term_map[key.split('_')[0]] + '_' + key.split('_')[1]

        sensor_data = data['touch'][key]
        if sensor_key != 'palm':
            sensor_data = sensor_data.T
        sensors[sensor_key].set_data(sensor_data)
        points, colors = sensors[sensor_key].get_data_as_colored_points()
        all_points.append(points)
        all_colors.append(colors)

    # ax.clear()
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    if side == 'right':
        ax.imshow(img_right, extent=[-0.05, 0.18, -0.05, 0.18], alpha=0.1)
    else:
        ax.imshow(img_left, extent=[-0.18, 0.05, -0.05, 0.18], alpha=0.1)

    ax.scatter(all_points[:, 1], all_points[:, 2], c=all_colors[:, :-1], alpha=all_colors[:, -1], s=50 + all_colors[:, -1]*150)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.invert_xaxis()
    

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=80, bbox_inches='tight')
    buf.seek(0)  # Move the cursor to the start of the buffer
    plt.close(fig)
    return buf
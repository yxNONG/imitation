import pathlib

# Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ['knee', 'lumbar_yaw', 'lumbar_pitch', 'lumbar_roll', 'neck_yaw', 'neck_pitch', 'neck_roll', 
               'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_yaw', 'right_elbow_pitch', 
               'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll', 'right_index', 'right_index_follow1', 'right_index_follow2', 
               'right_thumb_roll', 'right_thumb_pitch', 'right_thumb', 'right_thumb_follow1', 'right_middle', 'right_middle_follow1', 'right_middle_follow2', 
               'right_ring', 'right_ring_follow1', 'right_ring_follow2', 'right_pinky', 'right_pinky_follow1', 'right_pinky_follow2', 

               'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_yaw', 'left_elbow_pitch',
               'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll', 'left_index', 'left_index_follow1', 'left_index_follow2', 
               'left_middle', 'left_middle_follow1', 'left_middle_follow2', 'left_ring', 'left_ring_follow1', 'left_ring_follow2', 
               'left_pinky', 'left_pinky_follow1', 'left_pinky_follow2', 'left_thumb_roll', 'left_thumb_pitch', 'left_thumb', 'left_thumb_follow1']

GINGER_JOINT_POSE = [0,0,0,0,0,0,0,
                  0,0,0,0,
                  0,0,0,0,0,0,
                  0,0,0,0,0,0,0,
                  0,0,0,0,0,0,

                  0,0,0, 0,
                  0,0,0,0,0,0,
                  0,0,0,0,0,0,
                  0,0,0,0,0,0,0]

ACTUATOR_NAMES = ['knee', 'lumbar_yaw', 'lumbar_pitch', 'lumbar_roll', 'neck_yaw', 'neck_pitch', 'neck_roll', 
                  'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow_yaw', 'left_elbow_pitch', 'left_wrist_yaw', 'left_wrist_pitch', 'left_wrist_roll', 
                  'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow_yaw', 'right_elbow_pitch', 'right_wrist_yaw', 'right_wrist_pitch', 'right_wrist_roll', 
                  'left_thumb_roll', 'left_thumb_pitch', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky', 
                  'right_thumb_roll', 'right_thumb_pitch', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']


GINGER_ACTUATOR_POSE = [0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0]

GINGER_ACTUATOR_RANGE = [[-1.0000e-05, 1.0000e-05],
                         [-1.0000e-05, 1.0000e-05],
                         [-1.0000e-05, 1.0000e-05],
                         [-1.0000e-05, 1.0000e-05],
                         [-1.5708e+00, 1.5708e+00],
                         [-7.8540e-01, 7.8540e-01],
                         [-5.2360e-01, 5.2360e-01],
                         [-2.7925e+00, 2.2689e+00],
                         [-2.6180e-01, 2.0944e+00],
                         [-1.5708e+00, 1.5708e+00],
                         [-1.9199e+00, 5.2360e-01],
                         [-1.5708e+00, 1.5708e+00],
                         [-4.3630e-01, 4.3630e-01],
                         [-8.7270e-01, 8.7270e-01],
                         [-2.2689e+00, 2.7925e+00],
                         [-2.0944e+00, 2.6180e-01],
                         [-1.5708e+00, 1.5708e+00],
                         [-1.9199e+00, 5.2360e-01],
                         [-1.5708e+00, 1.5708e+00],
                         [-4.3630e-01, 4.3630e-01],
                         [-8.7270e-01, 8.7270e-01],
                         [-5.2360e-01, 5.2360e-01],
                         [-5.2360e-01, 5.2360e-01],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-5.2360e-01, 5.2360e-01],
                         [-5.2360e-01, 5.2360e-01],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00],
                         [-1.0000e-01, 1.1300e+00]]


GINGER_R_HAND_CTRL_OPEN = [0.25,0.35,-0.1,-0.1,-0.1,-0.1,-0.1]
GINGER_R_HAND_CTRL_CLOS = [    0,0.25, 0.5, 0.8, 0.8, 0.8, 0.8]

GINGER_L_HAND_CTRL_OPEN = [-0.25,-0.25,-0.1,-0.1,-0.1,-0.1,-0.1]
GINGER_L_HAND_CTRL_CLOS = [    0,-0.25, 0.5, 0.8, 0.8, 0.8, 0.8]

GINGER_R_HAND_CTRL_SPONGE_IDLE = [0.,0.,-0.1,-0.1,-0.1,-0.1,-0.1]
GINGER_R_HAND_CTRL_SPONGE_OPEN = [0.65,0.55,-0.1,-0.1,-0.1,-0.1,-0.1]
GINGER_R_HAND_CTRL_SPONGE_CLOS = [0.45,0.55, 0.5, 0.8, 0.8, 0.8, 0.8]

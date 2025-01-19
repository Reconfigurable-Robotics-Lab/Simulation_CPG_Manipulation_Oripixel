
#
# Copyright (c) 2025 Yuhao Jiang, RRL, EPFL
# license: Apache-2.0 license
#

import mujoco
import numpy as np
from jinja2 import Template
from scipy.spatial.transform import Rotation as R
import glfw
from shapely.geometry import Polygon

# Tile map
    # [ 4, 9, 14, 19, 24
    #   3, 8, 13, 18, 23
    #   2, 7, 12, 17, 22
    #   1, 6, 11, 16, 21
    #   0, 5, 10, 15, 20 ]

# Actuator map for each module (top view)
#       1   ---   2
#         \     /
#            3

xml_template = """
<mujoco model="oripixel">
    <statistic center="0 0 0" extent="1"/>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <visual>
    <headlight diffuse="1 1 1" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="30" elevation="-35" offwidth="1920" offheight="1080"/>
    </visual>

    <!-- set some defaults for units and lighting -->
    <compiler angle="degree" coordinate="local" />
	<option timestep="{{ sim_step }}" gravity="0 0 -9.81" integrator="Euler" solver="Newton" tolerance="1e-6" ls_iterations="30">
		<flag contact="enable"/>
	</option>
    <size memory="200M"/>
    <default>
        <geom contype="0" conaffinity="0"/>
    </default>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>  
        <body name="floor" pos="0 0 0">
            <geom name="floor" pos="0 0 0" size="0 0 1" type="plane" 
                material="groundplane" rgba="0.5 0.5 0.5 0.5" contype= "1" conaffinity="1"/>
                <site name="reference" pos="0 0 0" euler="0 0 0"/>
            <!-- BODY PLACEHOLDER for the GENERATED_XML -->

        </body>

    <!-- TOP_PLATE PLACEHOLDER for the GENERATED_XML -->
    <!-- OBJECT PLACEHOLDER for the GENERATED_XML -->
    </worldbody>

    <equality>
        <!-- EQUALITY PLACEHOLDER for the GENERATED_XML -->
        <!-- CONNECT PLATE PLACEHOLDER for the GENERATED_XML -->
	</equality>

    <actuator>
        <!-- ACTUATOR PLACEHOLDER for the GENERATED_XML -->
    </actuator>

    <contact>
        <!-- FLOOR CONTACT PLACEHOLDER for the GENERATED_XML -->
        <!-- PLATE CONTACT PLACEHOLDER for the GENERATED_XML -->
        <!-- OBJECT CONTACT PLACEHOLDER for the GENERATED_XML -->
    </contact>

    <sensor>
        <!-- SENSOR PLACEHOLDER for the GENERATED_XML -->
    </sensor>
</mujoco>

"""

xml_template_body = """
            <body name="{{ module_number }}_force_seat" pos = "{{ pos_force_seat }}" euler="{{ euler_force_seat }}">
                <geom name="{{ module_number }}_force_seat" type="box" size="30e-3 30e-3 1.5e-3" pos="0 0 0" euler="0 0 0" mass="100e-3"/>

                <!-- 1st Leg-->
                <body name="{{ module_number }}_body_1_seat" pos="{{ pos_seat_1 }}" euler="0 0 0">
                    <geom name="{{ module_number }}_body_1_seat" type="box" size="5e-3 5e-3 20e-3" pos="0 0 0" rgba="0.1 0.8 0.8 1"/>
                    <site name="{{ module_number }}_ft_sensor_bottom_1" pos="0 0 -20e-3" euler="0 0 -150"/>

                    <body name="{{ module_number }}_name_body_arm_1_1" pos="0 0 {{ arm_1_pos_z }}" euler="0 0 -120">
                        <joint name="{{ module_number }}_joint_1_01" type="hinge" axis="1 0 0" pos="0 0 {{ joint_1_pos_z }}" limited="true" range="1 89" stiffness= "{{ stiff_active }}" damping="{{ damp_active }}" solreflimit="1e-8 3"/> 
                        <geom name="{{ module_number }}_body_arm_1_1" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 0" mass="5e-3" rgba="1 1 0 1"/>

                        <body name="{{ module_number }}_body_arm_1_2" pos="0 0 {{ arm_2_pos_z }}" euler="0 0 180">
                            <joint name="{{ module_number }}_joint_1_12_x" type="hinge" axis="1 0 0" pos="0 0 0" limited="true" range="1 179" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_joint_1_12_y" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-45 45" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_joint_1_12_z" type="hinge" axis="0 0 1" pos="0 0 0" /> 
                            <geom name="{{ module_number }}_body_arm_1_2" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 {{ arm_2_pos_z }}" mass="5e-3" rgba="0 0 1 1"/>

                            <body name="{{ module_number }}_body_arm_1_3" pos="0 0 {{ arm_3_pos_z }}" euler="0 0 0">
                                <joint name="{{ module_number }}_joint_1_23_x" type="hinge" axis="1 0 0" pos="0 0 {{ joint_3_pos_z }}" limited="true" range="1 89" solreflimit="1e-8 3"/>    
                                <geom name="{{ module_number }}_body_arm_1_3" type="cylinder" size="1.5e-3 {{ arm_3_length }}" pos="0 0 0" mass="5e-3" rgba="1 0 1 1"/>

                            </body>  
                        </body>
                    </body>    
                </body>

                <!-- 2nd Leg-->
                <body name="{{ module_number }}_body_2_seat" pos="{{ pos_seat_2 }}" euler="0 0 0">
                    <geom name="{{ module_number }}_body_2_seat" type="box" size="5e-3 5e-3 20e-3" pos="0 0 0" rgba="0.1 0.8 0.8 1"/>
                    <site name="{{ module_number }}_ft_sensor_bottom_2" pos="0 0 -20e-3" euler="0 0 -150"/>

                    <body name="{{ module_number }}_body_arm_2_1" pos="0 0 {{ arm_1_pos_z }}" euler="0 0 120">
                        <joint name="{{ module_number }}_joint_2_01" type="hinge" axis="1 0 0" pos="0 0 {{ joint_1_pos_z }}" limited="true" range="1 89" stiffness= "{{ stiff_active }}" damping="{{ damp_active }}" solreflimit="1e-8 3"/> 
                        <geom name="{{ module_number }}_body_arm_2_1" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 0" mass="5e-3" rgba="1 1 0 1"/>
                        
                        <body name="{{ module_number }}_body_arm_2_2" pos="0 0 {{ arm_2_pos_z }}" euler="0 0 180">
                            <joint name="{{ module_number }}_joint_2_12_x" type="hinge" axis="1 0 0" pos="0 0 0" limited="true" range="1 179" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_joint_2_12_y" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-45 45" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_joint_2_12_z" type="hinge" axis="0 0 1" pos="0 0 0" /> 
                            <geom name="{{ module_number }}_body_arm_2_2" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 {{ arm_2_pos_z }}" mass="5e-3" rgba="0 0 1 1"/>

                            <body name="{{ module_number }}_body_arm_2_3" pos="0 0 {{ arm_3_pos_z }}" euler="0 0 0">
                                <joint name="{{ module_number }}_joint_2_23_x" type="hinge" axis="1 0 0" pos="0 0 {{ joint_3_pos_z }}" limited="true" range="1 89" solreflimit="1e-8 3"/>   
                                <geom name="{{ module_number }}_body_arm_2_3" type="cylinder" size="1.5e-3 {{ arm_3_length }}" pos="0 0 0" mass="5e-3" rgba="1 0 1 1"/>
                                
                            </body>  
                        </body>
                    </body>  
                </body>

                <!-- 3rd Leg-->
                <body name="{{ module_number }}_body_3_seat" pos="{{ pos_seat_3 }}" euler="0 0 0">
                    <geom name="{{ module_number }}_body_3_seat" type="box" size="5e-3 5e-3 20e-3" pos="0 0 0" rgba="0.1 0.8 0.8 1"/>
                    <site name="{{ module_number }}_ft_sensor_bottom_3" pos="0 0 -20e-3" euler="0 0 -150"/>

                    <body name="{{ module_number }}_body_arm_3_1" pos="0 0 {{ arm_1_pos_z }}" euler="0 0 0">
                        <joint name="{{ module_number }}_joint_3_01" type="hinge" axis="1 0 0" pos="0 0 {{ joint_1_pos_z }}" limited="true" range="1 89" stiffness= "{{ stiff_active }}" damping="{{ damp_active }}" solreflimit="1e-8 3"/> 
                        <geom name="{{ module_number }}_body_arm_3_1" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 0" mass="5e-3" rgba="1 1 0 1"/>

                        <body name="{{ module_number }}_body_arm_3_2" pos="0 0 {{ arm_2_pos_z }}" euler="0 0 180">
                            <joint name="{{ module_number }}_joint_3_12_x" type="hinge" axis="1 0 0" pos="0 0 0" limited="true" range="1 179" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_3_12_y" type="hinge" axis="0 1 0" pos="0 0 0" limited="true" range="-45 45" solreflimit="1e-8 3"/> 
                            <joint name="{{ module_number }}_joint_3_12_z" type="hinge" axis="0 0 1" pos="0 0 0" />
                            <geom name="{{ module_number }}_body_arm_3_2" type="cylinder" size="1.5e-3 {{ arm_length }}" pos="0 0 {{ arm_2_pos_z }}" mass="5e-3" rgba="0 0 1 1"/>

                            <body name="{{ module_number }}_body_arm_3_3" pos="0 0 {{ arm_3_pos_z }}" euler="0 0 0">
                                <joint name="{{ module_number }}_joint_3_23_x" type="hinge" axis="1 0 0" pos="0 0 {{ joint_3_pos_z }}" limited="true" range="1 89" solreflimit="1e-8 3"/>  
                                <geom name="{{ module_number }}_body_arm_3_3" type="cylinder" size="1.5e-3 {{ arm_3_length }}" pos="0 0 0" mass="5e-3" rgba="1 0 1 1"/>
                                
                            </body>  
                        </body>
                    </body>  
                </body>
            </body>
"""

xml_template_top_plate = """
    <body name="{{ module_number }}_top_plate" pos="{{ pos_top_plate }}" euler="{{ euler_top_plate }}">  
        <joint name = "{{ module_number }}_joint_top" type = "free"/>
        <geom name="{{ module_number }}_top_plate" type="box" size="{{ size_top_plate }}" pos="0 0 0" euler="0 0 0" mass="100e-3" rgba="0.8 0 0 0.4"
                    contype= "1" conaffinity="1" condim="3" friction="{{ contact_friction }}" solref="0.01 1" solimp=".95 .99 .0001"/>
        <site name="{{ module_number }}_touch" type="box" size="{{ size_top_plate }}" pos="0 0 0"/>
    </body>  
"""

xml_template_equality = """
        <weld name=" {{ module_number }}_weld_1_top" body1="{{ module_number }}_body_arm_1_3" active="true" body2="{{ module_number }}_top_plate"
            relpose="0 0 {{ equality_z }} -0.6123724 0.6123724 0.3535534 0.3535534" anchor="0 0 0" torquescale="200" solref="1e-8 3" solimp="1 1 0.0001 0.95 6"/>
        <weld name=" {{ module_number }}_weld_2_top" body1="{{ module_number }}_body_arm_2_3" active="true" body2="{{ module_number }}_top_plate"
            relpose="0 0 {{ equality_z }} -0.6123724 0.6123724 -0.3535534 -0.3535534" anchor="0 0 0" torquescale="200" solref="1e-8 3" solimp="1 1 0.0001 0.95 6"/>
        <weld name=" {{ module_number }}_weld_3_top" body1="{{ module_number }}_body_arm_3_3" active="true" body2="{{ module_number }}_top_plate"
            relpose="0 0 {{ equality_z }} 0 0 0.7071068 0.7071068" anchor="0 0 0" torquescale="200" solref="1e-8 3" solimp="1 1 0.0001 0.95 6"/>
"""

xml_template_actuator = """
        <position name="{{ module_number }}_servo1_pos" joint="{{ module_number }}_joint_1_01" kp="{{ kp }}" ctrlrange="0.1 1.57"/>
        <position name="{{ module_number }}_servo2_pos" joint="{{ module_number }}_joint_2_01" kp="{{ kp }}" ctrlrange="0.1 1.57"/>
        <position name="{{ module_number }}_servo3_pos" joint="{{ module_number }}_joint_3_01" kp="{{ kp }}" ctrlrange="0.1 1.57"/>
"""

xml_template_sensor = """
        <touch name="{{ module_number }}_touch" site="{{ module_number }}_touch" />
"""

xml_template_connection_plate = """
        <connect name=" {{ plate1 }}_connect_plate_{{ plate2 }}" body1="{{ plate1 }}_top_plate" active="true" body2="{{ plate2 }}_top_plate"
            anchor="{{ anchor_pos }}" solref="1e-8 3" solimp="1 1 0.0001 0.95 6"/>
"""

xml_template_object = """
    <body name="{{ object_number }}_object" pos="{{ pos_object }}" euler="{{ euler_object }}">  
        <joint name = "{{ object_number }}_joint_object" type = "free"/>
        <geom name="{{ object_number }}_object" type="box" size="{{ size_object }}" pos="0 0 0" euler="0 0 0" mass="{{ mass_object }}" rgba=".9 .9 .1 0.7"
                    contype= "1" conaffinity="1" condim="3" friction="{{ contact_friction }}" solref="0.01 1" solimp=".95 .99 .0001"/>
    </body>  
"""
xml_template_object_flex = """
    <flexcomp type="grid" count="8 8 2" spacing=".05 .05 .01" pos="0.2 0.0 0.14171"
              radius=".0" rgba="0 .7 .7 1" name="fabric_object" dim="3" mass=".5">
      <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001" selfcollide="none"/>
      <elasticity young="5e4" damping=".01" poisson="0"/>
    </flexcomp>
"""

xml_template_contact_floor = """
        <pair name="{{ object_number }}_object_contact_floor" geom1="floor" geom2="{{ object_number }}_object" margin = "{{ contact_margin }}" condim="3" friction="{{ contact_friction }}"/>
"""

xml_template_contact_object = """
        <pair name="{{ object_number1 }}_object_contact_object_{{ object_number2 }}" geom1="{{ object_number1 }}_object" geom2="{{ object_number2 }}_object" margin = "{{ contact_margin }}" condim="3" friction="{{ contact_friction }}"/>
"""

xml_template_contact_plate = """
        <pair name="{{ object_number }}_object_contact_plate_{{ module_number }}" geom1="{{ module_number }}_top_plate" geom2="{{ object_number }}_object" margin = "{{ contact_margin }}" condim="3" friction="{{ contact_friction }}"/>
"""

body_placeholder = "<!-- BODY PLACEHOLDER for the GENERATED_XML -->"
top_placeholder = "<!-- TOP_PLATE PLACEHOLDER for the GENERATED_XML -->"
equality_placeholder = "<!-- EQUALITY PLACEHOLDER for the GENERATED_XML -->"
actuator_placeholder = "<!-- ACTUATOR PLACEHOLDER for the GENERATED_XML -->"
sensor_placeholder = "<!-- SENSOR PLACEHOLDER for the GENERATED_XML -->"
plate_connect_placeholder = "<!-- CONNECT PLATE PLACEHOLDER for the GENERATED_XML -->"
object_placeholder = "<!-- OBJECT PLACEHOLDER for the GENERATED_XML -->"
plate_contact_placeholder = "<!-- PLATE CONTACT PLACEHOLDER for the GENERATED_XML -->"
floor_contact_placeholder = "<!-- FLOOR CONTACT PLACEHOLDER for the GENERATED_XML -->"
object_contact_placeholder = "<!-- OBJECT CONTACT PLACEHOLDER for the GENERATED_XML -->"


class Oripixel_manipulation():
    def __init__(
        self,
        sim_step = 5e-4,
        render_mode=None,
        distance_module_x=120e-3,
        distance_module_y=120e-3,
        dis_group_x=1000e-3,
        dis_group_y=1000e-3,
        kp = 3,
        z0=0,
        x_size=5,
        y_size=5,
        x_group_size=1,
        y_group_size=1,
        h_topplate=121.71e-3,
        contact_friction = [0.4, 0.4, 0.01],
        contact_margin = 1e-3,
        x_object_offset = 0,
        y_object_offset = 0,
        z_object_offset = 20e-3,
        euler_object = [0, 0, 0],
        mass_object = 30e-3,
        size_object = [10e-3, 10e-3, 5e-3],
        size_top_plate = [50e-3, 50e-3, 2e-3], # top plate size, should be half of the actual dimention
        arm_length = 15e-3, #leg length, should be half of the actual length
        arm_3_length = 20.21e-3/2, #top distance, should be half of the actual length
        render_flag = False,
        object_x = [0], # this indicates on which tile in x direction the object will be created
        object_y = [0], # this indicates on which tile in y direction the object will be created
        generate_object_contact = False, # depricated, now the contact is defined using contype= "1" conaffinity="1"
        compliant_stiff = 0.2,
        compliant_damp = 0.1,
        object_material = "rigid",
        
    ):
        self.sim_step = sim_step # sim step size
        self.render_fps = 60
        self.compliant_stiff = compliant_stiff
        self.compliant_damp = compliant_damp
        self.kp = kp
        self.distance_module_x = distance_module_x
        self.distance_module_y = distance_module_y
        self.dis_group_x = dis_group_x
        self.dis_group_y = dis_group_y
        self.z0 = z0
        self.x_size = x_size
        self.y_size = y_size
        self.x_group_size = x_group_size
        self.y_group_size = y_group_size
        self.h_topplate = h_topplate
        self.render_flag = render_flag
        self.object_material = object_material
        # x_size * y_size matrix indicating the height of each top plate
        self.h_matrix = np.array([[h_topplate for _ in range(y_size * y_group_size)] for _ in range(x_size * x_group_size)])
        self.t_settle = 1
        self.u = [0]*(self.x_size * self.y_size)
        self.fps_step_count = 0
        self.tap_torque = [0,0,0]
        self.contact_friction = contact_friction
        self.contact_margin = contact_margin
        self.object_x = object_x
        self.object_y = object_y
        self.x_object_offset = x_object_offset
        self.y_object_offset = y_object_offset
        self.z_object_offset = z_object_offset
        self.euler_object = euler_object
        self.mass_object = mass_object
        self.size_object = size_object
        if self.object_material == "fabric":
            self.size_object[0] = self.size_object[0]/15
            self.size_object[1] = self.size_object[1]/15
        self.generate_object_contact = generate_object_contact
        self.object_number_group = []

        # Position of base seat of each legs
        self.arm_length = arm_length
        self.arm_3_length = arm_3_length
        self.size_top_plate = size_top_plate
        self.pos_seat_1 = [-self.arm_3_length*2*np.sin(np.deg2rad(60)), self.arm_3_length*2*np.cos(np.deg2rad(60)), 20e-3]
        self.pos_seat_2 = [self.arm_3_length*2*np.sin(np.deg2rad(60)), self.arm_3_length*2*np.cos(np.deg2rad(60)), 20e-3]
        self.pos_seat_3 = [0, -self.arm_3_length*2, 20e-3]
        self.arm_1_pos_z = self.arm_length + 20e-3
        self.joint_1_pos_z = -self.arm_length
        self.arm_2_pos_z = self.arm_length
        self.arm_3_pos_z = self.arm_3_length + self.arm_length*2
        self.joint_3_pos_z = -self.arm_3_length
        self.equality_z = self.arm_3_length

        mujoco.set_mju_user_warning(self._mju_user_warning)
        self.render_mode = render_mode
        self.render_fps = 60
        self.window = None
        self.viewport = mujoco.MjrRect(0, 0, 1920, 1080)
        self.fps_step_count = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # FREE camera for manual positioning
        self.cam.lookat[0] = 240e-3  # x-coordinate to look at
        self.cam.lookat[1] = 240e-3  # y-coordinate to look at
        self.cam.lookat[2] = 80e-3  # z-coordinate to look at
        self.cam.distance = 1
        self.cam.azimuth = -90
        self.cam.elevation = -2.5
        self.opt = mujoco.MjvOption()

        self.model_xml = self._xml_generator()
        
    def _rotation_matrix_z(self, theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([[cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1]])
        return R

    def _find_neighbors(self, grid, row, col):
        neighbors = []

        if row > 0:
            neighbors.append(grid[row - 1][col])  # Up
        if row < len(grid) - 1:
            neighbors.append(grid[row + 1][col])  # Down
        if col > 0:
            neighbors.append(grid[row][col - 1])  # Left
        if col < len(grid[0]) - 1:
            neighbors.append(grid[row][col + 1])  # Right

        return neighbors
    
    def generate_xml(self, template_str, data):
        template = Template(template_str)
        xml_code = template.render(data)
        return xml_code

    def _add_template_to_xml(self, existing_xml_content, template, data, placeholder):
        generated_xml = self.generate_xml(template, data)
        updated_xml_content = existing_xml_content.replace(placeholder, f"{generated_xml}\n{placeholder}")
        return updated_xml_content

    def _xml_generator(self):
        existing_xml_content = ""
        data = {"sim_step": self.sim_step}
        existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template, data, existing_xml_content)

        object_number_list = []
        object_number = 0
        module_number = 0
        for i in range(self.x_group_size):
            for j in range(self.y_group_size):

                module_number_group = []
                
                for m in range (self.x_size):
                    for n in range (self.y_size):
                        x_seat = i * self.dis_group_x + m * self.distance_module_x
                        y_seat = j * self.dis_group_y + n * self.distance_module_y
                        z_seat = self.z0
                        z_topplate = z_seat + self.h_matrix[i*m, j*n]
                        
                        # object global location
                        x_object = x_seat + self.x_object_offset
                        y_object = y_seat + self.y_object_offset
                        z_object = z_topplate + self.z_object_offset 

                        module_number_group.append(module_number)

                        data = {
                            "object_number": object_number,
                            "module_number": module_number,
                            "size_top_plate": "{} {} {}".format(self.size_top_plate[0], self.size_top_plate[1], self.size_top_plate[2]),
                            "pos_seat_1": "{:.4f} {:.4f} {:.4f}".format(self.pos_seat_1[0], self.pos_seat_1[1], self.pos_seat_1[2]),
                            "pos_seat_2": "{:.4f} {:.4f} {:.4f}".format(self.pos_seat_2[0], self.pos_seat_2[1], self.pos_seat_2[2]),
                            "pos_seat_3": "{:.4f} {:.4f} {:.4f}".format(self.pos_seat_3[0], self.pos_seat_3[1], self.pos_seat_3[2]),
                            "arm_1_pos_z": self.arm_1_pos_z,
                            "joint_1_pos_z": self.joint_1_pos_z,
                            "arm_2_pos_z": self.arm_2_pos_z,
                            "arm_3_pos_z": self.arm_3_pos_z,
                            "joint_3_pos_z": self.joint_3_pos_z,
                            "arm_length": self.arm_length,
                            "arm_3_length": self.arm_3_length,
                            "equality_z": self.equality_z,
                            "pos_force_seat": "{} {} {}".format(x_seat, y_seat, z_seat),
                            "euler_force_seat": "0 0 0",
                            "pos_top_plate": "{} {} {}".format(x_seat, y_seat, z_topplate),
                            "euler_top_plate": "0 0 0",
                            "stiff_active": self.compliant_stiff,
                            "damp_active": self.compliant_damp,
                            "pos_object": "{} {} {}".format(x_object, y_object, z_object),
                            "euler_object": "{} {} {}".format(*self.euler_object),
                            "mass_object": "{}".format(self.mass_object),
                            "contact_margin": "{}".format(self.contact_margin),
                            "contact_friction": "{} {} {}".format(*self.contact_friction),
                            "size_object": "{} {} {}".format(*self.size_object),
                            "kp": self.kp
                        }
                        if n in self.object_x and m in self.object_y:
                            # Generate object at first tile of each group
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_body, data, body_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_top_plate, data, top_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_equality, data, equality_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_actuator, data, actuator_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_sensor, data, sensor_placeholder)
                            if self.object_material == "rigid":
                                existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_object, data, object_placeholder)
                            elif self.object_material == "fabric":
                                existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_object_flex, data, object_placeholder)
                            # existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_contact_floor, data, floor_contact_placeholder)
                            object_number_list.append(object_number)
                            self.object_number_group.append(object_number)
                            module_number += 1
                            object_number += 1
                        else:
                            # Generate the rest modules
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_body, data, body_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_top_plate, data, top_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_equality, data, equality_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_actuator, data, actuator_placeholder)
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_sensor, data, sensor_placeholder)
                            module_number += 1

                # Generate top plate contact
                if self.generate_object_contact:
                    for u in self.object_number_group:
                        for r in module_number_group:
                            data_plate_contact = {
                                "object_number": u,
                                "module_number": r,
                                "contact_margin": self.contact_margin,
                                "contact_friction": "{} {} {} {} {}".format(*self.contact_friction),
                            }
                            existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_contact_plate, data_plate_contact, plate_contact_placeholder)

        if self.generate_object_contact:
            # Generate contact between each objects
            for u in range(len(object_number_list)):
                for r in range(u + 1, len(object_number_list)):
                    data_object_contact = {
                        "object_number1": object_number_list[u],
                        "object_number2": object_number_list[r],
                        "contact_margin": self.contact_margin,
                        "contact_friction": "{} {} {} {} {}".format(*self.contact_friction),
                    }
                    existing_xml_content = self._add_template_to_xml(existing_xml_content, xml_template_contact_object, data_object_contact, object_contact_placeholder)

            
        return existing_xml_content
    
    def _mju_user_warning(self, e):
        raise ValueError(e)

    def save_xml(self):
        file_name = r"edited_xml_manipulation.xml"
        with open(file_name, "w") as file:
            file.write(self.model_xml)
        return
    
    def reset(self):
        self.model = mujoco.MjModel.from_xml_string(self.model_xml, {})
        self.data = mujoco.MjData(self.model)
        return self._get_states()
    
    def step(self, control_input):
        self.u = control_input
        self.data.ctrl[:] = self.u
     
        mujoco.mj_step(
            self.model, self.data,
            nstep=1
        )

        return self._get_states()

    def read_contact(self):
        num_contacts = self.data.ncon
        contact_data = {}
        for i in range(num_contacts):
            contact = self.data.contact[i]
            
            contact_pos = contact.pos  # this gives the (x, y, z) position of the contact
            
            geom1 = contact.geom1 + 1  # geometry ID of the first object in contact
            geom2 = contact.geom2 + 1  # geometry ID of the second object in contact
            normal_force = contact.frame[:3]  # normal vector of the contact

            if geom1 not in contact_data:
                contact_data[geom1] = []

            contact_data[geom1].append(contact_pos.tolist())

        return contact_data

    def _calculate_corner_points(self, position, size, rotation_matrix):

        """
        Calculate the 2D world coordinates of the corner points of a box projected onto the x-y plane.
        """
        corners_local = np.array([
            [-size[0], -size[1], -size[2]],
            [-size[0], size[1], -size[2]],
            [ size[0], size[1], -size[2]],
            [ size[0], -size[1], -size[2]],
        ])
        
        corners_rotated = np.dot(corners_local, rotation_matrix.T)
        corners_world = corners_rotated + position
        corners_world_2d = corners_world[:, :2]  # Project to 2D (x, y)

        return corners_world_2d

    def _get_square_corners(self):
        grid_squares = []
        for i in range(25):
            square_name = f"{i}_force_seat"
            square_center_pos_x = self.data.body(square_name).xpos[0]
            square_center_pos_y = self.data.body(square_name).xpos[1]
            square_length_x = self.size_top_plate[0]
            square_length_y = self.size_top_plate[1]

            square_corner = [(square_center_pos_x - square_length_x, square_center_pos_y - square_length_y), 
                            (square_center_pos_x - square_length_x, square_center_pos_y + square_length_y), 
                            (square_center_pos_x + square_length_x, square_center_pos_y + square_length_y), 
                            (square_center_pos_x + square_length_x, square_center_pos_y - square_length_y)]
            grid_squares.append(square_corner)

        return grid_squares

    def _get_covered_squares_with_threshold(self, rectangle_corners, grid_squares, threshold=0.5):

        rectangle = Polygon(rectangle_corners)
        covered_squares = []
        
        for idx, square_corners in enumerate(grid_squares):

            square_polygon = Polygon(square_corners)
            intersection = rectangle.intersection(square_polygon)
            
            if intersection.is_empty:
                continue
            
            intersection_area = intersection.area
            square_area = square_polygon.area
            
            if (intersection_area / square_area) >= threshold:
                covered_squares.append(idx)
        
        return covered_squares

    def which_tile(self, threshold):
        tile_corners = self._get_square_corners()
        tile_list = []
        
        for i in self.object_number_group:
            object_name = f"{i}_object"
            object_pos = self.data.body(object_name).xpos
            object_rot = self.data.body(object_name).xmat.reshape(3, 3)
            object_corners = self._calculate_corner_points(object_pos, self.size_object, object_rot)
            tile_covered = self._get_covered_squares_with_threshold(object_corners, tile_corners, threshold)
            tile_list.append(tile_covered)
        
        return tile_list
    
    def get_object_contacts_local(self, object_name):

        body_pos = self.data.body(object_name).xpos
        body_mat = self.data.body(object_name).xmat.reshape(3, 3)

        object_id = self.model.geom(object_name).id

        contact_list = []  # list to store contact information [x, y, z, fx, fy, fz]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (contact.geom1 == object_id) or (contact.geom2 == object_id):

                pos_world = contact.pos
                pos_relative = pos_world - body_pos
                pos_local = body_mat.T @ pos_relative

                if contact.efc_address != -1:

                    force_world = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force_world)  
                    force_world = force_world[:3]

                    force_local = body_mat.T @ force_world

                    contact_data = np.concatenate([pos_local, force_local])
                    contact_list.append(contact_data)

        return contact_list
    
    def render(self):
        if self.render_mode == 'real_time':
            if self.window is None:
                glfw.init()
                self.window = glfw.create_window(
                    self.viewport.width, self.viewport.height,
                    'Oripixel',
                    None, None
                )
                (
                    self.viewport.width,
                    self.viewport.height
                ) = glfw.get_framebuffer_size(self.window)
                glfw.make_context_current(self.window)
                glfw.swap_interval(1)

            self.fps_step_count += 1
            if (
                self.fps_step_count * self.sim_step >
                1 / self.render_fps - 1e-6
            ):
                scn = mujoco.MjvScene(self.model, maxgeom=1000)
                con = mujoco.MjrContext(
                    self.model, mujoco.mjtFontScale.mjFONTSCALE_100
                )
                mujoco.mjv_updateScene(
                    self.model, self.data,
                    self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, scn
                )
                mujoco.mjr_render(self.viewport, scn, con)
                mujoco.mjr_text(
                    mujoco.mjtFont.mjFONT_NORMAL,
                    f't_sim: {self.data.time:.2f}',
                    con,
                    0, 0,
                    0, 0, 0
                )
                self.fps_step_count = 0
                glfw.swap_buffers(self.window)
                glfw.poll_events()

        elif self.render_mode == 'save_video':
            if not hasattr(self, 'gl_context'):
                self.gl_context = mujoco.GLContext(self.viewport.width, self.viewport.height)
                self.gl_context.make_current()
                # Create scene and context once and store them
                self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
                self.context = mujoco.MjrContext(
                    self.model, mujoco.mjtFontScale.mjFONTSCALE_100
                )

            self.gl_context.make_current()
            
            mujoco.mjv_updateScene(
                self.model, self.data,
                self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
            )
            
            mujoco.mjr_render(self.viewport, self.scene, self.context)
            
            sim_time = f"Sim Time: {self.data.time:.2f}"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL.value,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT.value,
                self.viewport,
                sim_time.encode(),
                "".encode(),
                self.context
            )
            
            img = np.empty((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb=img, depth=None, viewport=self.viewport, con=self.context)
            
            return np.flip(img, axis=0)

    def _get_states(self):

        t = self.data.time
        joints_pos = []
        object_pos = []
        object_rot = []
        contacts = [] 

        for i in range(self.x_size * self.y_size):
            for j in range(1, 4): 
                actuator_joint_name = f"{i}_joint_{j}_01"
                temp_joint_angle = self.data.joint(actuator_joint_name).qpos
                joints_pos.extend(temp_joint_angle)

        for i in self.object_number_group:
            if self.object_material == "fabric":
                object_name = "fabric_object_112"
            else:
                object_name = f"{i}_object"

            temp_object_pos = self.data.body(object_name).xpos
            temp_object_rot = R.from_matrix(
                self.data.body(object_name).xmat.reshape(3, 3)
            ).as_euler('ZYX', degrees=True)
            temp_contacts = self.get_object_contacts_local(object_name)

            object_pos.extend(temp_object_pos)
            object_rot.extend(temp_object_rot)

            contacts.extend(temp_contacts)

        contacts_array = np.array(contacts) if contacts else np.empty((0, 6))

        return np.concatenate([
            [t],  # Time as 1D array
            object_pos,  # Object position
            object_rot,  # Object rotation
            contacts_array.flatten(),  # Flatten contacts to 1D [x, y, z, fx, fy, fz]
            self.u,  # Controls
            joints_pos,  # Joint positions
        ])
    
if __name__ == "__main__":    
    sim_step = 5e-4
    render_mode = None
    object_mass = 264e-3
    object_size = [150e-3, 150e-3, 5e-3]
    object_material = "rigid"
    x_object_offset=200e-3
    y_object_offset =0
    z_object_offset = 20e-3
    contact_friction = [0.5, 0.5, 0.01]
    servo_pos_init = [np.radians(85)] * 75

    robot = Oripixel_manipulation(x_size=5, y_size=5, x_group_size=1, y_group_size=1, object_material = object_material,
                                    kp=1, sim_step = sim_step, render_mode=render_mode,
                                    contact_friction = contact_friction, 
                                    mass_object=object_mass, size_object=object_size, 
                                    object_x=[0], object_y=[0], x_object_offset=x_object_offset, y_object_offset = y_object_offset, z_object_offset = z_object_offset)


    robot.save_xml() #save the generated model into a xml file
    s = robot.reset()
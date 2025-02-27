"""
Visualization module for the Tello drone digital twin.
"""

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class DroneRenderer:
    """
    A renderer for visualizing the Tello drone in 3D.
    
    This uses PyGame and OpenGL for rendering.
    """
    
    def __init__(self, render_mode='human', width=800, height=600):
        """Initialize the renderer."""
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.frame_count = 0
        self.camera_distance = 5.0
        self.camera_pitch = 30.0  # degrees
        self.camera_yaw = 45.0    # degrees
        
        # Initialize only if render_mode is specified
        if self.render_mode is not None:
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialize PyGame and OpenGL."""
        pygame.init()
        pygame.display.set_caption("Tello Drone Digital Twin")
        
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode(
                (self.width, self.height),
                DOUBLEBUF | OPENGL
            )
        else:  # render_mode == 'rgb_array'
            self.screen = pygame.Surface((self.width, self.height), DOUBLEBUF | OPENGL)
        
        self.clock = pygame.time.Clock()
        
        # Setup OpenGL
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set light position
        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 10, 1))
        
        # Initialize GLUT for primitives
        glutInit()
    
    def _draw_drone(self, drone):
        """Draw the drone model."""
        # Save current matrix
        glPushMatrix()
        
        # Translate to drone position
        glTranslatef(*drone.position)
        
        # Rotate according to drone orientation
        roll, pitch, yaw = drone.orientation
        glRotatef(np.degrees(yaw), 0, 0, 1)
        glRotatef(np.degrees(pitch), 0, 1, 0)
        glRotatef(np.degrees(roll), 1, 0, 0)
        
        # Draw drone body (a simple box)
        glColor3f(0.8, 0.8, 0.8)  # Light gray
        self._draw_box(0.15, 0.15, 0.05)
        
        # Draw drone arms
        arm_length = 0.1
        arm_width = 0.02
        
        # Front arm (red)
        glColor3f(1.0, 0.0, 0.0)  # Red
        glPushMatrix()
        glTranslatef(arm_length, 0, 0)
        glScalef(arm_length * 2, arm_width, arm_width)
        glutSolidCube(1)
        glPopMatrix()
        
        # Right arm (green)
        glColor3f(0.0, 1.0, 0.0)  # Green
        glPushMatrix()
        glTranslatef(0, arm_length, 0)
        glScalef(arm_width, arm_length * 2, arm_width)
        glutSolidCube(1)
        glPopMatrix()
        
        # Back arm (blue)
        glColor3f(0.0, 0.0, 1.0)  # Blue
        glPushMatrix()
        glTranslatef(-arm_length, 0, 0)
        glScalef(arm_length * 2, arm_width, arm_width)
        glutSolidCube(1)
        glPopMatrix()
        
        # Left arm (yellow)
        glColor3f(1.0, 1.0, 0.0)  # Yellow
        glPushMatrix()
        glTranslatef(0, -arm_length, 0)
        glScalef(arm_width, arm_length * 2, arm_width)
        glutSolidCube(1)
        glPopMatrix()
        
        # Draw propellers
        propeller_radius = 0.08
        propeller_height = 0.01
        
        # Function to draw a propeller
        def draw_propeller(x, y, z, spin_angle=0):
            glPushMatrix()
            glTranslatef(x, y, z)
            glRotatef(spin_angle, 0, 0, 1)
            
            # Draw propeller blades
            glBegin(GL_TRIANGLES)
            glColor3f(0.2, 0.2, 0.2)  # Dark gray
            for i in range(4):  # 4 blades
                angle = i * (360 / 4)
                angle_rad = np.radians(angle)
                next_angle_rad = np.radians(angle + 20)
                
                # Blade triangle
                glVertex3f(0, 0, 0)  # Center
                glVertex3f(
                    propeller_radius * np.cos(angle_rad),
                    propeller_radius * np.sin(angle_rad),
                    0
                )
                glVertex3f(
                    propeller_radius * np.cos(next_angle_rad),
                    propeller_radius * np.sin(next_angle_rad),
                    0
                )
            glEnd()
            
            # Draw center of propeller
            glColor3f(0.5, 0.5, 0.5)  # Gray
            glutSolidCylinder(0.02, propeller_height, 10, 2)
            
            glPopMatrix()
        
        # Draw the four propellers
        # Use motor speeds to determine spin angles
        front_spin = self.frame_count * 10 * (drone.motor_speeds[0] / drone.MAX_MOTOR_SPEED)
        right_spin = -self.frame_count * 10 * (drone.motor_speeds[1] / drone.MAX_MOTOR_SPEED)
        back_spin = self.frame_count * 10 * (drone.motor_speeds[2] / drone.MAX_MOTOR_SPEED)
        left_spin = -self.frame_count * 10 * (drone.motor_speeds[3] / drone.MAX_MOTOR_SPEED)
        
        draw_propeller(arm_length * 2, 0, 0.01, front_spin)
        draw_propeller(0, arm_length * 2, 0.01, right_spin)
        draw_propeller(-arm_length * 2, 0, 0.01, back_spin)
        draw_propeller(0, -arm_length * 2, 0.01, left_spin)
        
        # Restore matrix
        glPopMatrix()
    
    def _draw_box(self, length, width, height):
        """Draw a box centered at the origin."""
        l, w, h = length / 2, width / 2, height / 2
        
        vertices = [
            (l, w, h), (l, w, -h), (l, -w, h), (l, -w, -h),
            (-l, w, h), (-l, w, -h), (-l, -w, h), (-l, -w, -h)
        ]
        
        # Define the vertices of the box
        glBegin(GL_QUADS)
        
        # Top face
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[3])
        glVertex3f(*vertices[2])
        
        # Bottom face
        glVertex3f(*vertices[4])
        glVertex3f(*vertices[5])
        glVertex3f(*vertices[7])
        glVertex3f(*vertices[6])
        
        # Front face
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[5])
        glVertex3f(*vertices[4])
        
        # Back face
        glVertex3f(*vertices[2])
        glVertex3f(*vertices[3])
        glVertex3f(*vertices[7])
        glVertex3f(*vertices[6])
        
        # Left face
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[2])
        glVertex3f(*vertices[6])
        glVertex3f(*vertices[4])
        
        # Right face
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[3])
        glVertex3f(*vertices[7])
        glVertex3f(*vertices[5])
        
        glEnd()
    
    def _draw_ground(self):
        """Draw the ground plane with a grid."""
        grid_size = 20
        grid_step = 1
        
        # Draw gray ground plane
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex3f(-grid_size, -grid_size, 0)
        glVertex3f(grid_size, -grid_size, 0)
        glVertex3f(grid_size, grid_size, 0)
        glVertex3f(-grid_size, grid_size, 0)
        glEnd()
        
        # Draw grid lines
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        
        # Draw grid along X axis
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, -grid_size, 0.01)
            glVertex3f(i, grid_size, 0.01)
        
        # Draw grid along Y axis
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(-grid_size, i, 0.01)
            glVertex3f(grid_size, i, 0.01)
        
        glEnd()
    
    def _draw_axes(self):
        """Draw the world coordinate axes."""
        axis_length = 1.0
        
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(axis_length, 0, 0)
        
        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, axis_length, 0)
        
        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, axis_length)
        
        glEnd()
    
    def _setup_camera(self, drone):
        """Position the camera to follow the drone."""
        # Calculate camera position in spherical coordinates
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        
        # Convert to Cartesian coordinates
        cam_x = drone.position[0] + self.camera_distance * np.cos(yaw_rad) * np.cos(pitch_rad)
        cam_y = drone.position[1] + self.camera_distance * np.sin(yaw_rad) * np.cos(pitch_rad)
        cam_z = drone.position[2] + self.camera_distance * np.sin(pitch_rad)
        
        # Set up the camera view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            cam_x, cam_y, cam_z,          # Camera position
            drone.position[0], drone.position[1], drone.position[2],  # Look at
            0, 0, 1                        # Up vector
        )
    
    def _draw_drone_info(self, drone):
        """Draw text information about the drone state."""
        # This would ideally use a 2D overlay for text rendering
        # For now, we'll skip it as it's more complex in OpenGL
        pass
    
    def _process_events(self):
        """Process PyGame events (keyboard, mouse)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
            # Camera controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.camera_yaw = (self.camera_yaw - 5) % 360
                elif event.key == pygame.K_RIGHT:
                    self.camera_yaw = (self.camera_yaw + 5) % 360
                elif event.key == pygame.K_UP:
                    self.camera_pitch = min(89, self.camera_pitch + 5)
                elif event.key == pygame.K_DOWN:
                    self.camera_pitch = max(-89, self.camera_pitch - 5)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance = max(1.0, self.camera_distance - 0.5)
                elif event.key == pygame.K_MINUS:
                    self.camera_distance = min(20.0, self.camera_distance + 0.5)
        
        return True
    
    def reset(self, drone):
        """Reset the renderer for a new episode."""
        self.frame_count = 0
        # Nothing else needs to be reset for now
    
    def render(self, drone):
        """Render the current state of the environment."""
        if self.render_mode is None:
            return None
        
        if self.render_mode == 'human':
            # Process events for interactive rendering
            if not self._process_events():
                return None
        
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        self._setup_camera(drone)
        
        # Draw the world
        self._draw_ground()
        self._draw_axes()
        self._draw_drone(drone)
        self._draw_drone_info(drone)
        
        # Update the display
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        # Increment frame counter
        self.frame_count += 1
        
        # For rgb_array mode, return the screen capture
        if self.render_mode == 'rgb_array':
            # Read the pixels from the framebuffer
            data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
            
            # OpenGL returns image flipped, so flip it back
            image = np.flipud(image)
            
            return image
        
        return None
    
    def close(self):
        """Close the renderer."""
        if self.render_mode is not None:
            pygame.quit()

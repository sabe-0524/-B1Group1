import taichi as ti
import numpy
import math
import os

ti.init(arch=ti.cpu)  # Use CPU
# ti.init(arch=ti.gpu)  # Use GPU

# Computational domain
domain = ti.Vector.field(2, ti.f32, shape=(2))
domain[0] = (-1.0, -1.0)  # (x_min, y_min)
domain[1] = (1.0, 1.0)     # (x_max, y_max)

# Gravity
gravity = ti.Vector([0.0, -9.8])  # Gravitational acceleration [m/s^2]

# Fluid properties
# Define separate properties for water and alcohol
fluid_density_water = 1000.0       # Density of water [kg/m^3]
fluid_density_alcohol = 800.0      # Density of alcohol [kg/m^3]
fluid_viscosity = 0.001            # Viscosity [Pa*s]
fluid_sound = 10.0                  # (Virtual) speed of sound [m/s]

# Constants
psize = 0.0125                      # Initial particle spacing [m]
re = psize * 2.5                    # Influence radius
pnd0 = ti.field(ti.f32, shape=())
lambda0 = ti.field(ti.f32, shape=())
pnd0_gradP = ti.field(ti.f32, shape=())

# Collision model
collision_dist = psize * 0.9        # Collision detection distance
collision_coef = 0.5                # Restitution coefficient

# Particle type identifiers
type_fluid_water = 0
type_fluid_alcohol = 1
type_wall = 2
type_ghost = 3
type_rigid = 4

# Function to create rectangular particle positions
def create_rectangle(center_x, center_y, width, height):
    array_pos = []
    Nx = int(width / psize + 0.5)
    Ny = int(height / psize + 0.5)
    for ix in range(Nx):
        for iy in range(Ny):
            x_i = center_x - width / 2 + psize * (ix + 0.5)
            y_i = center_y - height / 2 + psize * (iy + 0.5)
            array_pos.append([x_i, y_i])
    array_pos = numpy.array(array_pos, dtype=numpy.float32)
    return array_pos

# Function to create rectangular wall particle positions
def create_rectangle_wall(center_x, center_y, width, height, layer=3):
    array_pos = []
    Nx = int(width / psize + 0.5)
    Ny = int(height / psize + 0.5)

    for ix in range(-layer, Nx + layer):
        for iy in range(-layer, Ny + layer):
            if 0 <= ix < Nx and 0 <= iy < Ny:
                continue

            x_i = center_x - width / 2 + psize * (ix + 0.5)
            y_i = center_y - height / 2 + psize * (iy + 0.5)
            array_pos.append([x_i, y_i])

    array_pos = numpy.array(array_pos, dtype=numpy.float32)
    return array_pos

# Function to calculate the center of a set of particles
def calculate_center(pset):
    center = numpy.zeros(2, dtype=numpy.float32)
    for i in range(len(pset)):
        center += pset[i]
    center /= len(pset)
    return center

# Function to calculate the moment of inertia of a rigid body
def calculate_inertia(pset, mass, center):
    inertia = 0.0
    m = mass / len(pset)  # Mass per particle
    for i in range(len(pset)):
        x = pset[i][0] - center[0]
        y = pset[i][1] - center[1]
        r_sqr = x**2 + y**2
        inertia += m * r_sqr
    return inertia

# Number of rigid bodies
N_rigids = 2

# Rigid body parameters and initial conditions
rigids_pset = [None] * N_rigids  # Particle sets representing rigid shapes
rigids_density = ti.field(ti.f32, shape=(N_rigids))  # Density [kg/m^3]
rigids_mass = ti.field(ti.f32, shape=(N_rigids))     # Mass [kg]
rigids_pos_ini = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # Initial center of mass position
rigids_vel_ini = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # Initial center of mass velocity [m/s]
rigids_angle_ini = ti.field(ti.f32, shape=(N_rigids))          # Initial angle [rad]
rigids_omega_ini = ti.field(ti.f32, shape=(N_rigids))          # Initial angular velocity [rad/s]
rigids_fixcenter = ti.field(ti.i32, shape=(N_rigids))          # Whether to fix the rotation axis 1: fixed, 0: not fixed
rigids_rcenter = ti.Vector.field(2, ti.f32, shape=(N_rigids))   # Rotation center
rigids_inertia = ti.field(ti.f32, shape=(N_rigids))            # Moment of inertia

for k in range(N_rigids):
    rigids_pset[k] = create_rectangle(-0.25 + 0.5 * k, 0.0, 0.2, 0.8)

    rigids_density[k] = 500.0

    rigids_mass[k] = rigids_density[k] * psize**2 * len(rigids_pset[k])

    rigids_pos_ini[k] = calculate_center(rigids_pset[k])

    rigids_vel_ini[k] = (0.0, 0.0)

    rigids_angle_ini[k] = math.radians(0.0)

    rigids_omega_ini[k] = math.radians(0.0)

    rigids_fixcenter[k] = 0

    if rigids_fixcenter[k]:
        rigids_rcenter[k] = (0.0, 0.0)
    else:
        rigids_rcenter[k] = rigids_pos_ini[k]

    rigids_inertia[k] = calculate_inertia(rigids_pset[k], rigids_mass[k], rigids_rcenter[k])

# Rigid body variables
rigids_pos = ti.Vector.field(2, ti.f32, shape=(N_rigids))      # Current center of mass position
rigids_vel = ti.Vector.field(2, ti.f32, shape=(N_rigids))      # Center of mass velocity [m/s]
rigids_angle = ti.field(ti.f32, shape=(N_rigids))              # Angle [rad]
rigids_omega = ti.field(ti.f32, shape=(N_rigids))              # Angular velocity [rad/s]
rigids_force = ti.Vector.field(2, ti.f32, shape=(N_rigids))    # Force
rigids_moment = ti.field(ti.f32, shape=(N_rigids))            # Torque
rigids_rmatrix = ti.Matrix.field(2, 2, ti.f32, shape=(N_rigids))  # Rotation matrices

# Initialize particle data arrays
array_type = []        # Temporary variable to store particle types
array_pos = []         # Temporary variable to store particle positions
array_rigid_id = []    # Temporary variable to store rigid body IDs

# Create fluid particle sets for water and alcohol
# For example, create water in the left half and alcohol in the right half
fluid_pset_water = create_rectangle(-0.4, 0.0, 0.8, 0.8)   # Water particles
fluid_pset_alcohol = create_rectangle(0.4, 0.0, 0.8, 0.8)  # Alcohol particles

for pos in fluid_pset_water:
    array_type.append(type_fluid_water)
    array_pos.append(pos)
    array_rigid_id.append(-1)

for pos in fluid_pset_alcohol:
    array_type.append(type_fluid_alcohol)
    array_pos.append(pos)
    array_rigid_id.append(-1)

# Create wall particles
wall_pset = create_rectangle_wall(0.0, 0.0, 1.6, 1.6)

for pos in wall_pset:
    array_type.append(type_wall)
    array_pos.append(pos)
    array_rigid_id.append(-1)

# Create rigid body particles
for k, pset in enumerate(rigids_pset):
    for pos in pset:
        array_type.append(type_rigid)
        array_pos.append(pos)
        array_rigid_id.append(k)

# Space for injecting particles (ghost particles)
N_space = 10000  # Number of ghost particle slots
for _ in range(N_space):
    array_type.append(type_ghost)
    array_pos.append([0.0, 0.0])
    array_rigid_id.append(-1)

array_type = numpy.array(array_type, dtype=numpy.int32)
array_pos = numpy.array(array_pos, dtype=numpy.float32)
array_rigid_id = numpy.array(array_rigid_id, dtype=numpy.int32)
N_particles = len(array_pos)  # Total number of particles

# Initial particle data
particles_type_ini = ti.field(ti.i32, shape=(N_particles))       # Initial particle types
particles_pos_ini = ti.Vector.field(2, ti.f32, shape=(N_particles))  # Initial positions
particles_rigid_id = ti.field(ti.i32, shape=(N_particles))      # Rigid body ID for each particle
particles_type_ini.from_numpy(array_type)
particles_pos_ini.from_numpy(array_pos)
particles_rigid_id.from_numpy(array_rigid_id)

# Variables
particles_type = ti.field(ti.i32, shape=(N_particles))          # Current particle types
particles_pos = ti.Vector.field(2, ti.f32, shape=(N_particles))  # Current positions
particles_vel = ti.Vector.field(2, ti.f32, shape=(N_particles))  # Current velocities [m/s]
particles_force = ti.Vector.field(2, ti.f32, shape=(N_particles))  # Force per unit volume [N/m^3]
particles_pnd = ti.field(ti.f32, shape=(N_particles))            # Particle number density
particles_pres = ti.field(ti.f32, shape=(N_particles))           # Pressure [Pa]
particles_color = ti.field(ti.i32, shape=(N_particles))          # Color for rendering

# Bucket data for neighbor search
Nx_buckets = int((domain[1]-domain[0]).x / re) + 1
Ny_buckets = int((domain[1]-domain[0]).y / re) + 1
N_buckets = Nx_buckets * Ny_buckets
cnt_max = 100  # Max particles per bucket
table_cnt = ti.field(ti.i32, shape=(Nx_buckets, Ny_buckets))
table_data = ti.field(ti.i32, shape=(Nx_buckets, Ny_buckets, cnt_max))

# Mouse variables
mouse_pos = ti.Vector.field(2, ti.f32, shape=())
mouse_state = ti.field(ti.i32, shape=())  # 0: not pressed, 1: left click

# Fluid injectors for water and alcohol
# Water injectors
array_pos_water = []
array_vel_water = []
for ix, iy in ti.ndrange((-10, 11), (-10, 11)):
    x_i = ix * psize
    y_i = iy * psize
    if x_i**2 + y_i**2 < (psize * 5.0)**2:
        array_pos_water.append([x_i - 0.5, y_i])  # Adjust position as needed
        array_vel_water.append([2.0, 0.0])

array_pos_water = numpy.array(array_pos_water, dtype=numpy.float32)
array_vel_water = numpy.array(array_vel_water, dtype=numpy.float32)
N_injectors_water = len(array_pos_water)
injectors_pos_water = ti.Vector.field(2, ti.f32, shape=(N_injectors_water))  # Water injection positions
injectors_vel_water = ti.Vector.field(2, ti.f32, shape=(N_injectors_water))  # Water injection velocities
injectors_pos_water.from_numpy(array_pos_water)
injectors_vel_water.from_numpy(array_vel_water)

# Alcohol injectors
array_pos_alcohol = []
array_vel_alcohol = []
for ix, iy in ti.ndrange((-10, 11), (-10, 11)):
    x_i = ix * psize
    y_i = iy * psize
    if x_i**2 + y_i**2 < (psize * 5.0)**2:
        array_pos_alcohol.append([x_i + 0.5, y_i])  # Adjust position as needed
        array_vel_alcohol.append([2.0, 0.0])

array_pos_alcohol = numpy.array(array_pos_alcohol, dtype=numpy.float32)
array_vel_alcohol = numpy.array(array_vel_alcohol, dtype=numpy.float32)
N_injectors_alcohol = len(array_pos_alcohol)
injectors_pos_alcohol = ti.Vector.field(2, ti.f32, shape=(N_injectors_alcohol))  # Alcohol injection positions
injectors_vel_alcohol = ti.Vector.field(2, ti.f32, shape=(N_injectors_alcohol))  # Alcohol injection velocities
injectors_pos_alcohol.from_numpy(array_pos_alcohol)
injectors_vel_alcohol.from_numpy(array_vel_alcohol)

# Stability conditions
dt_max = 0.00125        # Maximum dt
courant_max = 0.1       # Max Courant number
diffusion_max = 0.1     # Max diffusion number

# Time variables
step = ti.field(ti.i32, shape=())
time = ti.field(ti.f32, shape=())
dt = ti.field(ti.f32, shape=())
substeps = ti.field(ti.i32, shape=())
substeps_max = 50

# Weight functions
@ti.func
def weight(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = re / r - 1
    return result

@ti.func
def weight_gradP(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = (1 - r / re)**2
    return result

# Update bucket data
@ti.func
def bucket_update():
    for bx, by in ti.ndrange(Nx_buckets, Ny_buckets):
        table_cnt[bx, by] = 0

    for i in range(N_particles):
        if particles_type[i] == type_ghost:
            continue

        pos_i = particles_pos[i]
        bx = int((pos_i - domain[0]).x / re)
        by = int((pos_i - domain[0]).y / re)

        if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
            continue

        l = ti.atomic_add(table_cnt[bx, by], 1)
        if l < cnt_max:
            table_data[bx, by, l] = i

# Initialization kernel
@ti.kernel
def initialize():
    # Compute constant coefficients
    pnd0[None] = 0.0
    lambda0[None] = 0.0
    pnd0_gradP[None] = 0.0

    for jx, jy in ti.ndrange((-5, 6), (-5, 6)):
        if jx == 0 and jy == 0:
            continue
        pos_ij = ti.Vector([psize * jx, psize * jy])
        dist_ij = pos_ij.norm()
        w_ij = weight(dist_ij)
        pnd0[None] += w_ij
        lambda0[None] += w_ij * dist_ij**2
        pnd0_gradP[None] += weight_gradP(dist_ij)

    lambda0[None] /= pnd0[None]

    # Initialize rigid body variables
    for k in range(N_rigids):
        rigids_angle[k] = rigids_angle_ini[k]
        rigids_omega[k] = rigids_omega_ini[k]
        rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])
        if rigids_fixcenter[k]:
            rigids_pos[k] = rigids_rcenter[k] + rigids_rmatrix[k] @ (rigids_pos_ini[k] - rigids_rcenter[k])
            rigids_vel[k].x = -(rigids_pos[k] - rigids_rcenter[k]).y * rigids_omega[k]
            rigids_vel[k].y = (rigids_pos[k] - rigids_rcenter[k]).x * rigids_omega[k]
        else:
            rigids_pos[k] = rigids_pos_ini[k]
            rigids_vel[k] = rigids_vel_ini[k]
            rigids_rcenter[k] = rigids_pos[k]

    # Initialize particles
    for i in range(N_particles):
        particles_type[i] = particles_type_ini[i]
        particles_pos[i] = particles_pos_ini[i]
        particles_vel[i] = (0.0, 0.0)
        if particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            if rigids_fixcenter[k]:
                particles_pos[i] = rigids_rcenter[k] + rigids_rmatrix[k] @ (particles_pos_ini[i] - rigids_rcenter[k])
                particles_vel[i].x = -(particles_pos[i] - rigids_rcenter[k]).y * rigids_omega[k]
                particles_vel[i].y = (particles_pos[i] - rigids_rcenter[k]).x * rigids_omega[k]
            else:
                particles_pos[i] = rigids_pos[k] + rigids_rmatrix[k] @ (particles_pos_ini[i] - rigids_pos_ini[k])
                particles_vel[i].x = rigids_vel[k].x - (particles_pos[i] - rigids_rcenter[k]).y * rigids_omega[k]
                particles_vel[i].y = rigids_vel[k].y + (particles_pos[i] - rigids_rcenter[k]).x * rigids_omega[k]

    # Initialize time variables
    step[None] = 0
    time[None] = 0.0
    dt[None] = dt_max
    substeps[None] = 1

# Pre-update kernel (compute dt and substeps)
@ti.kernel
def preupdate():
    # Compute maximum velocity squared
    vel_sqr_max = 0.0
    for i in range(N_particles):
        vel_sqr_i = particles_vel[i].norm_sqr()
        ti.atomic_max(vel_sqr_max, vel_sqr_i)

    vel_max = ti.math.sqrt(vel_sqr_max)
    acc_max = gravity.norm()

    dt[None] = min(dt_max, dt[None] * 1.5)

    if vel_max > 0.0:
        dt[None] = min(dt[None], courant_max * psize / vel_max)
    if acc_max > 0.0:
        dt[None] = min(dt[None], ti.math.sqrt(courant_max * psize / acc_max))
    if fluid_viscosity > 0.0:
        # Use the higher density for stability
        dt[None] = min(dt[None], diffusion_max * psize**2 * max(fluid_density_water, fluid_density_alcohol) / fluid_viscosity)

    substeps[None] = ti.math.ceil(dt_max / dt[None], dtype=ti.i32)
    if substeps[None] > substeps_max:
        substeps[None] = substeps_max
    else:
        dt[None] = dt_max / substeps[None]

# Update kernel
@ti.kernel
def update():
    bucket_update()
    # Reset forces
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall:
            continue
        # Apply gravity based on particle type
        if particles_type[i] == type_fluid_water:
            particles_force[i] += (fluid_density_water) * gravity
        elif particles_type[i] == type_fluid_alcohol:
            particles_force[i] += (fluid_density_alcohol) * gravity
        elif particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            particles_force[i] += rigids_density[k] * gravity

    # Compute forces and moments on rigid bodies
    rigids_force.fill(0.0)
    rigids_moment.fill(0.0)

    for i in range(N_particles):
        if particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            force_i = particles_force[i] * psize**2

            rigids_force[k] += force_i
            rigids_moment[k] += (particles_pos[i] - rigids_rcenter[k]).cross(force_i)

    # Update rigid bodies' angular velocity, angle, and position
    for k in range(N_rigids):
        rigids_omega[k] += (rigids_moment[k] / rigids_inertia[k]) * dt[None]
        rigids_angle[k] += rigids_omega[k] * dt[None]
        rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])

        if rigids_fixcenter[k]:
            rigids_pos[k] = rigids_rcenter[k] + rigids_rmatrix[k] @ (rigids_pos_ini[k] - rigids_rcenter[k])
            rigids_vel[k].x = -(rigids_pos[k] - rigids_rcenter[k]).y * rigids_omega[k]
            rigids_vel[k].y = (rigids_pos[k] - rigids_rcenter[k]).x * rigids_omega[k]
        else:
            rigids_vel[k] += (rigids_force[k] / rigids_mass[k]) * dt[None]
            rigids_pos[k] += rigids_vel[k] * dt[None]
            rigids_rcenter[k] = rigids_pos[k]

    # Update particles' provisional velocities and positions
    for i in range(N_particles):
        if particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol:
            # Determine density based on type
            density = 0.0
            if particles_type[i] == type_fluid_water:
                density = fluid_density_water
            else:
                density = fluid_density_alcohol
            particles_vel[i] += (particles_force[i] / density) * dt[None]
            particles_pos[i] += particles_vel[i] * dt[None]
        elif particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            if rigids_fixcenter[k]:
                particles_pos[i] = rigids_rcenter[k] + rigids_rmatrix[k] @ (particles_pos_ini[i] - rigids_rcenter[k])
                particles_vel[i].x = -(particles_pos[i] - rigids_rcenter[k]).y * rigids_omega[k]
                particles_vel[i].y = (particles_pos[i] - rigids_rcenter[k]).x * rigids_omega[k]

    # Fluid injection based on mouse input
    if mouse_state[None] == 1:
        # Inject water
        for i in range(N_injectors_water):
            # Find the closest particle to injection position
            j_min = -1
            dist_sqr_min = re**2
            pos_i = mouse_pos[None] + injectors_pos_water[i]
            bx0 = int((pos_i - domain[0]).x / re)
            by0 = int((pos_i - domain[0]).y / re)

            for bx, by in ti.ndrange((bx0 - 1, bx0 + 2), (by0 - 1, by0 + 2)):
                if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                    continue

                for l in range(table_cnt[bx, by]):
                    j = table_data[bx, by, l]
                    pos_ij = particles_pos[j] - pos_i
                    dist_sqr_ij = ti.math.dot(pos_ij, pos_ij)

                    if dist_sqr_min > dist_sqr_ij:
                        dist_sqr_min = dist_sqr_ij
                        j_min = j

            # If overlapping with existing particle, update velocity
            if dist_sqr_min < (psize * 0.99)**2:
                if particles_type[j_min] == type_fluid_water:
                    particles_vel[j_min] = injectors_vel_water[i]
                elif particles_type[j_min] == type_fluid_alcohol:
                    particles_vel[j_min] = injectors_vel_water[i]  # Assuming injectors_vel_water applies to water
                continue

            # Find a ghost particle slot
            j_ghost = -1
            for j in range(N_particles):
                if particles_type[j] == type_ghost:
                    # Assign as water
                    type_j = ti.atomic_min(particles_type[j], type_fluid_water)
                    if type_j == type_ghost:
                        j_ghost = j
                        break

            # If found, inject water particle
            if j_ghost != -1:
                particles_type[j_ghost] = type_fluid_water
                particles_vel[j_ghost] = injectors_vel_water[i]
                particles_pos[j_ghost] = pos_i

        # Inject alcohol
        for i in range(N_injectors_alcohol):
            # Find the closest particle to injection position
            j_min = -1
            dist_sqr_min = re**2
            pos_i = mouse_pos[None] + injectors_pos_alcohol[i]
            bx0 = int((pos_i - domain[0]).x / re)
            by0 = int((pos_i - domain[0]).y / re)

            for bx, by in ti.ndrange((bx0 - 1, bx0 + 2), (by0 - 1, by0 + 2)):
                if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                    continue

                for l in range(table_cnt[bx, by]):
                    j = table_data[bx, by, l]
                    pos_ij = particles_pos[j] - pos_i
                    dist_sqr_ij = ti.math.dot(pos_ij, pos_ij)

                    if dist_sqr_min > dist_sqr_ij:
                        dist_sqr_min = dist_sqr_ij
                        j_min = j

            # If overlapping with existing particle, update velocity
            if dist_sqr_min < (psize * 0.99)**2:
                if particles_type[j_min] == type_fluid_water:
                    particles_vel[j_min] = injectors_vel_alcohol[i]  # Assign alcohol velocity
                elif particles_type[j_min] == type_fluid_alcohol:
                    particles_vel[j_min] = injectors_vel_alcohol[i]
                continue

            # Find a ghost particle slot
            j_ghost = -1
            for j in range(N_particles):
                if particles_type[j] == type_ghost:
                    # Assign as alcohol
                    type_j = ti.atomic_min(particles_type[j], type_fluid_alcohol)
                    if type_j == type_ghost:
                        j_ghost = j
                        break

            # If found, inject alcohol particle
            if j_ghost != -1:
                particles_type[j_ghost] = type_fluid_alcohol
                particles_vel[j_ghost] = injectors_vel_alcohol[i]
                particles_pos[j_ghost] = pos_i

    # Compute particle number density and pressure
    for i in range(N_particles):
        particles_pnd[i] = 0.0
        particles_pres[i] = 0.0
        if particles_type[i] == type_ghost:
            continue

        # Neighbor search
        bx0 = int((particles_pos[i] - domain[0]).x / re)
        by0 = int((particles_pos[i] - domain[0]).y / re)
        for bx, by in ti.ndrange((bx0 - 1, bx0 + 2), (by0 - 1, by0 + 2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx, by]):
                j = table_data[bx, by, l]
                if j == i:
                    continue
                pos_ij = particles_pos[j] - particles_pos[i]
                dist_ij = pos_ij.norm()
                if (particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol or particles_type[j] == type_fluid_water or particles_type[j] == type_fluid_alcohol):
                    if dist_ij < re:
                        # Apply viscosity force based on particle types
                        if ((particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol) and (particles_type[j] == type_fluid_water or particles_type[j] == type_fluid_alcohol)):
                            # Determine average density
                            density_i = 0.0
                            density_j = 0.0
                            if particles_type[i] == type_fluid_water:
                                density_i = fluid_density_water
                            else:
                                density_i = fluid_density_alcohol
                            if particles_type[j] == type_fluid_water:
                                density_j = fluid_density_water
                            else:
                                density_j = fluid_density_alcohol
                            average_density = 0.5 * (density_i + density_j)
                            particles_force[i] += fluid_viscosity * 4.0 / (pnd0[None] * lambda0[None]) * (particles_vel[j] - particles_vel[i]) * weight(dist_ij)

                # Particle number density
                if dist_ij < re:
                    particles_pnd[i] += weight(dist_ij)

        # Compute pressure based on number density
        if particles_pnd[i] > pnd0[None]:
            # Determine density based on particle type
            density = 0.0
            if particles_type[i] == type_fluid_water:
                density = fluid_density_water
            elif particles_type[i] == type_fluid_alcohol:
                density = fluid_density_alcohol
            else:
                density = fluid_density_water  # Default
            particles_pres[i] = (
                density * fluid_sound**2 * (particles_pnd[i] - pnd0[None]) / pnd0[None]
            )
        else:
            particles_pres[i] = 0.0

    # Recompute forces (second pass) with pressure and collision
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall or particles_type[i] == type_ghost:
            continue

        # Neighbor search
        bx0 = int((particles_pos[i] - domain[0]).x / re)
        by0 = int((particles_pos[i] - domain[0]).y / re)
        for bx, by in ti.ndrange((bx0 - 1, bx0 + 2), (by0 - 1, by0 + 2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx, by]):
                j = table_data[bx, by, l]
                if j == i:
                    continue
                pos_ij = particles_pos[j] - particles_pos[i]
                dist_ij = pos_ij.norm()

                # Pressure term
                if (particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol or particles_type[j] == type_fluid_water or particles_type[j] == type_fluid_alcohol):
                    if dist_ij < re and dist_ij > 1e-6:
                        # Compute pressure based on particle types
                        if particles_type[i] == type_fluid_water:
                            density_i = fluid_density_water
                        elif particles_type[i] == type_fluid_alcohol:
                            density_i = fluid_density_alcohol
                        else:
                            density_i = fluid_density_water  # Default

                        if particles_type[j] == type_fluid_water:
                            density_j = fluid_density_water
                        elif particles_type[j] == type_fluid_alcohol:
                            density_j = fluid_density_alcohol
                        else:
                            density_j = fluid_density_water  # Default

                        # Compute average pressure
                        avg_pres = 0.5 * (particles_pres[i] + particles_pres[j])
                        force_pressure = - (avg_pres + avg_pres) * pos_ij / (dist_ij**2 + 1e-6) * weight_gradP(dist_ij)
                        particles_force[i] += force_pressure

                # Collision model
                if (particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol or particles_type[j] == type_fluid_water or particles_type[j] == type_fluid_alcohol):
                    if dist_ij < collision_dist and dist_ij > 1e-6:
                        vel_ij = particles_vel[j] - particles_vel[i]
                        normal_ij = pos_ij / dist_ij
                        tmp = normal_ij.dot(vel_ij)

                        if tmp < 0.0:
                            # Determine mass based on particle types
                            m_ij = 0.0
                            density_i = 0.0
                            density_j = 0.0
                            if ((particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol) and (particles_type[j] == type_fluid_water or particles_type[j] == type_fluid_alcohol)):
                                # Both fluids
                                if particles_type[i] == type_fluid_water:
                                    density_i = fluid_density_water
                                else:
                                    density_i = fluid_density_alcohol
                                if particles_type[j] == type_fluid_water:
                                    density_j = fluid_density_water
                                else:
                                    density_j = fluid_density_alcohol
                                m_ij = 0.5 * (density_i + density_j) / 2
                            elif ((particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol) and
                                  particles_type[j] == type_wall):
                                # Fluid and wall
                                if particles_type[i] == type_fluid_water:
                                    density_i = fluid_density_water
                                else:
                                    density_i = fluid_density_alcohol
                                m_ij = density_i
                            elif ((particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol) and
                                  particles_type[j] == type_rigid):
                                # Fluid and rigid
                                if particles_type[i] == type_fluid_water:
                                    density_i = fluid_density_water
                                else:
                                    density_i = fluid_density_alcohol
                                density_j = rigids_density[particles_rigid_id[j]]
                                m_ij = (density_i * density_j) / (density_i + density_j)
                            elif (particles_type[i] == type_rigid and
                                  (particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol)):
                                # Rigid and fluid
                                if particles_type[j] == type_fluid_water:
                                    density_j = fluid_density_water
                                else:
                                    density_j = fluid_density_alcohol
                                density_i = rigids_density[particles_rigid_id[i]]
                                m_ij = (density_i * density_j) / (density_i + density_j)
                            else:
                                m_ij = fluid_density_water  # Default

                            # Apply restitution
                            particles_force[i] += normal_ij * (1.0 + collision_coef) * m_ij * tmp / dt_max

                # Rigid body repulsion
                if (particles_type[i] == type_rigid and particles_type[j] == type_rigid):
                    k_i = particles_rigid_id[i]
                    k_j = particles_rigid_id[j]
                    if k_i == k_j:
                        continue

                    len_ij = psize - dist_ij  # Overlap length
                    if len_ij > 0.0:
                        normal_ij = pos_ij.normalized()
                        m_i = rigids_density[k_i]
                        m_j = rigids_density[k_j]
                        m_ij = 2 * m_i * m_j / (m_i + m_j)

                        particles_force[i] -= normal_ij * (m_ij * len_ij / dt_max**2)

                        vel_ij = particles_vel[j] - particles_vel[i]
                        tmp = normal_ij.dot(vel_ij)  # Relative velocity in normal direction
                        if tmp < 0.0:
                            particles_force[i] += normal_ij * (m_ij * 10 * tmp / dt_max)

                        # Friction force
                        tau_ij = ti.Vector([-normal_ij.y, normal_ij.x])  # Tangent vector
                        tmp2 = tau_ij.dot(vel_ij)  # Relative velocity in tangent direction
                        particles_force[i] += tau_ij * (m_ij * tmp2 / dt_max)

                # Rigid body and wall collision
                if ((particles_type[i] == type_rigid and particles_type[j] == type_wall) or
                    (particles_type[j] == type_rigid and particles_type[i] == type_wall)):
                    if particles_type[i] == type_rigid:
                        k = particles_rigid_id[i]
                    else:
                        k = particles_rigid_id[j]
                    len_ij = psize - dist_ij  # Overlap length

                    if len_ij > 0.0:
                        normal_ij = pos_ij.normalized()
                        m_ij = 2 * rigids_density[k]

                        particles_force[i] -= normal_ij * (m_ij * len_ij / dt_max**2)

                        vel_ij = particles_vel[j] - particles_vel[i]
                        tmp = normal_ij.dot(vel_ij)  # Relative velocity in normal direction
                        if tmp < 0.0:
                            particles_force[i] += normal_ij * (m_ij * 10 * tmp / dt_max)

                        # Friction force
                        tau_ij = ti.Vector([-normal_ij.y, normal_ij.x])  # Tangent vector
                        tmp2 = tau_ij.dot(vel_ij)  # Relative velocity in tangent direction
                        particles_force[i] += tau_ij * (m_ij * tmp2 / dt_max)

    # Recompute forces and moments on rigid bodies
    rigids_force.fill(0.0)
    rigids_moment.fill(0.0)

    for i in range(N_particles):
        if particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            force_i = particles_force[i] * psize**2

            rigids_force[k] += force_i
            rigids_moment[k] += (particles_pos[i] - rigids_rcenter[k]).cross(force_i)

    # Update rigid bodies' angular velocity, angle, and position
    for k in range(N_rigids):
        rigids_omega[k] += (rigids_moment[k] / rigids_inertia[k]) * dt[None]
        rigids_angle[k] += rigids_omega[k] * dt[None]
        rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])

        if rigids_fixcenter[k]:
            rigids_pos[k] = rigids_rcenter[k] + rigids_rmatrix[k] @ (rigids_pos_ini[k] - rigids_rcenter[k])
            rigids_vel[k].x = -(rigids_pos[k] - rigids_rcenter[k]).y * rigids_omega[k]
            rigids_vel[k].y = (rigids_pos[k] - rigids_rcenter[k]).x * rigids_omega[k]
        else:
            rigids_vel[k] += (rigids_force[k] / rigids_mass[k]) * dt[None]
            rigids_pos[k] += rigids_vel[k] * dt[None]
            rigids_rcenter[k] = rigids_pos[k]

    # Update particles' velocity and position based on forces
    for i in range(N_particles):
        density = 0.0
        if particles_type[i] == type_fluid_water or particles_type[i] == type_fluid_alcohol:
            # Determine density based on type
            if particles_type[i] == type_fluid_water:
                density = fluid_density_water
            else:
                density = fluid_density_alcohol
            particles_vel[i] += (particles_force[i] / density) * dt[None]
            particles_pos[i] += particles_vel[i] * dt[None]
        elif particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            if rigids_fixcenter[k]:
                particles_pos[i] = rigids_rcenter[k] + rigids_rmatrix[k] @ (particles_pos_ini[i] - rigids_rcenter[k])
                particles_vel[i].x = -(particles_pos[i] - rigids_rcenter[k]).y * rigids_omega[k]
                particles_vel[i].y = (particles_pos[i] - rigids_rcenter[k]).x * rigids_omega[k]
            else:
                particles_pos[i] = rigids_pos[k] + rigids_rmatrix[k] @ (particles_pos_ini[i] - rigids_pos_ini[k])
                particles_vel[i].x = rigids_vel[k].x - (particles_pos[i] - rigids_rcenter[k]).y * rigids_omega[k]
                particles_vel[i].y = rigids_vel[k].y + (particles_pos[i] - rigids_rcenter[k]).x * rigids_omega[k]

    # Handle particles outside the domain
    for i in range(N_particles):
        if particles_type[i] != type_fluid_water and particles_type[i] != type_fluid_alcohol:
            continue
        if (particles_pos[i].x < domain[0].x or particles_pos[i].x >= domain[1].x or
            particles_pos[i].y < domain[0].y or particles_pos[i].y >= domain[1].y):
            particles_type[i] = type_ghost
            particles_pos[i] = (0.0, 0.0)
            particles_vel[i] = (0.0, 0.0)

    # Advance time
    step[None] += 1
    time[None] += dt[None]

# Color computation kernel
@ti.kernel
def update_colors():
    for i in range(N_particles):
        if particles_type[i] == type_fluid_water:
            # Water: Greenish color
            a = ti.math.clamp(particles_vel[i].norm(), 0.0, 1.0)
            r = 0.0
            g = a
            b = 1.0 - a
            particles_color[i] = 0x00FF00 * ti.i32(g * 255) + 0x0000FF * ti.i32(b * 255)
        elif particles_type[i] == type_fluid_alcohol:
            # Alcohol: Reddish color
            a = ti.math.clamp(particles_vel[i].norm(), 0.0, 1.0)
            r = a
            g = 0.0
            b = 1.0 - a
            particles_color[i] = 0xFF0000 * ti.i32(r * 255) + 0x0000FF * ti.i32(b * 255)
        elif particles_type[i] == type_rigid:
            particles_color[i] = 0xFFFF00  # Yellow for rigid bodies
        elif particles_type[i] == type_wall:
            particles_color[i] = 0x808080  # Gray for walls
        else:
            particles_color[i] = 0xFFFFFF  # White for ghosts and others

# Initialize the simulation
initialize()
print('pnd0 =', pnd0[None])
print('lambda0 =', lambda0[None])

# Create GUI
window_size = (640, 640)
window_title = os.path.basename(__file__)
scale_to_pixel = window_size[0] / (domain[1] - domain[0]).x
gui = ti.GUI(window_title, window_size)

# Widgets
slider_forwards = gui.slider('fast-forward', 1, 20)
slider_forwards.value = 10

while gui.running:
    # Get mouse information
    cursor_x, cursor_y = gui.get_cursor_pos()
    mouse_pos[None][0] = domain[0].x + (domain[1] - domain[0]).x * cursor_x
    mouse_pos[None][1] = domain[0].y + (domain[1] - domain[0]).y * cursor_y
    mouse_state[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        mouse_state[None] = 1
    forwards = int(slider_forwards.value)
    slider_forwards.value = forwards

    # Advance time
    for frame in range(forwards):
        preupdate()
        for substep in range(substeps[None]):
            update()

    # Handle keyboard input
    for e in gui.get_events(gui.PRESS):
        if e.key == ti.GUI.ESCAPE:
            initialize()

    # Update colors
    update_colors()

    # Render injection points
    if mouse_state[None] == 1:
        # Render water injectors (blue)
        J_water = injectors_pos_water.to_numpy()
        J_water[:, 0] = (J_water[:, 0] + mouse_pos[None].x - domain[0].x) / (domain[1] - domain[0]).x
        J_water[:, 1] = (J_water[:, 1] + mouse_pos[None].y - domain[0].y) / (domain[1] - domain[0]).y
        gui.circles(J_water, radius=psize * 0.5 * scale_to_pixel, color=0x0000FF)

        # Render alcohol injectors (red)
        J_alcohol = injectors_pos_alcohol.to_numpy()
        J_alcohol[:, 0] = (J_alcohol[:, 0] + mouse_pos[None].x - domain[0].x) / (domain[1] - domain[0]).x
        J_alcohol[:, 1] = (J_alcohol[:, 1] + mouse_pos[None].y - domain[0].y) / (domain[1] - domain[0]).y
        gui.circles(J_alcohol, radius=psize * 0.5 * scale_to_pixel, color=0xFF0000)

    # Render particles
    X = particles_pos.to_numpy()
    X[:, 0] = (X[:, 0] - domain[0].x) / (domain[1] - domain[0]).x
    X[:, 1] = (X[:, 1] - domain[0].y) / (domain[1] - domain[0]).y
    T = particles_type.to_numpy()
    C = particles_color.to_numpy()
    gui.circles(X[T != type_ghost, :], radius=psize * 0.5 * scale_to_pixel, color=C[T != type_ghost])

    # Render text information
    gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}, substeps = {substeps[None]}', 
             (0.0, 1.0), font_size=20, color=0xFFFFFF)
    gui.text(f'Particles: {numpy.count_nonzero(T == type_fluid_water)} / {numpy.count_nonzero(T == type_fluid_alcohol)} / {numpy.count_nonzero(T == type_wall)} / {numpy.count_nonzero(T == type_ghost)}', 
             (0.0, 0.975), font_size=20, color=0xFFFFFF)

    # Show GUI
    gui.show()

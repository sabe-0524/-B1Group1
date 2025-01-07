import taichi as ti
import numpy
import math
import os
ti.init(arch=ti.cpu) # CPUを使う場合
# ti.init(arch=ti.gpu) # GPUを使う場合

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

# 重力
gravity = ti.Vector([0.0,-9.8]) # 重力加速度[m/s^2]

# 流体の物性値
fluid_density = ti.field(ti.f32, shape=(2))  # 密度[kg/m^3]
fluid_density[0] = 1000 # 水
fluid_density[1] = 789 # アルコール
fluid_viscosity = 0.001 # 粘性係数[Pa*s]
fluid_sound = 10.0 # (仮想的な)音速[m/s]

# 定数
psize = 0.025 # 初期粒子間距離[m]
re = psize*2.5 # 影響半径
pnd0 = ti.field(ti.f32,shape=())
lambda0 = ti.field(ti.f32,shape=())
pnd0_gradP = ti.field(ti.f32,shape=())

# 衝突モデル
collision_dist = psize*0.9 # 衝突判定距離
collision_coef = 0.5 # 反発係数

# 粒子タイプ識別用定数
type_fluid = 0
type_wall = 1
type_ghost = 2
type_rigid = 3

# 矩形形状の粒子配置を返す関数
def create_rectangle(center_x,center_y,width,height):
    array_pos = []
    Nx = int(width/psize+0.5)
    Ny = int(height/psize+0.5)
    for ix in range(Nx):
        for iy in range(Ny):
            x_i = center_x-width/2+psize*(ix+0.5)
            y_i = center_y-height/2+psize*(iy+0.5)
            array_pos.append([x_i,y_i])
    array_pos = numpy.array(array_pos,dtype=numpy.float32)
    return array_pos

# 矩形タンク壁面の粒子配置を返す関数
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

# 重心を計算する関数
def calculate_center(pset):
    center = numpy.zeros(2, dtype=numpy.float32)
    for i in range(len(pset)):
        center += pset[i]
    center /= len(pset)
    return center

# 慣性モーメントを計算する関数
def calculate_inertia(pset, mass, center):
    inertia = 0.0
    m = mass / len(pset)  # 粒子1つあたりの質量
    for i in range(len(pset)):
        x = pset[i][0] - center[0]
        y = pset[i][1] - center[1]
        r_sqr = x**2 + y**2
        inertia += m * r_sqr
    return inertia

# 剛体の数
N_rigids = 2

# 剛体のパラメータと初期条件
rigids_pset = [None] * N_rigids  # 剛体形状を表す粒子集合
rigids_density = ti.field(ti.f32, shape=(N_rigids))  # 密度[kg/m^3]
rigids_mass = ti.field(ti.f32, shape=(N_rigids))  # 質量[kg]
rigids_pos_ini = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 初期の重心位置
rigids_vel_ini = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 初期の重心速度[m/s]
rigids_angle_ini = ti.field(ti.f32, shape=(N_rigids))  # 初期の角度[rad]
rigids_omega_ini = ti.field(ti.f32, shape=(N_rigids))  # 初期の各速度[rad/s]
rigids_fixcenter = ti.field(ti.i32, shape=(N_rigids))  # 回転軸を固定するかどうか 1:固定する 0:固定しない
rigids_rcenter = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 回転中心
rigids_inertia = ti.field(ti.f32, shape=(N_rigids))  # 慣性モーメント

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

# 剛体の変数
rigids_pos = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 現在の重心位置

rigids_vel = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 重心速度[m/s]

rigids_angle = ti.field(ti.f32, shape=(N_rigids))  # 角度[rad]

rigids_omega = ti.field(ti.f32, shape=(N_rigids))  # 角速度[rad/s]

rigids_force = ti.Vector.field(2, ti.f32, shape=(N_rigids))  # 力

rigids_moment = ti.field(ti.f32, shape=(N_rigids))  # 力のモーメント

rigids_rmatrix = ti.Matrix.field(2, 2, ti.f32, shape=(N_rigids))  # 回転行列



# 初期粒子データ用の配列を作成する
array_type = [] # 粒子タイプを格納する一時変数
array_pos = [] # 粒子位置を格納する一時変数
array_rigid_id = [] # 剛体番号を格納する一時変数
array_fluid_id = []

water_pset = create_rectangle(100.0,0.0,0.8,0.8) # 水
for i in range(len(water_pset)):
    array_type.append(type_fluid)
    array_pos.append(water_pset[i])
    array_rigid_id.append(-1)
    array_fluid_id.append(0)

alcohol_pset = create_rectangle(-100.0,0.0,0.8,0.8) # アルコール
for i in range(len(alcohol_pset)):
    array_type.append(type_fluid)
    array_pos.append(alcohol_pset[i])
    array_rigid_id.append(-1)
    array_fluid_id.append(1)
    
wall_pset = create_rectangle_wall(0.0, 0.0, 1.6, 1.6)

for i in range(len(wall_pset)):
    array_type.append(type_wall)
    array_pos.append(wall_pset[i])
    array_rigid_id.append(-1)
    array_fluid_id.append(-1)

    
# for k, pset in enumerate(rigids_pset):
#     for i in range(len(pset)):
#         array_type.append(type_rigid)
#         array_pos.append(pset[i])
#         array_rigid_id.append(k)
#         array_fluid_id.append(-1)

    
N_space = 10000 # 流入粒子用の空きスロット数
for i in range(N_space):
    array_type.append(type_ghost)
    array_pos.append([0.0,0.0])
    array_rigid_id.append(-1)
    array_fluid_id.append(-1)

array_type = numpy.array(array_type,dtype=numpy.int32)
array_pos = numpy.array(array_pos,dtype=numpy.float32)
array_rigid_id = numpy.array(array_rigid_id,dtype=numpy.int32)
array_fluid_id = numpy.array(array_fluid_id,dtype=numpy.int32)
N_particles = len(array_pos) # 粒子数

# 初期粒子データ
particles_type_ini = ti.field(ti.i32,shape=(N_particles)) # 初期粒子タイプ
particles_pos_ini = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 初期の位置ベクトル
particles_rigid_id = ti.field(ti.i32,shape=(N_particles)) # 粒子が属する剛体の番号
particles_fluid_id = ti.field(ti.i32,shape=(N_particles)) # 粒子が属する液体の番号
particles_type_ini.from_numpy(array_type)
particles_pos_ini.from_numpy(array_pos)
particles_rigid_id.from_numpy(array_rigid_id)
particles_fluid_id.from_numpy(array_fluid_id)

# 変数
particles_type = ti.field(ti.i32,shape=(N_particles)) # 粒子タイプ
particles_pos = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 位置ベクトル
particles_vel = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 速度ベクトル[m/s]
particles_force = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 単位体積あたりの力[N/m^3]
particles_pnd = ti.field(ti.f32,shape=(N_particles)) # 粒子数密度
particles_pres = ti.field(ti.f32,shape=(N_particles)) # 圧力[Pa]
particles_color = ti.field(ti.i32,shape=(N_particles)) # 描画する色
particles_dispersion = ti.field(ti.i32,shape=(N_particles))#粒子密度の分散
particles_entropy = ti.field(ti.f32, shape=(N_particles))#エントロピー
total_entropy = ti.field(ti.f32, shape=())

# 流入する流体の種類を保持するフィールド
inject_fluid_id = ti.field(ti.i32, shape=())
inject_rigid_id = ti.field(ti.i32, shape=())
inject_fluid_id[None] = 0  # 初期値: 水
inject_rigid_id[None] = 0  # 初期値: 0番目の剛体

# バケットデータ
Nx_buckets = int((domain[1]-domain[0]).x/re)+1
Ny_buckets = int((domain[1]-domain[0]).y/re)+1
N_buckets = Nx_buckets*Ny_buckets
cnt_max = 100 # バケット内の最大粒子数
table_cnt = ti.field(ti.i32,shape=(Nx_buckets,Ny_buckets))
table_data = ti.field(ti.i32,shape=(Nx_buckets,Ny_buckets,cnt_max))

# マウス変数
mouse_pos = ti.Vector.field(2,ti.f32,shape=())
mouse_state = ti.field(ti.i32,shape=()) # 0:押されていない, 1:左クリック

# 流体の流入口
array_pos = []
array_vel = []
for ix,iy in ti.ndrange((-10,11),(-10,11)):
    x_i = ix*psize
    y_i = iy*psize
    if x_i**2+y_i**2 < (psize*5.0)**2:
        array_pos.append([x_i,y_i])
        array_vel.append([2.0,0.0])
array_pos = numpy.array(array_pos,dtype=numpy.float32)
array_vel = numpy.array(array_vel,dtype=numpy.float32)
N_injectors = len(array_pos)
injectors_pos = ti.Vector.field(2,ti.f32,shape=(N_injectors)) # 流入位置
injectors_vel = ti.Vector.field(2,ti.f32,shape=(N_injectors)) # 流入速度
injectors_pos.from_numpy(array_pos)
injectors_vel.from_numpy(array_vel)

# 安定条件
dt_max = 0.0025 # dtの上限値
courant_max = 0.1 # 最大クーラン数
diffusion_max = 0.1 # 最大拡散数

# 時間
step = ti.field(ti.i32,shape=())
time = ti.field(ti.f32,shape=())
dt = ti.field(ti.f32,shape=())
substeps = ti.field(ti.i32,shape=())
substeps_max = 50

# 重み関数
@ti.func
def weight(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = re/r-1
    return result

# 圧力勾配用重み関数
@ti.func
def weight_gradP(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = (1-r/re)**2
    return result

# エントロピー計算カーネル
@ti.kernel
def compute_entropy():
    for i in range(N_particles):
        if particles_type[i] != type_fluid:
            particles_entropy[i] = 0.0
            continue
        
        count0 = 0
        count1 = 0
        # 粒子の近傍を探索
        bx0 = int((particles_pos[i] - domain[0]).x / re)
        by0 = int((particles_pos[i] - domain[0]).y / re)
        
        for bx, by in ti.ndrange((bx0 - 1, bx0 + 2), (by0 - 1, by0 + 2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx, by]):
                j = table_data[bx, by, l]
                if particles_type[j] != type_fluid:
                    continue
                if particles_fluid_id[j] == 0:
                    count0 += 1
                elif particles_fluid_id[j] == 1:
                    count1 += 1
        
        total = count0 + count1
        entropy = 0.0  # 初期化
        
        if total > 0:
            p0 = count0 / total
            p1 = count1 / total
            if p0 > 0:
                entropy -= p0 * ti.log(p0)
            if p1 > 0:
                entropy -= p1 * ti.log(p1)
        
        particles_entropy[i] = entropy

# エントロピーの合計を計算するカーネル
@ti.kernel
def compute_total_entropy():
    total_entropy[None] = 0.0
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            total_entropy[None] += particles_entropy[i]


# バケットデータ更新関数
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
        table_data[bx, by, l] = i


# 初期化カーネル
@ti.kernel
def initialize():
    # 定数係数の計算
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

    # 剛体の変数の初期化
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

    # 粒子の変数の初期化
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

        
    # 時間変数の初期化
    step[None] = 0
    time[None] = 0.0
    dt[None] = dt_max
    substeps[None] = 1

# 事前計算カーネル
@ti.kernel
def preupdate():
    # 最大速度の計算
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
        dt[None] = min(dt[None], diffusion_max * psize**2 * fluid_density[0] / fluid_viscosity)

    substeps[None] = ti.math.ceil(dt_max / dt[None], dtype=ti.i32)
    if substeps[None] > substeps_max:
        substeps[None] = substeps_max
    else:
        dt[None] = dt_max / substeps[None]


# 更新カーネル
@ti.kernel
def update():
    bucket_update()
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall:
            continue
        # 重力
        if particles_type[i] == type_fluid:
            k = particles_fluid_id[i]
            particles_force[i] += fluid_density[k] * gravity

        elif particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            particles_force[i] += rigids_density[k] * gravity

    # 剛体に作用する力の和と力のモーメントの和の計算
    rigids_force.fill(0.0)
    rigids_moment.fill(0.0)

    for i in range(N_particles):
        if particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            force_i = particles_force[i] * psize**2

            rigids_force[k] += force_i
            rigids_moment[k] += (particles_pos[i] - rigids_rcenter[k]).cross(force_i)

    # 剛体の仮角速度と仮角度と仮重心速度と仮重心位置
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

    # 粒子の仮速度と仮位置
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            k = particles_fluid_id[i]
            particles_vel[i] += (particles_force[i] / fluid_density[k]) * dt[None]
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

    # 流入
    if mouse_state[None] == 1:
        if inject_rigid_id[None] == 0:
            for i in range(N_injectors):
                # 流入口に近接する粒子を探す
                j_min = -1
                dist_sqr_min = re**2
                pos_i = mouse_pos[None] + injectors_pos[i]
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

                # 流入口に重なる粒子がある場合は流入せず、その粒子の速度を流入速度に書き換える
                if dist_sqr_min < (psize * 0.99)**2:
                    if particles_type[j_min] == type_fluid:
                        particles_vel[j_min] = injectors_vel[i]
                    continue

                # 空きスロットを探す
                j_ghost = -1
                for j in range(N_particles):
                    if particles_type[j] == type_ghost:
                        type_j = ti.atomic_min(particles_type[j], type_fluid)
                        if type_j == type_ghost:
                            j_ghost = j
                            break

                # 空きスロットがあれば粒子を発生させる
                if j_ghost != -1:
                    particles_type[j_ghost] = type_fluid
                    particles_vel[j_ghost] = injectors_vel[i]
                    particles_pos[j_ghost] = pos_i
                    particles_fluid_id[j_ghost] = inject_fluid_id[None]
        else:
            pass

    # 粒子数密度と圧力
    for i in range(N_particles):
        particles_pnd[i] = 0.0
        particles_pres[i] = 0.0
        if particles_type[i] == type_ghost:
            continue

        # 近傍粒子探索ループ
        bx0 = int((particles_pos[i]-domain[0]).x/re)
        by0 = int((particles_pos[i]-domain[0]).y/re)
        for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx,by]):
                j = table_data[bx,by,l]
                if j == i:
                    continue
                pos_ij = particles_pos[j] - particles_pos[i]
                dist_ij = pos_ij.norm()
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < re:
                        particles_force[i] += fluid_viscosity*4.0/(pnd0[None]*lambda0[None])*(particles_vel[j]-particles_vel[i])*weight(dist_ij)

                # 粒子数密度
                if dist_ij < re:
                    particles_pnd[i] += weight(dist_ij)

        # 圧力
        if particles_pnd[i] > pnd0[None]:
            particles_pres[i] = (
                fluid_density[0] * fluid_sound**2 * (particles_pnd[i] - pnd0[None]) / pnd0[None]
            )
        else:
            particles_pres[i] = 0.0
    # 力の計算(2回目)
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall or particles_type[i] == type_ghost:
            continue

        # 近傍粒子探索ループ
        bx0 = int((particles_pos[i]-domain[0]).x/re)
        by0 = int((particles_pos[i]-domain[0]).y/re)
        for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx,by]):
                j = table_data[bx,by,l]
                if j == i:
                    continue
                pos_ij = particles_pos[j] - particles_pos[i]
                dist_ij = pos_ij.norm()
            
            # # 粘性項
            # if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
            #     if dist_ij < re:
            #         particles_force[i] += fluid_viscosity*4.0/(pnd0[None]*lambda0[None])*(particles_vel[j]-particles_vel[i])*weight(dist_ij)

                # 圧力項
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < re:
                        particles_force[i] -= (
                            2.0 / pnd0_gradP[None]
                            * (particles_pres[i] + particles_pres[j])
                            * pos_ij / dist_ij**2
                            * weight_gradP(dist_ij)
                        )
                # 衝突モデル
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < collision_dist:
                        vel_ij = particles_vel[j] - particles_vel[i]
                        normal_ij = pos_ij / dist_ij
                        tmp = normal_ij.dot(vel_ij)

                        if tmp < 0.0:
                            if (particles_type[i], particles_type[j]) == (type_fluid, type_fluid):
                                m_ij = (
                                    fluid_density[particles_fluid_id[i]]
                                    * fluid_density[particles_fluid_id[j]]
                                    / (fluid_density[particles_fluid_id[i]] + fluid_density[particles_fluid_id[j]])
                                )
                                particles_force[i] += normal_ij * (1.0 + collision_coef) * m_ij * tmp / dt_max

                            elif (particles_type[i], particles_type[j]) == (type_fluid, type_wall):
                                m_ij = fluid_density[particles_fluid_id[i]]
                                particles_force[i] += normal_ij * (1.0 + collision_coef) * m_ij * tmp / dt_max

                            elif (particles_type[i], particles_type[j]) == (type_fluid, type_rigid):
                                m_ij = (
                                    fluid_density[particles_fluid_id[i]]
                                    * rigids_density[particles_rigid_id[j]]
                                    / (fluid_density[particles_fluid_id[i]] + rigids_density[particles_rigid_id[j]])
                                )
                                particles_force[i] += normal_ij * (1.0 + collision_coef) * m_ij * tmp / dt_max

                            elif (particles_type[i], particles_type[j]) == (type_rigid, type_fluid):
                                m_ij = (
                                    fluid_density[particles_fluid_id[j]]
                                    * rigids_density[particles_rigid_id[i]]
                                    / (fluid_density[particles_fluid_id[j]] + rigids_density[particles_rigid_id[i]])
                                )
                                particles_force[i] += normal_ij * (1.0 + collision_coef) * m_ij * tmp / dt_max

                # 剛体粒子同士の反発力
                if (particles_type[i], particles_type[j]) == (type_rigid, type_rigid):
                    k_i = particles_rigid_id[i]
                    k_j = particles_rigid_id[j]
                    if k_i == k_j:
                        continue

                    len_ij = psize - dist_ij  # 重なりの長さ
                    if len_ij > 0.0:
                        normal_ij = pos_ij.normalized()
                        m_i = rigids_density[k_i]
                        m_j = rigids_density[k_j]
                        m_ij = 2 * m_i * m_j / (m_i + m_j)

                        particles_force[i] -= normal_ij * (m_ij * len_ij / dt_max**2)

                        vel_ij = particles_vel[j] - particles_vel[i]
                        tmp = normal_ij.dot(vel_ij)  # 法線方向の相対速度
                        if tmp < 0.0:
                            particles_force[i] += normal_ij * (m_ij * 10 * tmp / dt_max)

                        if True:  # 摩擦力を考慮する場合
                            tau_ij = ti.Vector([-normal_ij.y, normal_ij.x])  # 接線ベクトル
                            tmp2 = tau_ij.dot(vel_ij)  # 接線方向の相対速度
                            particles_force[i] += tau_ij * (m_ij * tmp2 / dt_max)
                            
                        # 壁面からの反発力
                        if (particles_type[i], particles_type[j]) == (type_rigid, type_wall):
                            len_ij = psize - dist_ij  # 重なりの長さ

                            if len_ij > 0.0:
                                normal_ij = pos_ij.normalized()
                                k = particles_rigid_id[i]
                                m_ij = 2 * rigids_density[k]

                                particles_force[i] -= normal_ij * (m_ij * len_ij / dt_max**2)

                                vel_ij = particles_vel[j] - particles_vel[i]
                                tmp = normal_ij.dot(vel_ij)  # 法線方向の相対速度
                                if tmp < 0.0:
                                    particles_force[i] += normal_ij * (m_ij * 10 * tmp / dt_max)

                                # 摩擦力を考慮する場合
                                if True:
                                    tau_ij = ti.Vector([-normal_ij.y, normal_ij.x])  # 接線ベクトル
                                    tmp2 = tau_ij.dot(vel_ij)  # 接線方向の相対速度
                                    particles_force[i] += tau_ij * (m_ij * tmp2 / dt_max)

    # 剛体に作用する力の和と力のモーメントの和の計算
    rigids_force.fill(0.0)
    rigids_moment.fill(0.0)

    for i in range(N_particles):
        if particles_type[i] == type_rigid:
            k = particles_rigid_id[i]
            force_i = particles_force[i] * psize**2

            rigids_force[k] += force_i
            rigids_moment[k] += (particles_pos[i] - rigids_rcenter[k]).cross(force_i)

    # 剛体の角速度と角度と重心速度と重心位置の修正
    for k in range(N_rigids):
        rigids_omega[k] += (rigids_moment[k] / rigids_inertia[k]) * dt[None]
        rigids_angle[k] += (rigids_moment[k] / rigids_inertia[k]) * dt[None]**2
        rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])

        if rigids_fixcenter[k]:
            rigids_pos[k] = rigids_rcenter[k] + rigids_rmatrix[k] @ (rigids_pos_ini[k] - rigids_rcenter[k])
            rigids_vel[k].x = -(rigids_pos[k] - rigids_rcenter[k]).y * rigids_omega[k]
            rigids_vel[k].y = (rigids_pos[k] - rigids_rcenter[k]).x * rigids_omega[k]
        else:
            rigids_vel[k] += (rigids_force[k] / rigids_mass[k]) * dt[None]
            rigids_pos[k] += (rigids_force[k] / rigids_mass[k]) * dt[None]**2
            rigids_rcenter[k] = rigids_pos[k]

    # 粒子の速度と位置の修正
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            k = particles_fluid_id[i]
            particles_vel[i] += (particles_force[i] / fluid_density[k]) * dt[None]
            particles_pos[i] += (particles_force[i] / fluid_density[k]) * dt[None]**2

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

    # 領域外粒子の処理
    for i in range(N_particles):
        if particles_type[i] != type_fluid:
            continue
        if particles_pos[i].x < domain[0].x or particles_pos[i].x >= domain[1].x or particles_pos[i].y < domain[0].y or particles_pos[i].y >= domain[1].y:
            particles_type[i] = type_ghost
            particles_pos[i] = (0.0,0.0)
            particles_vel[i] = (0.0,0.0)
        # 時間ステップを進める
        step[None] += 1
        time[None] += dt[None]


# 色更新カーネル
@ti.kernel
def update_colors():
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            if particles_fluid_id[i] == 0:
                entropy = particles_entropy[i]
                # 最大エントロピーは ln(2)（2種類の流体の場合）
                normalized_entropy = entropy / ti.log(2.0)
                normalized_entropy = ti.min(ti.max(normalized_entropy, 0.0), 1.0)
                r = ti.cast(normalized_entropy * 255, ti.i32)
                g = 0
                b = 255
                particles_color[i] = (r << 16) + (g << 8) + b
            else:
                entropy = particles_entropy[i]
                # 最大エントロピーは ln(2)（2種類の流体の場合）
                normalized_entropy = entropy / ti.log(2.0)
                normalized_entropy = ti.min(ti.max(normalized_entropy, 0.0), 1.0)
                r = ti.cast(normalized_entropy * 255, ti.i32)
                g = 255
                b = 0
                particles_color[i] = (r << 16) + (g << 8) + b
        elif particles_type[i] == type_wall:
            particles_color[i] = 0x808080  # 灰色
        elif particles_type[i] == type_rigid:
            particles_color[i] = 0xFF0000  # 赤色
        elif particles_type[i] == type_ghost:
            particles_color[i] = 0x000000  # 黒色



# 初期化する
initialize()
print('pnd0 =',pnd0[None])
print('lambda0 =',lambda0[None])

# GUI作成
window_size = (640,640)
window_title = os.path.basename(__file__)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(window_title,window_size)

# ウィジェット
slider_forwards = gui.slider('fast-forward',1,20)
slider_forwards.value = 10

##描画
while gui.running:
    # マウスの情報を取得する
    cursor_x, cursor_y = gui.get_cursor_pos()
    mouse_pos[None][0] = domain[0].x + (domain[1] - domain[0]).x * cursor_x
    mouse_pos[None][1] = domain[0].y + (domain[1] - domain[0]).y * cursor_y
    mouse_state[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        mouse_state[None] = 1
    forwards = int(slider_forwards.value)
    slider_forwards.value = forwards
    
    # キーイベントの処理
    for e in gui.get_events(gui.PRESS):
        if e.key == ti.GUI.ESCAPE:
            initialize()
        elif e.key.lower() == 'w':
            inject_fluid_id[None] = 0  # 水を選択
            print("Injecting Water")
        elif e.key.lower() == 'a':
            inject_fluid_id[None] = 1  # アルコールを選択
            print("Injecting Alcohol")
        elif e.key == 's':
            inject_rigid_id[None] = 1  # 剛体0を選択
            print("Injecting Rigid 1")


    # 時間を進める
    for frame in range(forwards):
        preupdate()
        for substep in range(substeps[None]):
            update()
        compute_entropy()           # エントロピー計算を追加
        compute_total_entropy()     # エントロピーの合計を計算
        update_colors()             # 色更新を追加


    # キーボード入力の受け取り処理
    for e in gui.get_events(gui.PRESS):
        # ESCキーが押されたら初期化する
        if e.key == ti.GUI.ESCAPE:
            initialize()
    # 現在の状態を描画する
   
    if mouse_state[None] == 1:
        if inject_rigid_id[None] == 0:
            J = injectors_pos.to_numpy()
            J[:, 0] = (J[:, 0] + mouse_pos[None].x - domain[0].x) / (domain[1] - domain[0]).x
            J[:, 1] = (J[:, 1] + mouse_pos[None].y - domain[0].y) / (domain[1] - domain[0]).y
            gui.circles(J, radius=psize * 0.5 * scale_to_pixel, color=0x00FF00)
    X = particles_pos.to_numpy()
    X[:, 0] = (X[:, 0] - domain[0].x) / (domain[1] - domain[0]).x
    X[:, 1] = (X[:, 1] - domain[0].y) / (domain[1] - domain[0]).y
    T = particles_type.to_numpy()
    C = particles_color.to_numpy()
    gui.circles(X[(T != type_ghost), :], radius=psize * 0.5 * scale_to_pixel, color=C[(T != type_ghost)])
    
    # 現在選択されている流体の種類を表示
    if inject_fluid_id[None] == 0:
        current_fluid = "Water"
    elif inject_fluid_id[None] == 1:
        current_fluid = "Alcohol"
    else:
        current_fluid = "Unknown"
    
    gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}, substeps = {substeps[None]}', (0.0, 1.0), font_size=20, color=0xFFFFFF)
    gui.text(f'Particles: {numpy.count_nonzero(T == type_fluid)} / {numpy.count_nonzero(T == type_wall)} / {numpy.count_nonzero(T == type_ghost)}', (0.0, 0.975), font_size=20, color=0xFFFFFF)
    gui.text(f'Entropy: {total_entropy[None]:.6f}', (0.0, 0.95), font_size=20, color=0xFFFFFF)
    gui.text(f'Current Inject Fluid: {current_fluid}', (0.0, 0.925), font_size=20, color=0xFFFFFF)  # 現在の流体の表示
    
    gui.show()
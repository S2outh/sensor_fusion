use core::f32;
use libm::{atan2, cos, fabsf, powf, sin, sincos, sqrt, sqrtf};
use nalgebra::{Matrix3, Matrix4, Quaternion, SMatrix, SVector, UnitQuaternion, Vector3};
use std::io::{self, Write};
#[derive(Debug, Default, Clone)]

pub struct FlightData {
    pub accel_x_1: Vec<f32>,
    pub accel_y_1: Vec<f32>,
    pub accel_z_1: Vec<f32>,
    pub accel_x_2: Vec<f32>,
    pub accel_y_2: Vec<f32>,
    pub accel_z_2: Vec<f32>,

    pub roll_1: Vec<f32>,
    pub pitch_1: Vec<f32>,
    pub yaw_1: Vec<f32>,

    pub roll_2: Vec<f32>,
    pub pitch_2: Vec<f32>,
    pub yaw_2: Vec<f32>,

    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub alt: Vec<f64>,

    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,

    pub pressure: Vec<f32>,
}

pub struct RocketEKF {
    pub state: SVector<f64, 17>,
    pub p: SMatrix<f64, 17, 17>,
    pub q: SMatrix<f64, 17, 17>,
    pub r: SMatrix<f64, 10, 10>,
    pub baro_needs_sync: bool,
}

pub struct FlightManager {
    pub rocket_started: bool,
    pub ascent_flag: bool,
    pub calibration_active: bool,
    pub calibration_start_time: f64,
    pub calibration_count: usize,
    pub block_gps: bool,
    pub valid_gps_lat: f64,
    pub valid_gps_lon: f64,
    pub valid_gps_alt: f64,
    // Ringpuffer für den Tiefpass
    pub accel_gyro_window: Vec<[f64; 6]>,
    pub altitude_window: Vec<f64>,
}

fn confirm() {
    println!("\n--- DATEN-CHECK ---");
    println!("press ENTER to continue");
    io::stdout().flush().unwrap();

    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .expect("Fehler beim Lesen");
}

pub fn pres_to_alt(pres: f32) -> f32 {
    //44_330.0 * (1.0 - powf(pres/101274.697, 1.0/5.255)) //Graz
    44_330.0 * (1.0 - powf(pres / 104_315.0, 1.0 / 5.255)) //HyEnD
}

pub fn quaternion_rotation_matrix(q: &[f32; 4]) -> [[f32; 3]; 3] {
    //Covert a quaternion into a full three-dimensional rotation matrix.

    //Input
    //param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    //Output
    //return: A 3x3 element matrix representing the full 3D rotation matrix.
    //This rotation matrix converts a point in the local reference
    //frame to a point in the global reference frame.

    //Extract the values from Q
    let q0 = q[0];
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];

    let q00 = q0 * q0;
    let q11 = q1 * q1;
    let q22 = q2 * q2;
    let q33 = q3 * q3;

    // First row of the rotation matrix
    let r00 = 2.0 * (q00 + q11) - 1.0;
    let r01 = 2.0 * (q1 * q2 - q0 * q3);
    let r02 = 2.0 * (q1 * q3 + q0 * q2);

    // Second row
    let r10 = 2.0 * (q1 * q2 + q0 * q3);
    let r11 = 2.0 * (q00 + q22) - 1.0;
    let r12 = 2.0 * (q2 * q3 - q0 * q1);

    // Third row
    let r20 = 2.0 * (q1 * q3 - q0 * q2);
    let r21 = 2.0 * (q2 * q3 + q0 * q1);
    let r22 = 2.0 * (q00 + q33) - 1.0;

    // 3x3 rotation matrix
    [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
}

pub fn compute_d_rotation_d_quaternion(q: &[f64; 4]) -> [Matrix3<f64>; 4] {
    // q[0]=w, q[1]=x, q[2]=y, q[3]=z (Annahme: Hamilton-Notation)
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    // dR/dw
    let dr_dw = Matrix3::new(
        0.0,
        -4.0 * z,
        4.0 * y,
        4.0 * z,
        0.0,
        -4.0 * x,
        -4.0 * y,
        4.0 * x,
        0.0,
    );

    // dR/dx
    let dr_dx = Matrix3::new(
        0.0,
        4.0 * y,
        4.0 * z,
        4.0 * y,
        -8.0 * x,
        -4.0 * w,
        4.0 * z,
        4.0 * w,
        -8.0 * x,
    );

    // dR/dy
    let dr_dy = Matrix3::new(
        -8.0 * y,
        4.0 * x,
        4.0 * w,
        4.0 * x,
        0.0,
        4.0 * z,
        -4.0 * w,
        4.0 * z,
        -8.0 * y,
    );

    // dR/dz
    let dr_dz = Matrix3::new(
        -8.0 * z,
        -4.0 * w,
        4.0 * x,
        4.0 * w,
        -8.0 * z,
        4.0 * y,
        4.0 * x,
        4.0 * y,
        0.0,
    );

    [dr_dw, dr_dx, dr_dy, dr_dz]
}
pub fn latlonh_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    // WGS84 consts
    let a: f64 = 6_378_137.0;
    let e_sq: f64 = 0.00669437999014;

    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let (sin_lat, cos_lat) = sincos(lat);
    let (sin_lon, cos_lon) = sincos(lon);

    let n = sqrt(a / (1.0 - e_sq * sin_lat * sin_lat));

    let x = (n + h_m) * cos_lat * cos_lon;
    let y = (n + h_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - e_sq) + h_m) * sin_lat;

    [x, y, z]
}

pub fn ecef_to_ned_matrix(lat_deg: f64, lon_deg: f64) -> Matrix3<f64> {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let s_lat = sin(lat);
    let c_lat = cos(lat);
    let s_lon = sin(lon);
    let c_lon = cos(lon);

    // Standard Rotationsmatrix ECEF -> NED
    Matrix3::new(
        -s_lat * c_lon,
        -s_lat * s_lon,
        c_lat,
        -s_lon,
        c_lon,
        0.0,
        -c_lat * c_lon,
        -c_lat * s_lon,
        -s_lat,
    )
}

pub fn quaternion_from_vectors(v1_in: [f32; 3], v2_in: [f32; 3]) -> [f32; 4] {
    let eps = 1e-8_f32;

    let norm = |v: [f32; 3]| sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    let n1 = norm(v1_in);
    let n2 = norm(v2_in);

    let v1 = [v1_in[0] / n1, v1_in[1] / n1, v1_in[2] / n1];
    let v2 = [v2_in[0] / n2, v2_in[1] / n2, v2_in[2] / n2];

    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];

    if dot > -1.0 + eps {
        let c = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ];

        let q = [1.0 + dot, c[0], c[1], c[2]];

        let q_norm = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

        return [q[0] / q_norm, q[1] / q_norm, q[2] / q_norm, q[3] / q_norm];
    }

    let mut aux = [0.0, 0.0, 0.0];
    if fabsf(v1[0]) < fabsf(v1[1]) {
        if fabsf(v1[0]) < fabsf(v1[2]) {
            aux[0] = 1.0;
        } else {
            aux[2] = 1.0;
        }
    } else {
        if fabsf(v1[1]) < fabsf(v1[2]) {
            aux[1] = 1.0;
        } else {
            aux[2] = 1.0;
        }
    }

    let axis = [
        v1[1] * aux[2] - v1[2] * aux[1],
        v1[2] * aux[0] - v1[0] * aux[2],
        v1[0] * aux[1] - v1[1] * aux[0],
    ];

    let a_norm = norm(axis);
    [0.0, axis[0] / a_norm, axis[1] / a_norm, axis[2] / a_norm]
}

pub fn gps_to_ned(
    lat: f64,
    lon: f64,
    h: f64,
    lat_ref: f64,
    lon_ref: f64,
    alt_ref: f64,
) -> [f64; 3] {
    let ecef_ref_arr = latlonh_to_ecef(lat_ref, lon_ref, alt_ref);
    let ecef_ref = Vector3::from(ecef_ref_arr);

    let r: Matrix3<f64> = ecef_to_ned_matrix(lat_ref, lon_ref);

    let ecef_current_arr = latlonh_to_ecef(lat, lon, h);
    let ecef_current = Vector3::from(ecef_current_arr);

    let delta_ecef = ecef_current - ecef_ref;

    let ned = r * delta_ecef;

    [ned.x, ned.y, ned.z]
}

pub fn ned_to_gps(ned_arr: [f64; 3], lat_ref: f64, lon_ref: f64, alt_ref: f64) -> (f64, f64, f64) {
    // 1. Inputs in nalgebra-Typen wandeln
    let ned = Vector3::from(ned_arr);
    let ecef_ref = Vector3::from(latlonh_to_ecef(lat_ref, lon_ref, alt_ref));

    // 2. Rotationsmatrix holen (ECEF -> NED)
    let r: Matrix3<f64> = ecef_to_ned_matrix(lat_ref, lon_ref);

    // 3. Rücktransformation: delta_ecef = R^T * ned
    // Da R orthogonal ist, ist die Inverse gleich der Transponierten.
    let delta_ecef = r.transpose() * ned;

    // 4. ECEF Position berechnen
    let ecef_current = ecef_ref + delta_ecef;

    // 5. ECEF zu Geodätisch (WGS84 Konstanten)
    let a = 6378137.0;
    let b = 6356752.314245;
    let e_sq = 0.00669437999014;
    let ep_sq = 0.00673949674228;

    let x = ecef_current.x;
    let y = ecef_current.y;
    let z = ecef_current.z;

    // Bowring's Algorithmus
    let p = sqrt(x * x + y * y);
    let th = atan2(a * z, b * p);

    let sin_th = sin(th);
    let cos_th = cos(th);

    let lat_rad = atan2(
        z + ep_sq * b * sin_th * sin_th * sin_th,
        p - e_sq * a * cos_th * cos_th * cos_th,
    );
    let lon_rad = atan2(y, x);

    let sin_lat = sin(lat_rad);
    let n = a / sqrt(1.0 - e_sq * sin_lat * sin_lat);
    let alt = (p / cos(lat_rad)) - n;

    (lat_rad.to_degrees(), lon_rad.to_degrees(), alt)
}

pub fn get_reference_coordinates_new(data: &FlightData) -> (f64, f64, f64) {
    for i in 0..data.lat.len() {
        if data.lat[i].abs() > 0.1 {
            return (data.lat[i], data.lon[i], data.alt[i]);
        }
    }
    (0.0, 0.0, 0.0)
}

pub fn normalize_vector(vector: [f32; 3]) -> [f32; 3] {
    let norm_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];

    if norm_sq > 0.0 {
        let norm = sqrtf(norm_sq);
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    } else {
        vector
    }
}

pub fn normalize_quaternion(quaternion: [f64; 4]) -> [f64; 4] {
    let norm = sqrt(
        quaternion[0] * quaternion[0]
            + quaternion[1] * quaternion[1]
            + quaternion[2] * quaternion[2]
            + quaternion[3] * quaternion[3],
    );

    [
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm,
        quaternion[3] / norm,
    ]
}

pub fn state_transition(state: &SVector<f64, 17>, dt: f64,  mean_measurement: &[f64; 6], ref_gps: &[f64; 3]) -> SVector<f64, 17> {
    //println!("ich bin state_transition - new iteration");
    let mut next_state = *state;
    let accel = &mean_measurement[0..3];
    let gyro  = &mean_measurement[3..6];

    // Rotation matrix
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        state[6], state[7], state[8], state[9], // w, x, y, z
    ));
    println!("Höhe {}", -state[2]);
    let r_body_to_ned = q.to_rotation_matrix();

    // Gravitation matrix on body
    let g_ned = Vector3::new(0.0, 0.0, 9.8);
    let g_body = r_body_to_ned.transpose() * g_ned;

    // acceleration withouth gravitation
    let a_body = Vector3::new(
        accel[0] - state[10],
        accel[1] - state[11],
        accel[2] - state[12],
    ) + g_body;
    println!("accel {}", a_body[2]);
    let mut a_ned = r_body_to_ned * a_body;

    // Deadzone
    if a_ned.norm() < 1e-3 {
        a_ned = Vector3::zeros();
    }

    // speed update
    next_state[3] += a_ned.x * dt;
    next_state[4] += a_ned.y * dt;
    next_state[5] += a_ned.z * dt;
    let lat_ref = ref_gps[0];
    let lon_ref=ref_gps[1];
    let alt_ref=ref_gps[2];
    let ned_current = gps_to_ned(state[0], state[1], state[2], lat_ref, lon_ref, alt_ref);
    let n = ned_current[0] + (state[3] + next_state[3]) * 0.5 * dt;
    let e = ned_current[1] + (state[4] + next_state[4]) * 0.5 * dt;
    let d = ned_current[2] + (state[5] + next_state[5]) * 0.5 * dt;
    let (lat, lon, alt) = ned_to_gps([n, e, d], lat_ref, lon_ref, alt_ref);
    next_state[0] = lat;
    next_state[1] = lon;
    next_state[2] = alt;
    // Position (NED) = pos + (vel_alt + vel_neu)/2 * dt
    //next_state[0] += (state[3] + next_state[3]) * 0.5 * dt; // North
    //next_state[1] += (state[4] + next_state[4]) * 0.5 * dt; // East
    //next_state[2] += (state[5] + next_state[5]) * 0.5 * dt; // Down
    println!("a_ned: {}", a_ned);
    println!("v: {} {} {}", next_state[3], next_state[4], next_state[5]);
    println!("ned_current: {:?}", ned_current);
    println!("n e d: {} {} {}", n, e, d);
    println!("lat lon alt: {} {} {}", lat, lon, alt);
    // Attitude update (Quaternion Integration)
    let gx = (gyro[0] - state[13]) * dt;
    let gy = (gyro[1] - state[14]) * dt;
    let gz = (gyro[2] - state[15]) * dt;

    let q_alt = Matrix4::new(
        1.0,
        -gx / 2.0,
        -gy / 2.0,
        -gz / 2.0,
        gx / 2.0,
        1.0,
        gz / 2.0,
        -gy / 2.0,
        gy / 2.0,
        -gz / 2.0,
        1.0,
        gx / 2.0,
        gz / 2.0,
        gy / 2.0,
        -gx / 2.0,
        1.0,
    );

    let current_q = SVector::<f64, 4>::new(state[6], state[7], state[8], state[9]);
    //println!("current_q: {:.3}", {current_q});
    let next_q = (q_alt * current_q).normalize();

    next_state[6] = next_q[0];
    next_state[7] = next_q[1];
    next_state[8] = next_q[2];
    next_state[9] = next_q[3];

    next_state
}

pub fn state_transition_jacobian(state: &SVector<f64, 17>, dt: f64, mean_measurement: &[f64; 6]) -> SMatrix<f64, 17, 17> {
    let mut f = SMatrix::<f64, 17, 17>::identity(); // Diagonaleinträge = 1

    // Quaternion of state
    //let q_vec = SVector::<f64, 4>::new(state[6], state[7], state[8], state[9]);
    let q_x = state[7];
    let q_y = state[8];
    let q_z = state[9];
    let q_w = state[6];

    let q_vec = SVector::<f64, 4>::new(q_x, q_y, q_z, q_w);
    let q = UnitQuaternion::from_quaternion(Quaternion::from_vector(q_vec));
    let r = q.to_rotation_matrix();

    // Python: F[3:6, 6:9] = R * dt
    // Python: F[3:6, 16:19] = -R * dt
    let rad_to_deg = 180.0 / std::f64::consts::PI;
    let r_earth = 6378137.0;

    f[(0, 3)] = dt / r_earth * rad_to_deg;
    f[(1, 4)] = dt / (r_earth * state[0].to_radians().cos()) * rad_to_deg;
    f[(2, 5)] = -dt;
    for row in 0..3 {
        for col in 0..3 {
            let val = r[(row, col)] * dt;
            f[(3 + row, 10 + col)] = -val; //Bias
        }
    }

    // Gyro + Bias
    let gx = mean_measurement[3] - state[13];
    let gy = mean_measurement[4] - state[14];
    let gz = mean_measurement[5] - state[15];

    // Partial differentiation
    f[(6, 7)] = -0.5 * gx * dt;
    f[(6, 8)] = -0.5 * gy * dt;
    f[(6, 9)] = -0.5 * gz * dt;
    f[(7, 6)] = 0.5 * gx * dt;
    f[(7, 8)] = 0.5 * gz * dt;
    f[(7, 9)] = -0.5 * gy * dt;
    f[(8, 6)] = 0.5 * gy * dt;
    f[(8, 7)] = -0.5 * gz * dt;
    f[(8, 9)] = 0.5 * gx * dt;
    f[(9, 6)] = 0.5 * gz * dt;
    f[(9, 7)] = 0.5 * gy * dt;
    f[(9, 8)] = -0.5 * gx * dt;

    let q_vals = [state[6], state[7], state[8], state[9]]; // x, y, z, w

    // Zeile 12: [-0.5*q1, -0.5*q2, -0.5*q3]
    // Zeile 13: [ 0.5*q0, -0.5*q3,  0.5*q2]
    // Zeile 14: [ 0.5*q3,  0.5*q0, -0.5*q1]
    // Zeile 15: [-0.5*q2,  0.5*q1,  0.5*q0]
    /*
    let derivs = [
        [-0.5 * q_vals[1], -0.5 * q_vals[2], -0.5 * q_vals[3]],
        [0.5 * q_vals[0], -0.5 * q_vals[3], 0.5 * q_vals[2]],
        [0.5 * q_vals[3], 0.5 * q_vals[0], -0.5 * q_vals[1]],
        [-0.5 * q_vals[2], 0.5 * q_vals[1], 0.5 * q_vals[0]],
    ];

    for i in 0..4 {
        for j in 0..3 {
            let val = derivs[i][j] * dt;
            f[(6 + i, 13 + j)] = val; // Bias
        }
    }*/
    f
}

// h(x) simulation, if my predicted state is perfect, which numbers would show my sensors
pub fn measurement_function(
    state: &SVector<f64, 17>,
    calibration_active: bool,
) -> SVector<f64, 10> {
    // Expected Baro measurment (hight-baro offset)
    // NEUE umgedrehte vorzeichen
    let baro_expected = state[2] - state[16];

    // Measurment vector
    SVector::<f64, 10>::from_column_slice(&[
        state[0],         // gps_lat (bzw. North m)
        state[1],         // gps_lon (bzw. East m)
        state[2],         // gps_alt (bzw. Down m)
        state[3],      // low_g_ax
        state[4],      // low_g_ay
        state[5],      // low_g_az
        state[6],  // low_g_gx
        state[7], // low_g_gy
        state[8],   // low_g_gz
        baro_expected,    // baro_alt
    ])
}

// H, if a number change in my state, how infect this my theoretical measurments
pub fn measurement_jacobian(state: &SVector<f64, 17>) -> SMatrix<f64, 10, 17> {
    let mut h = SMatrix::<f64, 10, 17>::zeros();

    // GPS Position
    h[(0, 0)] = 1.0; // d(gps_n) / d(north)
    h[(1, 1)] = 1.0; // d(gps_e) / d(east)
    h[(2, 2)] = 1.0; // d(gps_d) / d(down)

    // Baro
    h[(9, 2)] = 1.0; // d(baro_alt) / d(down_pos)
    h[(9, 16)] = -1.0; // d(baro_alt) / d(baro_bias)

    // Accel-Bias
    h[(3, 10)] = -1.0;
    h[(4, 11)] = -1.0;
    h[(5, 12)] = -1.0;

    // Gyro bias
    h[(6, 13)] = -1.0;
    h[(7, 14)] = -1.0;
    h[(8, 15)] = -1.0;

    let accel_norm = (state.fixed_rows::<3>(6)).norm();
    if (accel_norm - 9.81).abs() < 1e-2 {
        let q = [state[6], state[7], state[8], state[9]]; // x, y, z, w
        let d_r_dq = compute_d_rotation_d_quaternion(&q);
        let g_ned = SVector::<f64, 3>::new(0.0, 0.0, 9.8);

        for j in 0..4 {
            let dr_dq_j = d_r_dq[j];
            let dg_body_dqj = dr_dq_j.transpose() * g_ned;

            for i in 0..3 {
                h[(3 + i, 6 + j)] = -dg_body_dqj[i];
            }
        }
    }

    h
}

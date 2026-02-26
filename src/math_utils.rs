use core::f32;
use nalgebra::{Matrix3, Matrix4, Quaternion, SMatrix, SVector, UnitQuaternion, Vector3};
use libm::{powf, sincosf, sqrtf, fabsf, cos, sin, atan2f, sqrt, sincos, atan2, cosf, sinf};

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

pub struct RocketEKF{
    pub state: SVector<f64, 23>,
    pub p: SMatrix<f64, 23, 23>,
    pub q: SMatrix<f64, 23, 23>,
    pub r: SMatrix<f64, 10, 10>,
    pub baro_needs_sync: bool,
}


pub fn pres_to_alt(pres: f32) -> f32{
    //44_330.0 * (1.0 - powf(pres/101274.697, 1.0/5.255)) //Graz
    44_330.0 * (1.0 - powf(pres/104_315.0, 1.0/5.255)) //HyEnD
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
    [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22],
    ]
}

pub fn compute_d_rotation_d_quaternion(q: &[f32; 4]) -> [[[f32; 3]; 4]; 3] {

    // Compute derivative of rotation matrix w.r.t. quaternion components
    // Returns tensor of shape (4, 3, 3) where:
        // result[j] = dR/dq_j for j in (0,1,2,3)

    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    let mut dr_dq = [[[0.0f32; 3]; 3]; 4];

    // dR/dw
    dr_dq[0] = [
        [0.0, -2.0 * z, 2.0 * y],
        [2.0 * z, 0.0, -2.0 * x],
        [-2.0 * y, 2.0 * x, 0.0],
    ];

    // dR/dx
    dr_dq[1] = [
        [2.0 * x, 2.0 * y, 2.0 * z],
        [2.0 * y, -2.0 * x, -2.0 * w],
        [2.0 * z, 2.0 * w, -2.0 * x],
    ];

    // dR/dy
    dr_dq[2] = [
        [-2.0 * y, 2.0 * x, 2.0 * w],
        [2.0 * x, 2.0 * y, 2.0 * z],
        [-2.0 * w, 2.0 * z, -2.0 * y],
    ];

    // dR/dz
    dr_dq[3] = [
        [-2.0 * z, -2.0 * w, 2.0 * x],
        [2.0 * w, -2.0 * z, 2.0 * y],
        [2.0 * x, 2.0 * y, 2.0 * z],
    ];

    let mut result = [[[0.0f32; 3]; 4]; 3];
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..3 {
                result[i][j][k] = dr_dq[j][i][k];
            }
        }
    }

    result
}


pub fn latlonh_to_ecef_old(lat_deg: f32, lon_deg: f32, h_m: f32) -> [f32; 3] {
    // WGS84 consts
    let a: f32 = 6_378_137.0;
    let e_sq: f32 = 0.00669437999014;

    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let (sin_lat, cos_lat) = sincosf(lat);
    let (sin_lon, cos_lon) = sincosf(lon);

    let n = sqrtf( a / (1.0 - e_sq * sin_lat * sin_lat));

    let x = (n + h_m) * cos_lat * cos_lon;
    let y = (n + h_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - e_sq) + h_m) * sin_lat;

    [x, y, z]
}

pub fn latlonh_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    // WGS84 consts
    let a: f64 = 6_378_137.0;
    let e_sq: f64 = 0.00669437999014;

    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    let (sin_lat, cos_lat) = sincos(lat);
    let (sin_lon, cos_lon) = sincos(lon);

    let n = sqrt( a / (1.0 - e_sq * sin_lat * sin_lat));

    let x = (n + h_m) * cos_lat * cos_lon;
    let y = (n + h_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - e_sq) + h_m) * sin_lat;

    [x, y, z]
}

pub fn ecef_to_ned_matrix_old(lat_ref_deg: f32, lon_ref_deg: f32) -> [[f32; 3]; 3] {
    let lat = lat_ref_deg.to_radians();
    let lon = lon_ref_deg.to_radians();
    
    let (sin_lat, cos_lat) = sincosf(lat);
    let (sin_lon, cos_lon) = sincosf(lon);
    
    [
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [-sin_lon,            cos_lon,            0.0    ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ]
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
        -s_lat * c_lon, -s_lat * s_lon,  c_lat,
        -s_lon,          c_lon,          0.0,
        -c_lat * c_lon, -c_lat * s_lon, -s_lat,
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

pub fn gps_to_ned_old(lat: f32, lon: f32, h: f32, lat_ref: f32, lon_ref: f32, alt_ref: f32) -> [f32; 3] {
    let ecef_ref = latlonh_to_ecef_old(lat_ref, lon_ref, alt_ref);
    let r = ecef_to_ned_matrix_old(lat_ref, lon_ref);
    let ecef_current = latlonh_to_ecef_old(lat, lon, h);
    
    let delta_ecef = [
        ecef_current[0] - ecef_ref[0],
        ecef_current[1] - ecef_ref[1],
        ecef_current[2] - ecef_ref[2],
    ];

    [
        r[0][0] * delta_ecef[0] + r[0][1] * delta_ecef[1] + r[0][2] * delta_ecef[2],
        r[1][0] * delta_ecef[0] + r[1][1] * delta_ecef[1] + r[1][2] * delta_ecef[2],
        r[2][0] * delta_ecef[0] + r[2][1] * delta_ecef[1] + r[2][2] * delta_ecef[2],
    ]
}

pub fn gps_to_ned(lat: f64, lon: f64, h: f64, lat_ref: f64, lon_ref: f64, alt_ref: f64) -> [f64; 3] {
    let ecef_ref_arr = latlonh_to_ecef(lat_ref, lon_ref, alt_ref);
    let ecef_ref = Vector3::from(ecef_ref_arr);

    let r: Matrix3<f64> = ecef_to_ned_matrix(lat_ref, lon_ref);

    let ecef_current_arr = latlonh_to_ecef(lat, lon, h);
    let ecef_current = Vector3::from(ecef_current_arr);
    
    let delta_ecef = ecef_current - ecef_ref;

    let ned = r * delta_ecef;

    [ned.x, ned.y, ned.z]
}

pub fn ned_to_gps_old(ned: [f32; 3], lat_ref: f32, lon_ref: f32, alt_ref: f32) -> (f32, f32, f32) {

    // Transform NED coordinates back to geodetic coordinates
    // ned: north, east, down relative to reference
    // Returns lat_deg, long_deg, alt_m
    let r = ecef_to_ned_matrix_old(lat_ref, lon_ref);
    
    let delta_ecef = [
        r[0][0] * ned[0] + r[1][0] * ned[1] + r[2][0] * ned[2],
        r[0][1] * ned[0] + r[1][1] * ned[1] + r[2][1] * ned[2],
        r[0][2] * ned[0] + r[1][2] * ned[1] + r[2][2] * ned[2],
    ];
    
    // ecef_ref = latlonh_to_ecef(lat_ref, lon_ref, alt_ref)
    let ecef_ref = latlonh_to_ecef_old(lat_ref, lon_ref, alt_ref);
    
    // ecef_current = ecef_ref + delta_ecef
    let ecef_current = [
        ecef_ref[0] + delta_ecef[0],
        ecef_ref[1] + delta_ecef[1],
        ecef_ref[2] + delta_ecef[2],
    ];

    // ECEF to Geodetic (WGS84)
    let a = 6378137.0;
    let b = 6356752.314245;
    let e_sq = 0.00669437999014;
    let ep_sq = 0.00673949674228;

    let x = ecef_current[0];
    let y = ecef_current[1];
    let z = ecef_current[2];

    let p = sqrtf(x * x + y * y);
    let th = atan2f(a * z, b * p);

    let sin_th = sinf(th);
    let cos_th = cosf(th);

    let lat_rad = atan2f(
        z + ep_sq * b * sin_th * sin_th * sin_th,
        p - e_sq * a * cos_th * cos_th * cos_th,
    );
    let lon_rad = atan2f(y, x);

    let sin_lat = sinf(lat_rad);
    let n = a / sqrtf(1.0 - e_sq * sin_lat * sin_lat);
    let alt = (p / cosf(lat_rad)) - n;

    (
        lat_rad * (180.0 / core::f32::consts::PI),
        lon_rad * (180.0 / core::f32::consts::PI),
        alt,
    )
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

    (
        lat_rad.to_degrees(),
        lon_rad.to_degrees(),
        alt,
    )
}
// pub fn build_dataframe(mut data: FlightData) -> Option<FlightData> {
//     if data.alt == 0.0 {
//         return None;
//     }
// 
//     data.accel_x_1 /= 100.0;
//     data.accel_y_1 /= 100.0;
//     data.accel_z_1 /= 100.0;
//     data.accel_x_2 /= 100.0;
//     data.accel_y_2 /= 100.0;
//     data.accel_z_2 /= 100.0;
// 
//     data.gyro_pitch = -data.gyro_pitch;
//     data.gyro_pitch_2 = -data.gyro_pitch_2;
// 
//     data.accel_y = -data.accel_y;
//     data.accel_y_2 = -data.accel_y_2;
// 
//     if data.pressure < 0.0 {
//         // Hier müsste im echten System der ffill-Wert eingesetzt werden
//         return None; 
//     }
// 
//     // data['pressure'] = data['pressure'].apply(pres_to_alt)
//     data.pressure = pres_to_alt(data.pressure);
// 
//     Some(data)
// }


pub fn get_reference_coordinates_new(data: &FlightData)-> (f64, f64, f64){
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
        quaternion[0] * quaternion[0] + 
        quaternion[1] * quaternion[1] + 
        quaternion[2] * quaternion[2] + 
        quaternion[3] * quaternion[3]
    );
    
    [
        quaternion[0] / norm,
        quaternion[1] / norm,
        quaternion[2] / norm,
        quaternion[3] / norm,
    ]
}


pub fn state_transition(state: &SVector<f64, 23>, dt: f64) -> SVector<f64, 23> {
    let mut next_state = *state;

    // Position (NED) = pos + vel * dt
    next_state[0] += state[3] * dt; // North
    next_state[1] += state[4] * dt; // East
    next_state[2] += state[5] * dt; // Down

    // Rotation matrix
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        state[15], state[12], state[13], state[14] // w, x, y, z
    ));
    let r_body_to_ned = q.to_rotation_matrix();

    // Gravitation matrix on body
    let g_ned = Vector3::new(0.0, 0.0, 9.8);
    let g_body = r_body_to_ned.transpose() * g_ned;

    // acceleration withouth gravitation
    let a_body = Vector3::new(
        state[6] + state[16],
        state[7] + state[17],
        state[8] + state[18]
    ) + g_body;
    let mut a_ned = r_body_to_ned * a_body;
    
    // Deadzone
    if a_ned.norm() < 1e-3 { a_ned = Vector3::zeros(); }

    // acceleration update
    next_state[3] += a_ned.x * dt;
    next_state[4] += a_ned.y * dt;
    next_state[5] += a_ned.z * dt;

    // Attitude update (Quaternion Integration)
    let gx = (state[9] + state[19]) * dt;
    let gy = (state[10] + state[20]) * dt;
    let gz = (state[11] + state[21]) * dt;

    let q_alt = Matrix4::new(
        1.0,      -gx/2.0,  -gy/2.0,  -gz/2.0,
        gx/2.0,   1.0,       gz/2.0,  -gy/2.0,
        gy/2.0,  -gz/2.0,    1.0,      gx/2.0,
        gz/2.0,   gy/2.0,   -gx/2.0,   1.0
    );

    let current_q = SVector::<f64, 4>::new(state[12], state[13], state[14], state[15]);
    let next_q = (q_alt * current_q).normalize();

    next_state[12] = next_q[0];
    next_state[13] = next_q[1];
    next_state[14] = next_q[2];
    next_state[15] = next_q[3];

    next_state
}

pub fn state_transition_jacobian(state: &SVector<f64, 23>, dt: f64) -> SMatrix<f64, 23, 23> {
    let mut f = SMatrix::<f64, 23, 23>::identity(); // Diagonaleinträge = 1
    
    // Quaternion of state
    let q_vec = SVector::<f64, 4>::new(state[12], state[13], state[14], state[15]);
    let q = UnitQuaternion::from_quaternion(Quaternion::from_vector(q_vec));
    let r = q.to_rotation_matrix();

    // Python: F[3:6, 6:9] = R * dt
    // Python: F[3:6, 16:19] = -R * dt
    for row in 0..3 {
        for col in 0..3 {
            let val = r[(row, col)] * dt;
            f[(3 + row, 6 + col)] = val;
            f[(3 + row, 16 + col)] = -val; //Bias
        }
    }

    // Gyro + Bias
    let gx = state[9] + state[19];
    let gy = state[10] + state[20];
    let gz = state[11] + state[21];

    // Partial differentiation
    f[(12, 13)] = -0.5 * gx * dt; f[(12, 14)] = -0.5 * gy * dt; f[(12, 15)] = -0.5 * gz * dt;
    f[(13, 12)] =  0.5 * gx * dt; f[(13, 14)] =  0.5 * gz * dt; f[(13, 15)] = -0.5 * gy * dt;
    f[(14, 12)] =  0.5 * gy * dt; f[(14, 13)] = -0.5 * gz * dt; f[(14, 15)] =  0.5 * gx * dt;
    f[(15, 12)] =  0.5 * gz * dt; f[(15, 13)] =  0.5 * gy * dt; f[(15, 14)] = -0.5 * gx * dt;

    let q_vals = [state[12], state[13], state[14], state[15]]; // x, y, z, w
    
    // Zeile 12: [-0.5*q1, -0.5*q2, -0.5*q3]
    // Zeile 13: [ 0.5*q0, -0.5*q3,  0.5*q2]
    // Zeile 14: [ 0.5*q3,  0.5*q0, -0.5*q1]
    // Zeile 15: [-0.5*q2,  0.5*q1,  0.5*q0]
    let derivs = [
        [-0.5 * q_vals[1], -0.5 * q_vals[2], -0.5 * q_vals[3]],
        [ 0.5 * q_vals[0], -0.5 * q_vals[3],  0.5 * q_vals[2]],
        [ 0.5 * q_vals[3],  0.5 * q_vals[0], -0.5 * q_vals[1]],
        [-0.5 * q_vals[2],  0.5 * q_vals[1],  0.5 * q_vals[0]],
    ];

    for i in 0..4 {
        for j in 0..3 {
            let val = derivs[i][j] * dt;
            f[(12 + i, 19 + j)] = val; // Bias
            f[(12 + i, 9 + j)] = val;  // Gyro
        }
    }

    f
}

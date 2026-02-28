use std::{fs::File, time::Instant};
use std::error::Error;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, UnitQuaternion, Vector3};
use crate::math_utils::{self, ecef_to_ned_matrix, get_reference_coordinates_new, latlonh_to_ecef, measurement_function, measurement_jacobian, normalize_quaternion, state_transition, state_transition_jacobian, FlightManager, RocketEKF, FlightData};

const CALIBRAION_DURATION: f64 = 0.5;

fn open_csv(path: &str, limit:usize, step: usize, row: usize) -> Result<(Vec<f64>, Vec<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();
    let mut times = Vec::new();

    //Limit for testen, only every 10th data point
    for (i, result) in rdr.records().enumerate()  {
        if i >= limit { break; }
        if i % step == 0 {
            let record = result?;
            if let (Some(t_str), Some(v_str)) = (record.get(0), record.get(row)) {
                let t: f64 = t_str.trim().parse()?;
                let v: f32 = v_str.trim().parse()?;
                times.push(t);
                values.push(v);        
            }
        }
    }
    Ok((times, values))
}

fn open_csv_64(path: &str, limit:usize, step: usize, row: usize) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();
    let mut times = Vec::new();

    //Limit for testen, only every 10th data point
    for (i, result) in rdr.records().enumerate()  {
        if i >= limit { break; }
        if i % step == 0 {
            let record = result?;
            if let (Some(t_str), Some(v_str)) = (record.get(0), record.get(row)) {
                let t: f64 = t_str.trim().parse()?;
                let v: f64 = v_str.trim().parse()?;
                times.push(t);
                values.push(v);        
            }
        }
    }
    Ok((times, values))
}


fn interpolate(target_t: f64, times: &[f64], values: &[f64], last_idx: &mut usize) -> f64 {
    if times.is_empty() { return 0.0; }
    
    while *last_idx < times.len() - 1 && times[*last_idx + 1] < target_t {
        *last_idx += 1;
    }

    if *last_idx >= times.len() - 1 { return values[values.len() - 1]; }

    let t0 = times[*last_idx];
    let t1 = times[*last_idx + 1];
    let v0 = values[*last_idx];
    let v1 = values[*last_idx + 1];

    v0 + (v1 - v0) * (target_t - t0) / (t1 - t0)
}


pub fn load_all_data() -> Result<(FlightData, Vec<f64>), Box<dyn Error>> {
    let base_path = "./src/data_set_1/";
    let limit = 280000;
    let step = 10;

    let (raw_times, accel_x_1_raw) = open_csv(&format!("{}FSMS_ACC_Z_1.csv", base_path), limit, step, 1)?;
    let t_start = raw_times[0];
    let timestamps: Vec<f64> = raw_times.iter().map(|t| (t - t_start) / 1000.0).collect();
    
    let sync_f32 = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        let (s_times, s_vals) = open_csv(&format!("{}{}", base_path, name), limit, step, 1)?;
        let mut last_idx = 0;
        let s_vals_f64: Vec<f64> = s_vals.iter().map(|&v| v as f64).collect();
        Ok(raw_times.iter().map(|&t| interpolate(t, &s_times, &s_vals_f64, &mut last_idx) as f32).collect())
    };

    let sync_f64 = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let (s_times, s_vals) = open_csv_64(&format!("{}{}", base_path, name), limit, step, 1)?;
        let mut last_idx = 0;
        Ok(raw_times.iter().map(|&t| interpolate(t, &s_times, &s_vals, &mut last_idx)).collect())
    };

    let data = FlightData {
        accel_x_1: accel_x_1_raw,
        accel_y_1: sync_f32("FSMS_ACC_Y_1.csv")?,
        accel_z_1: sync_f32("FSMS_ACC_X_1.csv")?,

        accel_x_2: sync_f32("FSMS_ACC_Z_2.csv")?,
        accel_y_2: sync_f32("FSMS_ACC_Y_2.csv")?,
        accel_z_2: sync_f32("FSMS_ACC_X_2.csv")?,

        roll_1: sync_f32("FSMS_GYRO_Z_1.csv")?,
        pitch_1: sync_f32("FSMS_GYRO_Y_1.csv")?,
        yaw_1: sync_f32("FSMS_GYRO_X_1.csv")?,

        roll_2: sync_f32("FSMS_GYRO_Z_2.csv")?,
        pitch_2: sync_f32("FSMS_GYRO_Y_2.csv")?,
        yaw_2: sync_f32("FSMS_GYRO_X_2.csv")?,

        lat: sync_f64("FSMS_PX_LAT.csv")?,
        lon: sync_f64("FSMS_PX_LONG.csv")?,
        alt: sync_f64("FSMS_PX_HEIGHT.csv")?,

        x: sync_f64("FSMS_ECEF_X.csv")?,
        y: sync_f64("FSMS_ECEF_Y.csv")?,
        z: sync_f64("FSMS_ECEF_Z.csv")?,

        pressure: sync_f32("FSMS_PRESSURE.csv")?,
    };

    println!("Data synced and loaded into FlightData.");
    Ok((data, timestamps))
}

// --------------------------Filter --------------------------

pub fn filter(data: &mut FlightData) {
    let (lat_ref, lon_ref, alt_ref) = get_reference_coordinates_new(&data);
    let ecef_ref = Vector3::from(latlonh_to_ecef(lat_ref, lon_ref, alt_ref));
    let rotation_matrix = ecef_to_ned_matrix(lat_ref, lon_ref);

    for i in 0..data.lat.len() {
        let ecef_current = Vector3::new(data.x[i], data.y[i], data.z[i]);
        let delta_ecef = ecef_current - ecef_ref;
        let ned = rotation_matrix * delta_ecef;

        data.x[i] = ned.x;  //North
        data.y[i] = ned.y;  //East
        data.z[i] = ned.z;  //Down

        data.roll_1[i] = data.roll_1[i].to_radians();
        data.pitch_1[i] = data.pitch_1[i].to_radians();
        data.yaw_1[i] = data.yaw_1[i].to_radians();
        
    }
    // Initalisierung
    let calibration_active = false;
    let calibration_start_time  = Instant::now();

    let mut accel_sums = Vector3::new(0.0, 0.0, 0.0);
    let mut gyro_sums = Vector3::new(0.0, 0.0, 0.0);
    let mut g_ned = Vector3::new(0.0, 0.0, 9.8);

    let mut count = 0;
    let mut calibration_count = 0;
    let mut last_valid_gps = 0;
    // Fehlend measurments, measurments_complete, altitudes, accelerations
    let rocket_started = false;


    // Initialization of orientation
    let g_body = Vector3::new(
        data.accel_x_1[0] as f64, 
        data.accel_y_1[0] as f64, 
        data.accel_z_1[0] as f64);
    let g_ned_norm = g_ned.normalize();
    let g_body_norm = g_body.normalize();

    // Quaternion (Rotationn from NED to body)
    let q_i2b = UnitQuaternion::rotation_between(&g_ned_norm, &g_body_norm).unwrap();


    // Kalman matrix initialization
    type StateVector = SVector<f64, 23>;
    let mut x = StateVector::zeros();
    
    //GPS
    x[0] = lat_ref;
    x[1] = lon_ref;
    x[2] = alt_ref;
    // 3, 4, 5 = 0 -> Speed
    // Acceleration
    x[6] = data.accel_x_1[0] as f64;
    x[7] = data.accel_y_1[0] as f64;
    x[8] = data.accel_z_1[0] as f64;
    // 9, 10, 11 = 0 -> Gyroscop
    // Quaternions
    x[12] = q_i2b[0];
    x[13] = q_i2b[1];
    x[14] = q_i2b[2];
    x[15] = q_i2b[3];
    //Biases, 16, 17, 18, 19, 20, 21 = 0
    x[22] = alt_ref - data.pressure[0] as f64;

    let p = SMatrix::<f64, 23, 23>::identity() * 0.1; // covariance
    let mut q = SMatrix::<f64, 23, 23>::identity() * 0.01; // process noise
    let mut r = SMatrix::<f64, 10, 10>::identity() * 0.5; // measurment noise
 
    // increasing process noise for baro, reducing measurment noise for gps
    q[(22, 22)] = 1.0;
    r[(0, 0)] = 0.01;
    r[(1, 1)] = 0.01;
    r[(2, 2)] = 0.01;

}

impl RocketEKF{
    pub fn new(initial_state: SVector<f64, 23>, p: SMatrix<f64, 23, 23>, q: SMatrix<f64, 23, 23>, r: SMatrix<f64, 10, 10>) -> Self {
        Self {
            state: initial_state,
            p,
            q,
            r,
            baro_needs_sync: false,
        }
    }
    pub fn predict(&mut self, dt: f64){
        let f = state_transition_jacobian(&self.state, dt);
        self.state = state_transition(&self.state, dt);
        self.p = f * self.p * f.transpose() + self.q;


        let q_slice = self.state.fixed_rows::<4>(12);
        let q_raw: [f64; 4] = [q_slice[0], q_slice[1], q_slice[2], q_slice[3]];
        
        let q_norm = normalize_quaternion(q_raw);

        self.state.fixed_rows_mut::<4>(12).copy_from_slice(&q_norm);
    }

    pub fn update(&mut self, z_measured: &SVector<f64, 10>, mask: &[bool; 10]) {
        let z_pred_full = measurement_function(&self.state, false);
        let h_full = measurement_jacobian(&self.state);
       
        // Only activ measurments
        let idx: Vec<usize> = mask.iter()
            .enumerate()
            .filter(|&(_, &active)| active)
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() { return; }

        let mut z_pred = DVector::zeros(idx.len());
        let mut h = DMatrix::zeros(idx.len(), 23);
        for (i, &current_idx) in idx.iter().enumerate() {
            z_pred[i] = z_pred_full[current_idx];
            h.set_row(i, &h_full.row(current_idx));
        }

        // R = self.R[np.ix_(idx, idx)]
        let mut r = DMatrix::zeros(idx.len(), idx.len());
        for (i, &row_idx) in idx.iter().enumerate() {
            for (j, &col_idx) in idx.iter().enumerate() {
                r[(i, j)] = self.r[(row_idx, col_idx)];
            }
        }

        // Kalman Gain
        let mut s = &h * &self.p * h.transpose() + &r;
        s = (&s + s.transpose()) / 2.0;
        
        let s_inv = s.clone().lu().try_inverse().expect("S matrix inversion failed");
        let mut k = &self.p * h.transpose() * s_inv;

        // quat slow with little gain
        for i in 0..4 {
            for col in 0..k.ncols() {
                k[(12 + i, col)] *= 0.05;
            }
        }

        // calculating innovation
        let mut innovation = DVector::zeros(idx.len());
        for (i, &current_idx) in idx.iter().enumerate() {
            innovation[i] = z_measured[current_idx] - z_pred[i];
        }

        // Catcht suddenly GPS intro
        if idx.contains(&2) {
            let h_idx_in_innovation = idx.iter().position(|&x| x == 2).unwrap();
            let h_innovation = innovation[h_idx_in_innovation];

            if h_innovation.abs() > 1000.0 {
                self.state[0] = z_measured[0];
                self.state[1] = z_measured[1];
                self.state[2] = z_measured[2];
                
                self.p[(22, 22)] = 100.0;
                innovation[h_idx_in_innovation] = 0.0;
                
                self.p.fixed_view_mut::<3, 3>(0, 0).copy_from(&(self.r.fixed_view::<3, 3>(0, 0) * 5.0));
                // In Rust setzen wir r-Werte im Update-Schritt nur lokal für die Rechnung um
                // Da R in Python self.R ist, passen wir hier self.r an:
                let mut r_gps = self.r.fixed_view_mut::<3, 3>(0, 0);
                r_gps *= 5.0;

                self.p.fixed_view_mut::<3, 3>(0, 3).scale_mut(0.05);
                self.p.fixed_view_mut::<3, 3>(3, 0).scale_mut(0.05);
                self.baro_needs_sync = true;
            }
        }

        // Baro Sync after intro GPS
        if idx.contains(&9) {
            let b_idx = idx.iter().position(|&x| x == 9).unwrap();
            if self.baro_needs_sync {
                let baro_meas = z_measured[9];
                self.state[22] = self.state[2] - baro_meas;
                self.baro_needs_sync = false;
                self.p[(22, 22)] = 100.0;
                innovation[b_idx] = 0.0;
            }
        }

        let correction = &k * innovation;
        self.state += correction;

        // Kovarianz (Joseph Form)
        // P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        let i = SMatrix::<f64, 23, 23>::identity();
        let i_kh = i - (&k * h);
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();

        // quaternion normalize
        let q_raw = [self.state[12], self.state[13], self.state[14], self.state[15]];
        let q_norm = normalize_quaternion(q_raw);
        self.state.fixed_rows_mut::<4>(12).copy_from_slice(&q_norm);

        // Covarianzmatrix symmetrical
        self.p = (&self.p + self.p.transpose()) / 2.0;
    }
}

impl FlightManager {
    pub fn new() -> Self {
        Self {
            rocket_started: false,
            ascent_flag: true,
            calibration_active: true,
            calibration_start_time: 0.0, 
            calibration_count: 0,
            last_valid_gps: 0,
            accel_gyro_window: Vec::with_capacity(21), 
            altitude_window: Vec::with_capacity(201),
        }
    }
    pub fn run_ekf_on_flightdata(
        &mut self,
        data: &FlightData,
        timestamps: &Vec<f64>,
        ekf: &mut RocketEKF,
    ) -> Vec<SVector<f64, 23>> 
    {
        let mut estimated_states = Vec::with_capacity(timestamps.len());
        let mut prev_time = timestamps[0];
        
        let mut z_prev: Option<SVector<f64, 10>> = None;

        for i in 0..timestamps.len() {
            let current_time = timestamps[i];
            let dt = current_time - prev_time;

            let cur_accel = [
                (data.accel_x_1[i] + data.accel_x_2[i]) as f64 / 2.0,
                (data.accel_y_1[i] + data.accel_y_2[i]) as f64 / 2.0,
                (data.accel_z_1[i] + data.accel_z_2[i]) as f64 / 2.0,
            ];
            let cur_gyro = [
                (data.roll_1[i] + data.roll_2[i]) as f64 / 2.0,
                (data.pitch_1[i] + data.pitch_2[i]) as f64 / 2.0,
                (data.yaw_1[i] + data.yaw_2[i]) as f64 / 2.0,
            ];

            // low pass
            self.accel_gyro_window.push([
                cur_accel[0], cur_accel[1], cur_accel[2],
                cur_gyro[0], cur_gyro[1], cur_gyro[2]
            ]);
            if self.accel_gyro_window.len() > 20 { self.accel_gyro_window.remove(0); }

            let mut mean_measurement = [0.0; 6];
            for window_row in &self.accel_gyro_window {
                for j in 0..6 { mean_measurement[j] += window_row[j]; }
            }
            for j in 0..6 { mean_measurement[j] /= self.accel_gyro_window.len() as f64; }

            // calibration
            if self.calibration_active && (current_time - self.calibration_start_time <= 5.0) { // 5s Dauer
                self.calibration_count += 1;
                ekf.q.fixed_view_mut::<4, 4>(12, 12).copy_from(&(SMatrix::<f64, 4, 4>::identity() * 1e-9));
                for j in 3..6 { mean_measurement[j] = 0.0; }
            } else if self.calibration_active {
                self.calibration_active = false;
            }

            let total_accel = (cur_accel[0].powi(2) + cur_accel[1].powi(2) + cur_accel[2].powi(2)).sqrt();
            if total_accel > 12.0 { self.rocket_started = true; }

            if self.rocket_started {
                self.altitude_window.push(ekf.state[2]);
                if self.altitude_window.len() > 200 { self.altitude_window.remove(0); }
                let mean_alt: f64 = self.altitude_window.iter().sum::<f64>() / self.altitude_window.len() as f64;
                if self.ascent_flag && (ekf.state[2] * 1.05 < mean_alt) {
                    self.ascent_flag = false;
                }
            }

            // predict
            if dt > 0.0 {
                ekf.predict(dt);
            }

            let mut z_measured = SVector::<f64, 10>::zeros();
            z_measured[0] = data.lat[i];
            z_measured[1] = data.lon[i];
            z_measured[2] = data.alt[i];
            for j in 0..6 { z_measured[3+j] = mean_measurement[j]; }
            z_measured[9] = data.pressure[i] as f64;

            let mut mask = [false; 10];
            if let Some(prev) = z_prev {
                for j in 0..10 {
                    if (z_measured[j] - prev[j]).abs() > 1e-9 {
                        mask[j] = true;
                    }
                }
            } else {
                mask = [true; 10]; 
            }

            ekf.update(&z_measured, &mask);

            estimated_states.push(ekf.state.clone());
            z_prev = Some(z_measured);
            prev_time = current_time;
        }
        estimated_states
    }
}

        

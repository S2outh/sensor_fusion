use crate::math_utils::{
    FlightData, FlightManager, RocketEKF, ecef_to_ned_matrix, latlonh_to_ecef,
    measurement_function, measurement_jacobian, normalize_quaternion, pres_to_alt,
    state_transition, state_transition_jacobian,
};
use csv::Writer;
use nalgebra::{DMatrix, DVector, SMatrix, SVector, UnitQuaternion, Vector3};
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::str::FromStr;
use std::{f32, usize};

fn open_csv<T>(
    path: &str,
    limit: usize,
    step: usize,
    row: usize,
) -> Result<(Vec<f64>, Vec<T>), Box<dyn Error>>
where
    T: FromStr,
    T::Err: Error + 'static,
{
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();
    let mut times = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        if i >= limit {
            break;
        }
        if i % step != 0 {
            continue;
        }

        let record = result?;
        if let (Some(t_str), Some(v_str)) = (record.get(0), record.get(row)) {
            let t: f64 = t_str.trim().parse()?;
            let v: T = v_str.trim().parse()?;
            times.push(t);
            values.push(v);
        }
    }

    Ok((times, values))
}

fn interpolate(target_t: f64, times: &[f64], values: &[f64], last_idx: &mut usize) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    while *last_idx < times.len() - 1 && times[*last_idx + 1] < target_t {
        *last_idx += 1;
    }
    // bfill
    if target_t <= times[0] {
        return values[0];
    }

    // ffill
    if *last_idx >= times.len() - 1 {
        return values[values.len() - 1];
    }

    values[*last_idx]
}

pub fn load_all_data() -> Result<(FlightData, Vec<f64>), Box<dyn Error>> {
    let base_path = "./src/data_set_1/";
    let limit = usize::MAX;
    //let limit = 50000;
    let step = 10;

    let start_timestamp = 486612436.0 / 100000.0;
    let skip_count = 90_000;
    //let skip_count = 1;

    let master_path = format!("{}{}", base_path, "FSMS_ACC_Z_1.csv");
    let (master_times_raw, master_vals_raw) = open_csv::<f32>(&master_path, limit, step, 1)?;

    let master_times_raw: Vec<f64> = master_times_raw
        .into_iter()
        .map(|t| (t / 10.0).round() * 10.0)
        .collect();

    let master_times_raw = &master_times_raw[skip_count..];
    let master_vals_raw = &master_vals_raw[skip_count..];
    let mut master_raw_times = Vec::new();
    let mut timestamps = Vec::new();
    let mut accel_x_1 = Vec::new();

    for (&t, &v) in master_times_raw.iter().zip(master_vals_raw.iter()) {
        let t_sec = t / 100000.0;
        if t_sec >= start_timestamp {
            master_raw_times.push(t);
            timestamps.push(t_sec);
            accel_x_1.push(v / 100.0);
        }
    }

    println!("END OF ALL NUMBERS: {}", master_raw_times.len());

    let sync_f32_scaled =
        |name: &str, scale: f32, invert: bool| -> Result<Vec<f32>, Box<dyn Error>> {
            let (s_times, s_vals) =
                open_csv::<f32>(&format!("{}{}", base_path, name), limit, step, 1)?;

            let s_times: Vec<f64> = s_times
                .into_iter()
                .map(|t| (t / 10.0).round() * 10.0)
                .collect();

            let mut last_idx = 0;
            let s_vals_f64: Vec<f64> = s_vals.iter().map(|&v| v as f64).collect();

            Ok(master_raw_times
                .iter()
                .map(|&t| {
                    let val = interpolate(t, &s_times, &s_vals_f64, &mut last_idx) as f32;
                    let scaled = val / scale;
                    if invert { -scaled } else { scaled }
                })
                .collect())
        };

    let sync_f64 = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let (s_times, s_vals) = open_csv::<f64>(&format!("{}{}", base_path, name), limit, step, 1)?;

        let s_times: Vec<f64> = s_times
            .into_iter()
            .map(|t| (t / 10.0).round() * 10.0)
            .collect();

        let mut last_idx = 0;

        Ok(master_raw_times
            .iter()
            .map(|&t| interpolate(t, &s_times, &s_vals, &mut last_idx))
            .collect())
    };

    let mut pressure = sync_f32_scaled("FSMS_PRESSURE.csv", 1.0, false)?;

    let mut last = 100_000.0_f32;
    for v in &mut pressure {
        if *v > 0.0 {
            last = *v;
        }
        *v = pres_to_alt(last);
    }

    let data = FlightData {
        accel_x_1,
        accel_y_1: sync_f32_scaled("FSMS_ACC_Y_1.csv", 100.0, true)?,
        accel_z_1: sync_f32_scaled("FSMS_ACC_X_1.csv", 100.0, false)?,

        accel_x_2: sync_f32_scaled("FSMS_ACC_Z_2.csv", 100.0, false)?,
        accel_y_2: sync_f32_scaled("FSMS_ACC_Y_2.csv", 100.0, true)?,
        accel_z_2: sync_f32_scaled("FSMS_ACC_X_2.csv", 100.0, false)?,

        roll_1: sync_f32_scaled("FSMS_GYRO_Z_1.csv", 1.0, false)?,
        pitch_1: sync_f32_scaled("FSMS_GYRO_Y_1.csv", 1.0, true)?,
        yaw_1: sync_f32_scaled("FSMS_GYRO_X_1.csv", 1.0, false)?,

        roll_2: sync_f32_scaled("FSMS_GYRO_Z_2.csv", 1.0, false)?,
        pitch_2: sync_f32_scaled("FSMS_GYRO_Y_2.csv", 1.0, true)?,
        yaw_2: sync_f32_scaled("FSMS_GYRO_X_2.csv", 1.0, false)?,

        lat: sync_f64("FSMS_PX_LAT.csv")?,
        lon: sync_f64("FSMS_PX_LONG.csv")?,
        alt: sync_f64("FSMS_PX_HEIGHT.csv")?,

        x: sync_f64("FSMS_ECEF_X.csv")?,
        y: sync_f64("FSMS_ECEF_Y.csv")?,
        z: sync_f64("FSMS_ECEF_Z.csv")?,

        pressure: pressure.clone(),
    };

    let pathi = "./vsinput.csv";
    export_flight_data_to_csv(&data, &timestamps, &pathi)?;
    println!("Data synced and loaded into FlightData.");
    Ok((data, timestamps))
}

pub fn export_flight_data_to_csv(
    data: &FlightData,
    timestamps: &[f64],
    file_path: &str,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(file_path)?;
    let mut wtr = Writer::from_writer(file);

    wtr.write_record(&[
        "timestamp",
        "accel_x_1",
        "accel_y_1",
        "accel_z_1",
        "gyro_roll_1",
        "gyro_pitch_1",
        "gyro_yaw_1",
        "pressure_alt",
        "gps_lat",
        "gps_lon",
        "gps_alt",
        "ecef_x",
        "ecef_y",
        "ecef_z",
    ])?;

    for i in 0..timestamps.len() {
        wtr.write_record(&[
            format!("{:.6}", timestamps[i]),
            format!("{:.2}", data.accel_x_1[i]),
            format!("{:.2}", data.accel_y_1[i]),
            format!("{:.2}", data.accel_z_1[i]),
            format!("{:.2}", data.roll_1[i]),
            format!("{:.2}", data.pitch_1[i]),
            format!("{:.2}", data.yaw_1[i]),
            format!("{:.2}", data.pressure[i]),
            format!("{:.5}", data.lat[i]),
            format!("{:.5}", data.lon[i]),
            format!("{:.5}", data.alt[i]),
            format!("{:.2}", data.x[i]),
            format!("{:.2}", data.y[i]),
            format!("{:.2}", data.z[i]),
        ])?;
    }

    wtr.flush()?;
    println!("CSV-Export abgeschlossen: {}", file_path);
    Ok(())
}

fn confirm() {
    println!("\n--- DATEN-CHECK ---");
    println!("press ENTER to continue");
    io::stdout().flush().unwrap();

    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .expect("Error while reading");
}

// --------------------------Filter --------------------------

pub fn init_ekf(data: &mut FlightData) -> RocketEKF {
    let start_idx = (0..data.lat.len())
        .find(|&i| data.lat[i].abs() > 0.1 && data.lon[i].abs() > 0.1 && data.alt[i] > 100.0)
        .expect("Where the hell are we!");

    let lat_ref = data.lat[start_idx];
    println!("lat_ref {}", lat_ref);
    let lon_ref = data.lon[start_idx];
    println!("lon_ref {}", lon_ref);
    let alt_ref = data.alt[start_idx];
    println!("alt_ref {}", alt_ref);
    println!("start_idk {}", start_idx);

    let ecef_ref = Vector3::from(latlonh_to_ecef(lat_ref, lon_ref, alt_ref));
    let rotation_matrix = ecef_to_ned_matrix(lat_ref, lon_ref);

    for i in start_idx..data.lat.len() {
        let ecef_current = Vector3::new(data.x[i], data.y[i], data.z[i]);
        let delta_ecef = ecef_current - ecef_ref;
        let ned = rotation_matrix * delta_ecef;

        data.x[i] = ned.x; //North
        data.y[i] = ned.y; //East
        data.z[i] = ned.z; //Down
    }

    let g_ned = Vector3::new(0.0, 0.0, 9.8); //9.8
    // Initialization of orientation
    let g_body = Vector3::new(
        data.accel_x_1[start_idx] as f64,
        data.accel_y_1[start_idx] as f64,
        data.accel_z_1[start_idx] as f64,
    );
    let g_ned_norm = g_ned.normalize();
    let g_body_norm = g_body.normalize();

    // Quaternion (lRotationn from NED to body), x, y, z, w
    let q_i2b = UnitQuaternion::rotation_between(&g_ned_norm, &g_body_norm).unwrap();

    // Kalman matrix initialization
    type StateVector = SVector<f64, 17>;
    let mut x = StateVector::zeros();

    //GPS
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    // 3, 4, 5 = 0 -> Speed
    // Quaternions
    x[6] = q_i2b.w;
    x[7] = q_i2b.i;
    x[8] = q_i2b.j;
    x[9] = q_i2b.k;
    //Biases, 10, 11, 12, 13, 14, 15
    x[16] = 0.0;

    let p = SMatrix::<f64, 17, 17>::identity() * 0.1; // covariance
    let mut q = SMatrix::<f64, 17, 17>::identity() * 0.01; // process noise
    let mut r = SMatrix::<f64, 10, 10>::identity() * 0.5; // measurment noise

    for i in 10..16 {
        q[(i, i)] = 1e-12;
    }

    // old initialization values
    q[(22, 22)] = 1.0;
    r[(0, 0)] = 0.01;
    r[(1, 1)] = 0.01;
    r[(2, 2)] = 0.01;
    r[(9, 9)] = 10_000.0;

    //let gps_pos_std = 50.0_f64;
    //let gps_alt_std = 50.0_f64;
    //let baro_alt_std = 200.0_f64;
    //let accel_std = 1.5_f64;
    //let gyro_std = 0.1_f64;
    //let deg_per_meter = 1.0 / 111132.0;
    //r[(0, 0)] = (gps_pos_std * deg_per_meter).powi(2); // lat
    //r[(1, 1)] = (gps_pos_std * deg_per_meter).powi(2); // lon
    //r[(2, 2)] = gps_alt_std.powi(2); // alt
    //r[(3, 3)] = accel_std.powi(2);
    //r[(4, 4)] = accel_std.powi(2);
    //r[(5, 5)] = accel_std.powi(2);
    //r[(6, 6)] = gyro_std.powi(2);
    //r[(7, 7)] = gyro_std.powi(2);
    //r[(8, 8)] = gyro_std.powi(2);
    //r[(9, 9)] = baro_alt_std.powi(2);
    //q[(22, 22)] = 1.0;

    RocketEKF::new(x, p, q, r)
}

impl RocketEKF {
    pub fn new(
        initial_state: SVector<f64, 17>,
        p: SMatrix<f64, 17, 17>,
        q: SMatrix<f64, 17, 17>,
        r: SMatrix<f64, 10, 10>,
    ) -> Self {
        Self {
            state: initial_state,
            p, // Kovarianz
            q, // Prozessrauschen
            r, // Messrauschen
            baro_needs_sync: false,
        }
    }
    pub fn print_state(&self) {
        println!("--- EKF State (23 Elements) ---");
        println!(
            "Pos (x,y,z):    {:>10.4} {:>10.4} {:>10.4}",
            self.state[0], self.state[1], self.state[2]
        );
        println!(
            "Vel (vx,vy,vz): {:>10.4} {:>10.4} {:>10.4}",
            self.state[3], self.state[4], self.state[5]
        );
        println!(
            "Acc : {:>10.4} {:>10.4} {:>10.4} ",
            self.state[6], self.state[7], self.state[8]
        );
        println!(
            "Gyro: {:>10.4} {:>10.4} {:>10.4}",
            self.state[9], self.state[10], self.state[11]
        );
        println!(
            "Quat (w, x, y, z): {:>10.8} {:>10.8} {:>10.8} {:>10.8}",
            self.state[12], self.state[13], self.state[14], self.state[15]
        );

        print!("Rest:           ");
        for i in 16..23 {
            print!("{:>10.4} ", self.state[i]);
        }
        println!("\n-------------------------------");
    }
    pub fn predict(&mut self, dt: f64, mean_measurment: &[f64; 6], ref_gps: &[f64; 3]) {
        // Meine
        self.p = (&self.p + self.p.transpose()) * 0.5;
        if dt > 1.0 {
            println!("There is an time issue");
        };
        let f = state_transition_jacobian(&self.state, dt, &mean_measurment);
        self.print_state();
        //if f.iter().any(|&x| x.is_nan()) {
        //println!("f after state_transition jacobian {}", f);
        //self.print_state();
        //}
        self.state = state_transition(&self.state, dt, mean_measurment, &ref_gps);
        self.p = f * self.p * f.transpose() + self.q;

        let q_slice = self.state.fixed_rows::<4>(6);
        let q_raw: [f64; 4] = [q_slice[0], q_slice[1], q_slice[2], q_slice[3]];
        let q_norm = normalize_quaternion(q_raw);
        self.state.fixed_rows_mut::<4>(6).copy_from_slice(&q_norm);
    }

    pub fn update(&mut self, z_measured: &SVector<f64, 10>, mask: &[bool; 10]) {
        let z_pred_full = measurement_function(&self.state, false);
        let h_full = measurement_jacobian(&self.state);

        // Only activ measurments
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|&(_, &active)| active)
            .map(|(i, _)| i)
            .collect();
        if idx.is_empty() {
            return;
        }

        let mut z_pred = DVector::zeros(idx.len());
        let mut h = DMatrix::zeros(idx.len(), 17);
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
        if s.iter().any(|&x| x.is_nan()) {
            println!("s kalman gain {}", s);
            //confirm();
        }
        s = (&s + s.transpose()) / 2.0;

        let s_inv = s
            .clone()
            .lu()
            .try_inverse()
            .expect("S matrix inversion failed");
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

        // GPS data update
        if idx.contains(&2) {
            let h_idx_in_innovation = idx.iter().position(|&x| x == 2).unwrap();
            let h_innovation = innovation[h_idx_in_innovation];

            // Hard reset
            if h_innovation.abs() > 1000.0 {
                println!("Not normal GPS, hard reset of position");
                //confirm();
                self.state[0] = z_measured[0];
                self.state[1] = z_measured[1];
                self.state[2] = z_measured[2];

                //increasing baro bias
                self.p[(22, 22)] = 100.0;
                innovation[h_idx_in_innovation] = 0.0;

                // increasing gps uncertainty
                self.p
                    .fixed_view_mut::<3, 3>(0, 0)
                    .copy_from(&(self.r.fixed_view::<3, 3>(0, 0) * 5.0));

                let mut r_gps = self.r.fixed_view_mut::<3, 3>(0, 0);
                r_gps *= 5.0;

                // decople gps and velocity
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
        let i = SMatrix::<f64, 17, 17>::identity();
        let i_kh = i - (&k * h);
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();
        if self.p.iter().any(|&x| x.is_nan()) {
            println!("p joseph form {}", self.p);
            confirm();
        }

        // quaternion normalize
        // w, x, y, z
        let q_raw = [
            self.state[12],
            self.state[13],
            self.state[14],
            self.state[15],
        ];
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
            block_gps: false,
            valid_gps_lat: 0.0,
            valid_gps_lon: 0.0,
            valid_gps_alt: 0.0,
            accel_gyro_window: Vec::with_capacity(21),
            altitude_window: Vec::with_capacity(201),
        }
    }
    pub fn run_ekf_on_flightdata(
        &mut self,
        data: &mut FlightData,
        timestamps: &Vec<f64>,
        ekf: &mut RocketEKF,
        start_idx: usize,
    ) -> Vec<SVector<f64, 17>> {
        let mut estimated_states = Vec::with_capacity(timestamps.len());
        let mut prev_time = timestamps[start_idx];

        let mut z_prev: Option<SVector<f64, 10>> = None;

        for i in start_idx..timestamps.len() {
            println!(
                "---------------------- Datapoint {} --------------------",
                i - start_idx
            );
            let current_time = timestamps[i];
            let dt = current_time - prev_time;

            data.roll_1[i] = data.roll_1[i].to_radians();
            data.pitch_1[i] = data.pitch_1[i].to_radians();
            data.yaw_1[i] = data.yaw_1[i].to_radians();

            data.roll_2[i] = data.roll_2[i].to_radians();
            data.pitch_2[i] = data.pitch_2[i].to_radians();
            data.yaw_2[i] = data.yaw_2[i].to_radians();

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
                cur_accel[0],
                cur_accel[1],
                cur_accel[2],
                cur_gyro[0],
                cur_gyro[1],
                cur_gyro[2],
            ]);
            if self.accel_gyro_window.len() > 20 {
                self.accel_gyro_window.remove(0);
            }

            let mut mean_measurement = [0.0; 6];
            for window_row in &self.accel_gyro_window {
                for j in 0..6 {
                    mean_measurement[j] += window_row[j];
                }
            }
            for j in 0..6 {
                mean_measurement[j] /= self.accel_gyro_window.len() as f64;
            }

            // calibration
            if self.calibration_active && (current_time - self.calibration_start_time <= 5.0) {
                // 5s Dauer
                self.calibration_count += 1;
                ekf.q
                    .fixed_view_mut::<4, 4>(12, 12)
                    .copy_from(&(SMatrix::<f64, 4, 4>::identity() * 1e-9));
                for j in 3..6 {
                    mean_measurement[j] = 0.0;
                }
            } else if self.calibration_active {
                self.calibration_active = false;
            }

            let total_accel =
                (cur_accel[0].powi(2) + cur_accel[1].powi(2) + cur_accel[2].powi(2)).sqrt();
            if total_accel > 12.0 && !self.rocket_started {
                self.rocket_started = true;
            }

            if self.rocket_started && self.ascent_flag {
                println!("Rocket started");
                //confirm();
                for i in 10..22 {
                    ekf.q[(i, i)] = 1e-12;
                }
                if ekf.state[5].abs() <= 10.0 && total_accel <= 0.25 {
                    self.ascent_flag = false;
                }
            }
            let mut ref_gps = [67.8936, 21.1053, 0.0];

            // predict
            if dt > 0.0 {
                ekf.predict(dt, &mean_measurement, &ref_gps);
            }

            if self.rocket_started {
                if self.ascent_flag{
                    if data.alt[i] >= self.valid_gps_alt{
                        self.block_gps = false;
                    }else{
                        self.block_gps = true;
                    }
                }else{
                    if data.alt[i] <= self.valid_gps_alt{
                        self.block_gps = false;
                    }else{
                        self.block_gps = true;
                    }
                }
            }

            let mut z_measured = SVector::<f64, 10>::zeros();
            z_measured[0] = data.x[i];
            z_measured[1] = data.y[i];
            z_measured[2] = data.z[i];
            for j in 0..6 {
                z_measured[3 + j] = mean_measurement[j];
            }
            z_measured[9] = data.pressure[i] as f64 - data.pressure[start_idx] as f64;

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
            if mask[9] {
                let baro_vs_gps = (z_measured[9] - z_measured[2]).abs();
                if baro_vs_gps > 100.0 {
                    mask[9] = false;
                }
            }

            ekf.update(&z_measured, &mask);
            estimated_states.push(ekf.state.clone());
            z_prev = Some(z_measured);
            prev_time = current_time;
            if i == 10{
                confirm();
            }
        }
        estimated_states
    }
}

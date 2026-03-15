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
use std::{f32, usize};

fn open_csv(
    path: &str,
    limit: usize,
    step: usize,
    row: usize,
) -> Result<(Vec<f64>, Vec<f32>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();
    let mut times = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        if i >= limit {
            break;
        }
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

fn open_csv_64(
    path: &str,
    limit: usize,
    step: usize,
    row: usize,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();
    let mut times = Vec::new();

    for (i, result) in rdr.records().enumerate() {
        if i >= limit {
            break;
        }
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
    if times.is_empty() {
        return 0.0;
    }

    while *last_idx < times.len() - 1 && times[*last_idx + 1] < target_t {
        *last_idx += 1;
    }

    if *last_idx >= times.len() - 1 {
        return values[values.len() - 1];
    }

    let t0 = times[*last_idx];
    let t1 = times[*last_idx + 1];
    let v0 = values[*last_idx];
    let v1 = values[*last_idx + 1];

    v0 + (v1 - v0) * (target_t - t0) / (t1 - t0)
}

pub fn get_times() -> Result<Vec<f64>, Box<dyn Error>> {
    let base_path = "./src/data_set_1/";
    let files = vec![
        "FSMS_ACC_Z_1.csv",
        "FSMS_ACC_Y_1.csv",
        "FSMS_ACC_X_1.csv",
        "FSMS_ACC_Z_2.csv",
        "FSMS_ACC_Y_2.csv",
        "FSMS_ACC_X_2.csv",
        "FSMS_GYRO_Z_1.csv",
        "FSMS_GYRO_Y_1.csv",
        "FSMS_GYRO_X_1.csv",
        "FSMS_GYRO_Z_2.csv",
        "FSMS_GYRO_Y_2.csv",
        "FSMS_GYRO_X_2.csv",
        "FSMS_PRESSURE.csv",
        "FSMS_PX_LAT.csv",
        "FSMS_PX_LONG.csv",
        "FSMS_PX_HEIGHT.csv",
        "FSMS_ECEF_X.csv",
        "FSMS_ECEF_Y.csv",
        "FSMS_ECEF_Z.csv",
    ];

    let mut all_timestamps: Vec<f64> = Vec::new();

    for file_name in &files {
        let path = format!("{}{}", base_path, file_name);

        if let Ok((times, _)) = open_csv_64(&path, usize::MAX, 10, 1) {
            //let times = &times[75..];
            all_timestamps.extend(times);
        }
    }

    if all_timestamps.is_empty() {
        return Err("time is an illusion".into());
    }

    all_timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_timestamps.dedup();

    Ok(all_timestamps)
}

pub fn load_all_data() -> Result<(FlightData, Vec<f64>), Box<dyn Error>> {
    let base_path = "./src/data_set_1/";
    let limit = usize::MAX;
    let step = 10;

    let master_raw_times = get_times()?;
    //let start_timestamp = 11179.65185;
    //let start_timestamp = 17077.09405;
    
    let start_timestamp = 486612436.0/100000.0;
    //let skip_count = 7512;
    let skip_count = 1;
    let master_raw_times = &master_raw_times[skip_count..];
    let timestamps: Vec<f64> = master_raw_times
        .into_iter()
        .map(|t| t / 100000.0)
        .filter(|&t| t >= start_timestamp)
        .collect();

    let sync_f32_scaled =
        |name: &str, scale: f32, invert: bool| -> Result<Vec<f32>, Box<dyn Error>> {
            let (s_times, s_vals) = open_csv(&format!("{}{}", base_path, name), limit, step, 1)?;
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
        let (s_times, s_vals) = open_csv_64(&format!("{}{}", base_path, name), limit, step, 1)?;
        let mut last_idx = 0;
        Ok(master_raw_times
            .iter()
            .map(|&t| interpolate(t, &s_times, &s_vals, &mut last_idx))
            .collect())
    };

    let mut pressure = sync_f32_scaled("FSMS_PRESSURE.csv", 1.0, false)?;
    let mut prev_valid_pressure: f32 = 100_000.0;
    for v in pressure.iter_mut() {
        let p = if *v > 0.0 {
            prev_valid_pressure = *v;
            *v
        } else {
            prev_valid_pressure
        };
        *v = pres_to_alt(p);
    }

    let data = FlightData {
        accel_x_1: sync_f32_scaled("FSMS_ACC_Z_1.csv", 100.0, false)?,
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

        //pressure: sync_f32_scaled("FSMS_PRESSURE.csv", 1.0, false)?,
        pressure: pressure,
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
            format!("{:.2}", timestamps[i]),
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
        .expect("Fehler beim Lesen");
}

// --------------------------Filter --------------------------

pub fn init_ekf(data: &mut FlightData) -> RocketEKF {
    let start_idx = data
        .lat
        .iter()
        .position(|&lat| lat.abs() > 0.1)
        .unwrap_or(0); // Fallback auf 0, falls gar nichts gefunden wird

    let lat_ref = data.lat[start_idx];
    println!("lat_ref {}", lat_ref);
    let lon_ref = data.lon[start_idx];
    println!("lon_ref {}", lon_ref);
    let alt_ref = data.alt[start_idx];
    println!("alt_ref {}", alt_ref);
    println!("start_idk {}", start_idx);
    confirm();

    let ecef_ref = Vector3::from(latlonh_to_ecef(lat_ref, lon_ref, alt_ref));
    let rotation_matrix = ecef_to_ned_matrix(lat_ref, lon_ref);

    for i in 0..data.lat.len() {
        let ecef_current = Vector3::new(data.x[i], data.y[i], data.z[i]);
        let delta_ecef = ecef_current - ecef_ref;
        let ned = rotation_matrix * delta_ecef;

        data.x[i] = ned.x; //North
        data.y[i] = ned.y; //East
        data.z[i] = ned.z; //Down

        // wird schon umgewandelt in run_ekf_on_flightdata
        //data.roll_1[i] = data.roll_1[i].to_radians();
        //data.pitch_1[i] = data.pitch_1[i].to_radians();
        //data.yaw_1[i] = data.yaw_1[i].to_radians();
    }

    let g_ned = Vector3::new(0.0, 0.0, 9.8);
    // Initialization of orientation
    let g_body = Vector3::new(
        data.accel_x_1[start_idx] as f64,
        data.accel_y_1[start_idx] as f64,
        data.accel_z_1[start_idx] as f64,
    );
    let g_ned_norm = g_ned.normalize();
    let g_body_norm = g_body.normalize();

    // Quaternion (lRotationn from NED to body)
    let q_i2b = UnitQuaternion::rotation_between(&g_ned_norm, &g_body_norm).unwrap();

    //println!("whup whup quat {}", q_i2b);

    // Kalman matrix initialization
    type StateVector = SVector<f64, 17>;
    let mut x = StateVector::zeros();

    //GPS
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    // 3, 4, 5 = 0 -> Speed

    // Quaternions x, y, z, w
    x[6] = q_i2b[1];
    x[7] = q_i2b[2];
    x[8] = q_i2b[3];
    x[9] = q_i2b[0];
    //Biases, 10, 11, 12, 13, 14, 15
    //x[22] = alt_ref - data.pressure[start_idx] as f64;
    //let pressure_offset = data.pressure[start_idx];
    x[16] = 0.0;

    let p = SMatrix::<f64, 17, 17>::identity() * 0.1; // covariance
    //println!("ppppppppppppppppppp: {}", p);
    let mut q = SMatrix::<f64, 17, 17>::identity() * 0.001; // process noise
    let mut r = SMatrix::<f64, 10, 10>::identity() * 0.001; // measurment noise
    for i in 10..16 {
        q[(i, i)] = 1e-12;
    }
    // increasing process noise for baro, reducing measurment noise for gps
    let gps_pos_std = 50.0_f64;
    let gps_alt_std = 50.0_f64;
    let baro_alt_std = 200.0_f64;
    let accel_std = 1.5_f64;
    let gyro_std = 0.1_f64;

    let deg_per_meter = 1.0 / 111132.0;

    // GPS position (lat, lon in degrees, alt in meters)
    r[(0, 0)] = (gps_pos_std * deg_per_meter).powi(2); // lat
    r[(1, 1)] = (gps_pos_std * deg_per_meter).powi(2); // lon
    r[(2, 2)] = gps_alt_std.powi(2);                   // alt

    // Accelerometer (indices 3–5)
    r[(3, 3)] = accel_std.powi(2);
    r[(4, 4)] = accel_std.powi(2);
    r[(5, 5)] = accel_std.powi(2);

    // Gyroscope (indices 6–8)
    r[(6, 6)] = gyro_std.powi(2);
    r[(7, 7)] = gyro_std.powi(2);
    r[(8, 8)] = gyro_std.powi(2);

    // Barometer alt (index 9)
    r[(9, 9)] = baro_alt_std.powi(2);

    // Process noise for baro state
    q[(16, 16)] = 1.0;
    // println!("p {}", p);
    //  println!("q {}", q);
    //  println!("x {}", x);
    //  println!("r {}", r);

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
    pub fn predict(&mut self, dt: f64, mean_measurement: &[f64; 6], ref_gps: &[f64; 3]) {
        if dt > 1.0 {
            println!(
                "AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
            HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHhh, zeit"
            );
            confirm();
        };
        let f = state_transition_jacobian(&self.state, dt, &mean_measurement);
        if f.iter().any(|&x| x.is_nan()) {
            // println!("f after state_transition jacobian {}", f);
        }
        self.state = state_transition(&self.state, dt, mean_measurement, &ref_gps);
        self.p = f * self.p * f.transpose() + self.q;

        let q_slice = self.state.fixed_rows::<4>(6);
        let q_raw: [f64; 4] = [q_slice[0], q_slice[1], q_slice[2], q_slice[3]];
        let q_norm = normalize_quaternion(q_raw);

        self.state.fixed_rows_mut::<4>(6).copy_from_slice(&q_norm);
    }

    pub fn update(&mut self, z_measured: &SVector<f64, 10>, mask: &[bool; 10]) {
        let z_pred_full = measurement_function(&self.state, false);
        //println!("after measurment {}", self.state);
        let h_full = measurement_jacobian(&self.state);
        //println!("after measurment jacobian {}", self.state);

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
            //println!("s kalman gain {}", s);
        }
        //println!("kalman h {}", h);
        //println!("kalman p {}", self.p);
        //println!("kalman h.transpose {}", h.transpose());
        //println!("kalman r {}", r);
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
            //println!("z: {}", z_measured[current_idx]);
            //println!("z_pred: {}", z_pred[i])
        }
        
        let correction = &k * innovation;
        self.state += correction;
        // Kovarianz (Joseph Form)
        // P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        let i = SMatrix::<f64, 17, 17>::identity();
        let i_kh = i - (&k * h);
        self.p = &i_kh * &self.p * i_kh.transpose() + &k * r * k.transpose();
        if self.p.iter().any(|&x| x.is_nan()) {
           // println!("p joseph form {}", self.p);
        }

        // quaternion normalize
        let q_raw = [
            self.state[6],
            self.state[7],
            self.state[8],
            self.state[9],
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
            /*println!(
                "---------------------- Datapoint {} --------------------",
                i - start_idx
            );*/
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
                //println!("Calibration counter: {}", self.calibration_count);
                //ekf.q.fixed_view_mut::<4, 4>(12, 12).copy_from(&(SMatrix::<f64, 4, 4>::identity() * 1e-9));
                for j in 3..6 {
                    mean_measurement[j] = 0.0;
                }
            } else if self.calibration_active {
                self.calibration_active = false;
            }

            let total_accel =
                (cur_accel[0].powi(2) + cur_accel[1].powi(2) + cur_accel[2].powi(2)).sqrt();
            if total_accel > 12.0 && !self.rocket_started{
                self.rocket_started = true;
            }

            if self.rocket_started && self.ascent_flag {
                println!("ROcket started");
                //confirm();
                for i in 10..22 {
                    ekf.q[(i, i)] = 1e-12;
                }
                if ekf.state[5].abs() <= 10.0 && total_accel <= 0.25 {
                    self.ascent_flag = false;
                }
            }
            let mut ref_gps= [67.8936,21.1053,0.0];
            // predict
            if dt > 0.0 {
                ekf.predict(dt, &mean_measurement, &ref_gps);
            }

            //Two options to catch wrong GPS: lower altitude during ascent, higher altitude during descent
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
            if self.block_gps{
                z_measured[0] = self.valid_gps_lat;
                z_measured[1] = self.valid_gps_lon;
                z_measured[2] = self.valid_gps_alt;

            }else{
                z_measured[0] = data.lat[i];
                z_measured[1] = data.lon[i];
                z_measured[2] = data.alt[i];
                self.valid_gps_lat = z_measured[0];
                self.valid_gps_lon = z_measured[1];
                self.valid_gps_alt = z_measured[2];
            }
            
            //NEUE

            //z_measured[0] = data.x[i];
            //z_measured[1] = data.y[i];
            //z_measured[2] = data.z[i];
            for j in 0..6 {
                z_measured[3 + j] = mean_measurement[j];
            }
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
            if i == 10{
                confirm();
            }
        }
        estimated_states
    }
}

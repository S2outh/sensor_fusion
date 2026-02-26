use std::{fs::File, time::Instant};
use std::error::Error;
use nalgebra::{SMatrix, SVector, UnitQuaternion, Vector3, Matrix3};
use crate::math_utils::{self, ecef_to_ned_matrix, get_reference_coordinates_new, latlonh_to_ecef, normalize_quaternion, state_transition, state_transition_jacobian};

const CALIBRAION_DURATION: f64 = 0.5;

fn open_csv(path: &str, limit:usize, step: usize, row: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();

    //Limit for testen, only every 10th data point
    for (i, result) in rdr.records().enumerate()  {
        if i >= limit { break; }
        if i % step == 0 {
            let record = result?;
            if let Some(val_str) = record.get(row){
                let val: f32 = val_str.trim().parse().unwrap();
                values.push(val);
            }
        }
    }
    Ok(values)
}

fn open_csv_64(path: &str, limit:usize, step: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut values = Vec::new();

    //Limit for testen, only every 10th data point
    for (i, result) in rdr.records().enumerate()  {
        if i >= limit { break; }
        if i % step == 0 {
            let record = result?;
            if let Some(val_str) = record.get(1){
                let val: f64 = val_str.trim().parse().unwrap();
                values.push(val);
            }
        }
    }
    Ok(values)
}

pub fn load_all_data() -> Result<math_utils::FlightData, Box<dyn Error>>{
    let base_path = "./src/data_set_1/";
    // how many data points in .csv file
    let limit = 280000;
    // how many points should be read. Bsp: step = 2, every 2th point
    let step = 10;

    let data = math_utils::FlightData{
        accel_x_1 : open_csv(&format!("{}FSMS_ACC_Z_1.csv", base_path), limit, step, 1)?,
        accel_y_1 : open_csv(&format!("{}FSMS_ACC_Y_1.csv", base_path), limit, step, 1)?,
        accel_z_1 : open_csv(&format!("{}FSMS_ACC_X_1.csv", base_path), limit, step, 1)?,
        
        accel_x_2 : open_csv(&format!("{}FSMS_ACC_Z_2.csv", base_path), limit, step, 1)?,
        accel_y_2 : open_csv(&format!("{}FSMS_ACC_Y_2.csv", base_path), limit, step, 1)?,
        accel_z_2 : open_csv(&format!("{}FSMS_ACC_X_2.csv", base_path), limit, step, 1)?,

        roll_1 : open_csv(&format!("{}FSMS_GYRO_Z_1.csv", base_path), limit, step, 1)?,
        pitch_1 : open_csv(&format!("{}FSMS_GYRO_Y_1.csv", base_path), limit, step, 1)?,
        yaw_1 : open_csv(&format!("{}FSMS_GYRO_X_1.csv", base_path), limit, step, 1)?,

        roll_2 : open_csv(&format!("{}FSMS_GYRO_Z_2.csv", base_path), limit, step, 1)?,
        pitch_2 : open_csv(&format!("{}FSMS_GYRO_Y_2.csv", base_path), limit, step, 1)?,
        yaw_2 : open_csv(&format!("{}FSMS_GYRO_X_2.csv", base_path), limit, step, 1)?,

        lat : open_csv_64(&format!("{}FSMS_PX_LAT.csv", base_path), limit, step)?,
        lon : open_csv_64(&format!("{}FSMS_PX_LONG.csv", base_path), limit, step)?,
        alt : open_csv_64(&format!("{}FSMS_PX_HEIGHT.csv", base_path), limit, step)?,

        x : open_csv_64(&format!("{}FSMS_ECEF_X.csv", base_path), limit, step)?,
        y : open_csv_64(&format!("{}FSMS_ECEF_Y.csv", base_path), limit, step)?,
        z : open_csv_64(&format!("{}FSMS_ECEF_Z.csv", base_path), limit, step)?,


        pressure : open_csv(&format!("{}FSMS_PRESSURE.csv", base_path), limit, step, 1)?,
    };
    println!("data collected!");
    Ok(data)
}

pub fn test(data: &mut math_utils::FlightData){
    println!("test data");
    for val in data.lat.iter(){
        println!("{}", val);
    }
    println!("data collected")
}

//------------------------ Filter --------------------------

pub fn filter(data: &mut math_utils::FlightData) {
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

    let mut accel_sums = nalgebra::Vector3::new(0.0, 0.0, 0.0);
    let mut gyro_sums = nalgebra::Vector3::new(0.0, 0.0, 0.0);
    let mut g_ned = nalgebra::Vector3::new(0.0, 0.0, 9.8);

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

    let mut ekf = math_utils::RocketEKF::new(x, p, q, r);
}

impl math_utils::RocketEKF{
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
}


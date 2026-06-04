use csv::Writer;
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::str::FromStr;
use std::{f32, usize};

use crate::math_utils::{
    FlightData, pres_to_alt,
}; 


// Help function for opening csv
pub fn open_csv<T>(
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
        // times and value pushen
        if let (Some(t_str), Some(v_str)) = (record.get(0), record.get(row)) {
            let t: f64 = t_str.trim().parse()?;
            let v: T = v_str.trim().parse()?;
            times.push(t);
            values.push(v);
        }
    }

    Ok((times, values))
}


// Help function, some sensor updates more frequently than the others. 
// This function is fixing this, by filling up the gaps with the previous numbers
pub fn interpolate(target_t: f64, times: &[f64], values: &[f64], last_idx: &mut usize) -> f64 {
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


// Reading all csv and writing the numbers into FlightData
pub fn load_all_data() -> Result<(FlightData, Vec<f64>), Box<dyn Error>> {
    let base_path = "./src/data_set_1/";

    // Limit if you only want the first x numbers
    let limit = usize::MAX;
    //let limit = 50000;
    
    // Step counter if you want faster results and you only want to calc every xth number
    let step = 10;
    
    // First timestamp, hard coded on first update
    let start_timestamp = 486612436.0 / 100000.0;

    // skip counter to skip rocket standing on launch pad
    let skip_count = 90_000;
    //let skip_count = 1;

    let master_path = format!("{}{}", base_path, "FSMS_ACC_Z_1.csv");
    let (master_times_raw, master_vals_raw) = open_csv::<f32>(&master_path, limit, step, 1)?;

    let master_times_raw: Vec<f64> = master_times_raw
        .into_iter()
        .map(|t| (t / 10.0).round() * 10.0)
        .collect();
    
    // Matching timestamps and values
    let master_times_raw = &master_times_raw[skip_count..];
    let master_vals_raw = &master_vals_raw[skip_count..];
    let mut master_raw_times = Vec::new();
    let mut timestamps = Vec::new();
    let mut accel_x_1 = Vec::new();

    //building master timeline
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
    // converting pressure into altitude
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


// export function for building csv
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

// confirm function for stopping code
pub fn confirm() {
    println!("\n--- DATEN-CHECK ---");
    println!("press ENTER to continue");
    io::stdout().flush().unwrap();

    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .expect("Error while reading");
}


use std::{error::Error, fs::File, io::Write};
use crate::{fusion::{init_ekf, load_all_data}, math_utils::FlightManager};

pub mod fusion;
pub mod math_utils;

// pos, lage, sicherheit

fn main() -> Result<(), Box<dyn Error>> {
    let mut manager = FlightManager::new();

    let (mut data, timestamps) = load_all_data().unwrap();
    let start_idx = data.lat.iter().position(|&l| l.abs() > 0.1).unwrap_or(0);
    let mut ekf = init_ekf(& mut data);
    let results = manager.run_ekf_on_flightdata(&mut data, &timestamps, &mut ekf, start_idx);

    let path = "./vsoutput.csv";
    let mut file = File::create(path)?;
    let lat_ref: f64 = 67.8935; // Beispielwert Kiruna
    let lon_ref: f64 = 21.0; 

    writeln!(file, "lat,lon,alt,v_n,v_e,v_d,q1,q2,q3,q4")?; 

    for state in results {
        let lat_est = lat_ref + (state[0] / 111132.0);
        let lon_est = lon_ref + (state[1] / (111132.0 * lat_ref.to_radians().cos()));
        let alt_est = -state[2]; 

        let line = format!(
            "{:.8},{:.8},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}",
            lat_est, lon_est, alt_est,  // Transformierte Position
            state[3], state[4], state[5],  // Vel (bleibt m/s)
            state[12], state[13], state[14], state[15] // Quat
        );
        writeln!(file, "{}", line)?;
    }
    println!("Fusion complete, result: {}", path);
    Ok(())
}


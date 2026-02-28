use std::{error::Error, fs::File, io::Write};
use nalgebra::{SMatrix, SVector};
use crate::{fusion::load_all_data, math_utils::{FlightManager, RocketEKF}};

pub mod fusion;
pub mod math_utils;


fn main() -> Result<(), Box<dyn Error>> {
    let initial_state = SVector::<f64, 23>::zeros(); 
    let p = SMatrix::<f64, 23, 23>::identity() * 1.0;
    let q = SMatrix::<f64, 23, 23>::identity() * 0.1;
    let r = SMatrix::<f64, 10, 10>::identity() * 0.5;

    let mut ekf = RocketEKF::new(initial_state, p, q, r);
    let mut manager = FlightManager::new();

    let (data, timestamps) = load_all_data().unwrap();

    let results = manager.run_ekf_on_flightdata(&data, &timestamps, &mut ekf);

    let path = "./vsoutput.csv";
    let mut file = File::create(path)?;
    
    writeln!(file, "lat_est,lon_est,alt_est,v_n,v_e,v_d,q1,q2,q3,q4")?; 
    for state in results {
        let line = format!(
            "{:.8},{:.8},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}",
            state[0], state[1], state[2],  // Pos
            state[3], state[4], state[5],  // Vel
            state[12], state[13], state[14], state[15] // Quat
        );
        writeln!(file, "{}", line)?;
    }

    println!("Berechnung abgeschlossen. Ergebnisse gespeichert in: {}", path);
    Ok(())
}


use crate::fusion::{load_all_data, test};

pub mod fusion;
pub mod math_utils;


fn main() {
    let mut flight_data = load_all_data().expect("Error while reading .csv files");
    test(&mut flight_data);
}



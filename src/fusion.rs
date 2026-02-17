use core::option::Option::{self, Some, None};

pub struct RocketEKF {
    pub state: [f32; 23],
    pub p: [[f32; 23]; 23],
    pub q: [[f32; 23]; 23],
    pub r: [[f32; 10]; 10],
    pub baro_needs_sync: bool,
}


pub mod vsv_loader{
    extern crate std;
    use std::vec::Vec;
    use std::fs::File;
    use std::string::String;
    use std::println;

    pub fn test(){
        println!("test test test");
    }
}

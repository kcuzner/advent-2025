mod day1 {
    struct Counter {
        dial: i32,
        zeros: u32,
    }

    impl Counter {
        fn new() -> Counter {
            Counter { dial: 50, zeros: 0 }
        }

        fn rotate(&mut self, line: &str) {
            if line.len() == 0 { return; }
            let rot: i32 = match line.chars().nth(0).expect("Invalid input, empty string") {
                'L' => -line[1..].parse::<i32>().expect("Invalid input, no number"),
                'R' => line[1..].parse().expect("Invalid input, no number"),
                _ => panic!("Invalid input, bad prefix"),
            };
            self.dial = (self.dial + rot).rem_euclid(100);
            if self.dial == 0 { self.zeros += 1; }
        }
    }
    pub fn run(input: &str) {
        let pw = input.split("\n").fold(Counter::new(), |mut c, i| {
            c.rotate(i);
            c
        }).zeros;
        println!("Password: {pw}")
    }
}

static DAYS: phf::Map<&'static str, fn(&str)> = phf::phf_map! {
    "1" => day1::run,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}

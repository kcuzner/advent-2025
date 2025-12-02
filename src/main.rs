mod day1 {
    struct Counter {
        dial: i32,
        zeros: u32,
    }

    impl Counter {
        fn new() -> Counter {
            Counter { dial: 50, zeros: 0 }
        }

        fn rotate(&mut self, line: &str) -> Option<()> {
            let r = rot(line)?;
            self.dial = (self.dial + r).rem_euclid(100);
            if self.dial == 0 { self.zeros += 1; }
            Some(())
        }

        fn rotate_434c49434b(&mut self, line: &str) -> Option<()> {
            let mut r = rot(line)?;
            while r != 0 {
                if r > 0 {
                    self.dial += 1;
                    r -= 1;
                } else if r < 0 {
                    self.dial -= 1;
                    r += 1;
                }
                self.dial = self.dial.rem_euclid(100);
                if self.dial == 0 { self.zeros += 1; }
            }
            Some(())
        }
    }

    fn rot(line: &str) -> Option<i32> {
        if line.len() == 0 { return None; }
        Some(match line.chars().nth(0).expect("Invalid input, empty string") {
            'L' => -line[1..].parse::<i32>().expect("Invalid input, no number"),
            'R' => line[1..].parse().expect("Invalid input, no number"),
            _ => panic!("Invalid input, bad prefix"),
        })
    }


    pub fn run1(input: &str) {
        let pw = input.split("\n").fold(Counter::new(), |mut c, i| {
            c.rotate(i);
            c
        }).zeros;
        println!("Password: {pw}")
    }

    pub fn run2(input: &str) {
        let pw = input.split("\n").fold(Counter::new(), |mut c, i| {
            c.rotate_434c49434b(i);
            c
        }).zeros;
        println!("Password: {pw}")
    }
}

static DAYS: phf::Map<&'static str, fn(&str)> = phf::phf_map! {
    "1.1" => day1::run1,
    "1.2" => day1::run2,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}

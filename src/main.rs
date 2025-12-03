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
            if self.dial == 0 {
                self.zeros += 1;
            }
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
                if self.dial == 0 {
                    self.zeros += 1;
                }
            }
            Some(())
        }
    }

    fn rot(line: &str) -> Option<i32> {
        if line.len() == 0 {
            return None;
        }
        Some(
            match line.chars().nth(0).expect("Invalid input, empty string") {
                'L' => -line[1..].parse::<i32>().expect("Invalid input, no number"),
                'R' => line[1..].parse().expect("Invalid input, no number"),
                _ => panic!("Invalid input, bad prefix"),
            },
        )
    }

    pub fn run1(input: &str) {
        let pw = input
            .split("\n")
            .fold(Counter::new(), |mut c, i| {
                c.rotate(i);
                c
            })
            .zeros;
        println!("Password: {pw}")
    }

    pub fn run2(input: &str) {
        let pw = input
            .split("\n")
            .fold(Counter::new(), |mut c, i| {
                c.rotate_434c49434b(i);
                c
            })
            .zeros;
        println!("Password: {pw}")
    }
}

mod day2 {
    fn groups<'a>(input: &'a str) -> impl Iterator<Item = &'a str> {
        input.trim().split(",")
    }

    // parse NNN-MMMM into a tuple of N, M
    fn start_end(range: &str) -> (u64, u64) {
        match range
            .split("-")
            .map(|n| n.parse::<u64>().expect("invalid number"))
            .collect::<Vec<_>>()[..]
        {
            [s, e, ..] => (s, e),
            _ => unreachable!(),
        }
    }

    pub fn run1(input: &str) {
        let sum = groups(input.trim()).fold(0, |s, group| {
            if group.len() == 0 {
                return s;
            }
            let (start, end) = start_end(group);
            (start..=end).fold(0, |c, n| {
                // Sum up all numbers in the range that have the same
                // digit pattern, repeated twice
                let v = n.to_string();
                match v.len() % 2 {
                    0 => {
                        if v[0..v.len() / 2] == v[v.len() / 2..] {
                            c + n
                        } else {
                            c
                        }
                    }
                    _ => c,
                }
            }) + s
        });
        println!("Sum of invalid IDs: {sum}")
    }
    pub fn run2(input: &str) {
        let sum = groups(input.trim()).fold(0, |sum, group| {
            if group.len() == 0 {
                return sum;
            }
            let (start, end) = start_end(group);
            (start..=end).fold(sum, |sum, n| {
                // Sum up all numbers in the range that are made up of any
                // repeated pattern. The pattern must repeat at least twice,
                // so we don't need to search for a divisor larger than half
                // the length of the string.
                let v = n.to_string();
                let invalid = (1..=v.len() / 2).fold(false, |invalid, divisor| {
                    invalid
                        || match v.len() % divisor {
                            0 => {
                                let mut iter = (0..v.len() / divisor)
                                    .map(|chunk| &v[chunk * divisor..(chunk + 1) * divisor]);
                                let first = iter.next().expect("wasn't at least one item in list");
                                iter.all(|chunk| chunk == first)
                            }
                            _ => false,
                        }
                });
                if invalid { sum + n } else { sum }
            })
        });
        println!("Sum of invalid IDs: {sum}")
    }
}

static DAYS: phf::Map<&'static str, fn(&str)> = phf::phf_map! {
    "1.1" => day1::run1,
    "1.2" => day1::run2,
    "2.1" => day2::run1,
    "2.2" => day2::run2,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}

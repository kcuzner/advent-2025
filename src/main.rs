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

mod day3 {
    fn first_highest(line: &str, start: Option<usize>, stop: Option<usize>) -> (u32, usize) {
        let begin = start.or(Some(0)).unwrap();
        let end = stop.or(Some(line.len())).unwrap();
        let (left, pos) = line[begin..end].chars().enumerate().fold(
            (None::<u32>, begin),
            |(left, pos), (index, value)| {
                let i = value.to_digit(10).expect("Couldn't decode digit");
                if left.is_none() || left.is_some_and(|l| i > l) {
                    (Some(i), index+begin)
                } else {
                    (left, pos)
                }
            },
        );
        (left.expect("Must have been an empty line"), pos)
    }

    pub fn run1(input: &str) {
        let sum = input.trim().split("\n").fold(0, |total, bank| {
            let (left, pos) = first_highest(bank, None, Some(bank.len() - 1));
            let (right, _) = first_highest(bank, Some(pos + 1), None);
            total + left * 10 + right
        });
        println!("Total joltage: {sum}");
    }

    pub fn run2(input: &str) {
        let sum = input.trim().split("\n").fold(0, |total, bank| {
            let (numbers, _) =
                (0..12)
                    .rev()
                    .fold((Vec::<u64>::new(), 0), |(mut stack, start), digit| {
                        // println!("Starting at {start} for digit {digit}");
                        let (value, pos) =
                            first_highest(bank, Some(start), Some(bank.len() - digit));
                        stack.push((value as u64) * 10_u64.pow(digit as u32));
                        (stack, pos + 1)
                    });
            // println!("{:?}", numbers);
            numbers.iter().sum::<u64>() + total
        });
        println!("Total emergency jotage: {sum}");
    }
}

static DAYS: phf::Map<&'static str, fn(&str)> = phf::phf_map! {
    "1.1" => day1::run1,
    "1.2" => day1::run2,
    "2.1" => day2::run1,
    "2.2" => day2::run2,
    "3.1" => day3::run1,
    "3.2" => day3::run2,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}

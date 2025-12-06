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
                    (Some(i), index + begin)
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

mod day4 {
    #[derive(Debug, PartialEq, Eq, Copy, Clone)]
    struct Point {
        x: usize,
        y: usize,
    }
    impl Point {
        fn new(x: usize, y: usize) -> Self {
            Self { x: x, y: y }
        }
    }

    #[derive(Debug, PartialEq, Eq, Copy, Clone)]
    enum Spot {
        Free,
        Occupied,
    }
    struct Grid {
        grid: Vec<Vec<Spot>>,
    }
    impl Grid {
        fn new(input: &str) -> Self {
            Self {
                grid: input
                    .trim()
                    .split("\n")
                    .map(|l| {
                        l.chars()
                            .map(|c| match c {
                                '@' => Spot::Occupied,
                                _ => Spot::Free,
                            })
                            .collect()
                    })
                    .collect(),
            }
        }
        fn height(&self) -> usize {
            self.grid.len()
        }
        fn width(&self) -> usize {
            self.grid.get(0).unwrap().len()
        }
        fn get(&self, p: Point) -> Option<Spot> {
            self.grid
                .get(p.y)
                .and_then(|l| l.get(p.x))
                .and_then(|s| Some(s.clone()))
        }
        fn get_adjacent(&self, p: Point, dir: Dir) -> Option<Spot> {
            let get = |x: i32, y: i32| -> Option<Spot> {
                // Need to explicitly check if we might underflow
                if p.x == 0 && x < 0 {
                    None
                } else if p.y == 0 && y < 0 {
                    None
                } else {
                    self.get(Point::new(
                        ((p.x as i32) + x) as usize,
                        ((p.y as i32) + y) as usize,
                    ))
                }
            };
            match dir {
                Dir::NW => get(-1, -1),
                Dir::N => get(0, -1),
                Dir::NE => get(1, -1),
                Dir::E => get(1, 0),
                Dir::SE => get(1, 1),
                Dir::S => get(0, 1),
                Dir::SW => get(-1, 1),
                Dir::W => get(-1, 0),
            }
        }
        fn take(&mut self, p: Point) -> bool {
            match self.grid.get_mut(p.y).and_then(|l| l.get_mut(p.x)) {
                Some(s) => {
                    *s = Spot::Free;
                    true
                }
                _ => false,
            }
        }
    }

    #[rustfmt::skip]
    #[derive(Debug, PartialEq, Eq, Copy, Clone)]
    enum Dir {
        NW, N, NE,
        W,     E,
        SW, S, SE,
    }

    impl Dir {
        fn next(&self) -> Option<Self> {
            match self {
                Dir::NW => Some(Dir::N),
                Dir::N => Some(Dir::NE),
                Dir::NE => Some(Dir::E),
                Dir::E => Some(Dir::SE),
                Dir::SE => Some(Dir::S),
                Dir::S => Some(Dir::SW),
                Dir::SW => Some(Dir::W),
                Dir::W => None,
            }
        }
    }

    struct DirIter {
        curr: Option<Dir>,
    }
    impl DirIter {
        fn new() -> Self {
            DirIter {
                curr: Some(Dir::NW),
            }
        }
    }
    impl Iterator for DirIter {
        type Item = Dir;
        fn next(&mut self) -> Option<Self::Item> {
            match self.curr {
                None => None,
                Some(c) => {
                    self.curr = c.next();
                    Some(c)
                }
            }
        }
    }

    pub fn run1(input: &str) {
        let grid = Grid::new(input);
        let sum = (0..grid.height()).fold(0, |acc, y| {
            (0..grid.width()).fold(acc, |acc, x| {
                let p = Point::new(x, y);
                match grid.get(p) {
                    Some(Spot::Occupied) => {
                        let adjacent = DirIter::new().fold(0, |adjacent, dir| {
                            match grid.get_adjacent(p, dir) {
                                Some(Spot::Occupied) => adjacent + 1,
                                _ => adjacent,
                            }
                        });
                        if adjacent < 4 { acc + 1 } else { acc }
                    }
                    _ => acc,
                }
            })
        });
        println!("Total accessible papers: {sum}");
    }

    pub fn run2(input: &str) {
        let mut grid = Grid::new(input);
        let mut removed_count = 0;
        loop {
            let to_remove = (0..grid.height()).fold(Vec::<Point>::new(), |to_remove, y| {
                (0..grid.width()).fold(to_remove, |mut to_remove, x| {
                    let p = Point::new(x, y);
                    match grid.get(p) {
                        Some(Spot::Occupied) => {
                            let adjacent = DirIter::new().fold(0, |adjacent, dir| {
                                match grid.get_adjacent(p, dir) {
                                    Some(Spot::Occupied) => adjacent + 1,
                                    _ => adjacent,
                                }
                            });
                            if adjacent < 4 {
                                to_remove.push(p);
                            }
                        }
                        _ => (),
                    }
                    to_remove
                })
            });
            removed_count += to_remove.len();
            let removed = to_remove
                .iter()
                .fold(false, |removed, p| grid.take(*p) || removed);
            if !removed {
                break;
            }
        }
        println!("Total paper removed: {removed_count}");
    }
}

mod day5 {
    // NOTE: I'm using Range rather than Range because its internal
    // fields are public. That's needed for Day 2 to make things easy.
    #[derive(PartialEq, Debug)]
    struct Range(std::ops::Range<u64>);
    impl std::ops::Deref for Range {
        type Target = std::ops::Range<u64>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl std::ops::DerefMut for Range {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl FromIterator<u64> for Range {
        fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
            let mut i = iter.into_iter();
            let start = i.next().expect("Missing start");
            let end = i.next().expect("Missing end");
            Self(start..end + 1)
        }
    }

    struct BuildRanges<'a, T>
    where
        T: Iterator<Item = &'a str>,
    {
        iter: T,
    }
    impl<'a, T> BuildRanges<'a, T>
    where
        T: Iterator<Item = &'a str>,
    {
        fn new(iter: T) -> Self {
            Self { iter: iter }
        }
        fn remaining(self) -> T {
            self.iter
        }
    }
    impl<'a, T> Iterator for BuildRanges<'a, T>
    where
        T: Iterator<Item = &'a str>,
    {
        type Item = Range;
        fn next(&mut self) -> Option<Self::Item> {
            match self.iter.next() {
                Some(s) => {
                    if s.len() == 0 {
                        return None;
                    }
                    Some(
                        s.split("-")
                            .map(|n| n.parse::<u64>().expect("Invalid integer"))
                            .collect(),
                    )
                }
                None => panic!("BuildRanges queried after range section"),
            }
        }
    }

    fn check_fresh(fresh: &Vec<Range>, id: u64) -> bool {
        for range in fresh {
            if range.contains(&id) {
                return true;
            }
        }
        false
    }

    pub fn run1(input: &str) {
        let mut builder = BuildRanges::new(input.trim().split("\n"));
        let fresh = builder.by_ref().collect::<Vec<Range>>();
        let count = builder.remaining().fold(0, |count, s| {
            if check_fresh(&fresh, s.parse::<u64>().expect("Invalid integer for id")) {
                count + 1
            } else {
                count
            }
        });
        println!("Fresh ingredients: {count}");
    }

    pub fn run2(input: &str) {
        let mut ranges = Vec::<Range>::new();
        for range in BuildRanges::new(input.trim().split("\n")) {
            match ranges.iter_mut().fold(Some(range), |range, existing| {
                match range {
                    Some(mut range) => {
                        let start_inside = existing.contains(&range.start);
                        let end_inside = existing.contains(&(range.end - 1));
                        let existing_start_inside = range.contains(&existing.start);
                        let existing_end_inside = range.contains(&(existing.end - 1));
                        if start_inside && end_inside {
                            // This range is fully contained by the existing range.
                            // We can drop this range from the list.
                            None
                        } else if existing_start_inside && existing_end_inside {
                            // The other range is fully contained by by this range.
                            // Null it out.
                            existing.end = existing.start;
                            Some(range)
                        } else {
                            if start_inside {
                                // This range overlaps the existing range at its start.
                                // Move our start past the existing range's end
                                range.start = existing.end
                            }
                            if end_inside {
                                // This range overlaps the existing range at its end.
                                // Move our end before the existing range's start
                                range.end = existing.start
                            }
                            assert!(
                                !existing.contains(&range.start)
                                    && !existing.contains(&(range.end - 1)),
                                "Range still overlaps"
                            );
                            if range.is_empty() { None } else { Some(range) }
                        }
                    }
                    _ => None,
                }
            }) {
                Some(range) => ranges.push(range),
                _ => {}
            }
        }
        let id_count = ranges.iter().fold(0, |id_count, range| {
            if !range.is_empty() {
                // Check for overlaps, I don't trust my algorithm.
                for other in ranges.iter() {
                    if other == range {
                        continue;
                    }
                    if other.contains(&range.start) || other.contains(&(range.end - 1)) {
                        println!("{other:?} contains\n{range:?}");
                    }
                }
            }
            id_count + (range.end - range.start)
        });
        println!("Total fresh IDs possible: {id_count}");
    }
}

mod day6 {
    struct Calculator {
        stack: Vec<i64>,
    }
    impl Calculator {
        fn new() -> Self {
            Self { stack: Vec::new() }
        }
        fn sum<F>(&mut self, op: F)
        where
            F: Fn(Option<i64>, i64) -> Option<i64>,
        {
            let sum = self
                .stack
                .iter()
                .fold(None, |sum, value| op(sum, *value))
                .expect("Empty stack?!??");
            // println!("Sum: '{sum}'");
            self.stack.clear();
            self.stack.push(sum);
        }
        fn process(&mut self, input: &str) {
            // println!("Processing '{input}'");
            match input.chars().nth(0) {
                Some('*') => self.sum(|sum, v| sum.and_then(|sum| Some(sum * v)).or(Some(v))),
                Some('+') => self.sum(|sum, v| sum.and_then(|sum| Some(sum + v)).or(Some(v))),
                _ => {
                    if input.trim().len() > 0 {
                        self.stack.push(input.parse().expect("Invalid number"))
                    }
                }
            }
        }
        fn value(&self) -> Option<i64> {
            match self.stack.len() {
                0 => None,
                len => self.stack.get(len - 1).and_then(|v| Some(*v)),
            }
        }
        fn clear(&mut self) {
            self.stack.clear()
        }
    }

    pub fn run1(input: &str) {
        let mut lines = input.trim().split("\n");
        let mut calculators = lines
            .next()
            .expect("No first line?")
            .split_whitespace()
            .map(|i| {
                let mut c = Calculator::new();
                c.process(i);
                c
            })
            .collect::<Vec<Calculator>>();
        lines.fold(&mut calculators, |calculators, l| {
            l.split_whitespace()
                .enumerate()
                .fold(calculators, |calculators, (index, val)| {
                    calculators[index].process(val);
                    calculators
                })
        });
        let sum = calculators
            .iter()
            .fold(0, |sum, calc| sum + calc.value().expect("No value?"));
        println!("Grand total of homework: {sum}");
    }

    pub fn run2(input: &str) {
        // It'll be easiest to parse this if I turn it on its side.
        let mut chars: Vec<Vec<char>> = Vec::new();
        // do NOT trim, alignment is important
        for line in input.split("\n") {
            if line.len() == 0 {
                continue;
            }
            for (row, ch) in line.chars().enumerate() {
                if chars.len() <= row {
                    chars.push(Vec::new())
                }
                chars
                    .get_mut(row)
                    .expect("Whoops algorithm mistake")
                    .push(ch);
            }
        }
        // They read right to left
        chars.reverse();
        // Now as we process the rows, we can just use one calculator
        let mut calculator = Calculator::new();
        let mut sum = 0;
        for row in chars {
            let string: String = row.iter().collect();
            if string.as_str().trim().len() == 0 {
                // Blank lines are the end of a problem
                sum = calculator
                    .value().expect("We need a value!") + sum;
                calculator.clear();
            } else {
                // Process the number half (all but last char) and the operator
                // half. Blank inputs will be treated as nops
                calculator.process(string[0..string.len()-1].trim());
                calculator.process(string[string.len()-1..].trim());
            }
        }
        sum = calculator.value().expect("We need a value!") + sum;
        println!("Grand total of homework: {sum}");
    }
}

static DAYS: phf::Map<&'static str, fn(&str)> = phf::phf_map! {
    "1.1" => day1::run1,
    "1.2" => day1::run2,
    "2.1" => day2::run1,
    "2.2" => day2::run2,
    "3.1" => day3::run1,
    "3.2" => day3::run2,
    "4.1" => day4::run1,
    "4.2" => day4::run2,
    "5.1" => day5::run1,
    "5.2" => day5::run2,
    "6.1" => day6::run1,
    "6.2" => day6::run2,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}

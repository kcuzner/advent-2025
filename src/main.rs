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
                sum = calculator.value().expect("We need a value!") + sum;
                calculator.clear();
            } else {
                // Process the number half (all but last char) and the operator
                // half. Blank inputs will be treated as nops
                calculator.process(string[0..string.len() - 1].trim());
                calculator.process(string[string.len() - 1..].trim());
            }
        }
        sum = calculator.value().expect("We need a value!") + sum;
        println!("Grand total of homework: {sum}");
    }
}

mod day7 {
    use std::collections::{HashMap, HashSet};

    pub fn run1(input: &str) {
        struct Data {
            // Beams for the next row
            beams: HashSet<usize>,
            splits: u32,
        }
        let data = input.trim().split("\n").fold(
            Data {
                beams: HashSet::new(),
                splits: 0,
            },
            |data, line| {
                let active = data.beams.clone();
                line.chars()
                    .enumerate()
                    .fold(data, |mut data, (index, ch)| {
                        match ch {
                            'S' => {
                                data.beams.insert(index);
                            }
                            '^' => {
                                if active.contains(&index) {
                                    data.beams.remove(&index);
                                    data.beams.insert(index - 1);
                                    data.beams.insert(index + 1);
                                    data.splits += 1;
                                }
                            }
                            _ => (),
                        }
                        data
                    })
            },
        );
        println!("Times split: {}", data.splits);
    }

    pub fn run2(input: &str) {
        struct Data {
            // Keyed by index, number of active paths at that index
            paths: HashMap<usize, u64>,
        }
        let data = input.trim().split("\n").fold(
            Data {
                paths: HashMap::new(),
            },
            |data, line| {
                line.chars()
                    .enumerate()
                    .fold(data, |mut data, (index, ch)| {
                        match ch {
                            'S' => {
                                data.paths.insert(index, 1);
                            }
                            '^' => match data.paths.remove(&index) {
                                Some(count_here) => {
                                    let (left, right) = (index - 1, index + 1);
                                    match data.paths.get_mut(&left) {
                                        Some(count_left) => *count_left += count_here,
                                        None => {
                                            data.paths.insert(left, count_here);
                                        }
                                    }
                                    match data.paths.get_mut(&right) {
                                        Some(count_right) => *count_right += count_here,
                                        None => {
                                            data.paths.insert(right, count_here);
                                        }
                                    }
                                }
                                _ => (),
                            },
                            _ => (),
                        }
                        data
                    })
            },
        );
        let sum = data.paths.iter().fold(0, |sum, (_, count)| sum + count);
        println!("Active paths: {sum}");
    }
}

mod day8 {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Eq, PartialEq, Hash, Clone, Debug)]
    struct Point {
        x: i64,
        y: i64,
        z: i64,
        circuit: usize,
    }
    impl Point {
        fn new(x: i64, y: i64, z: i64) -> Self {
            Self {
                x,
                y,
                z,
                circuit: 0,
            }
        }
        fn distance_to(&self, other: &Point) -> f64 {
            (((self.x - other.x).pow(2) + (self.y - other.y).pow(2) + (self.z - other.z).pow(2))
                as f64)
                .sqrt()
        }
    }

    impl FromIterator<i64> for Point {
        fn from_iter<I: IntoIterator<Item = i64>>(iter: I) -> Self {
            let mut iter = iter.into_iter();
            Point::new(
                iter.next().expect("Missing X"),
                iter.next().expect("Missing Y"),
                iter.next().expect("Missing Z"),
            )
        }
    }

    #[derive(Debug)]
    struct Distance {
        index_a: usize,
        index_b: usize,
        distance: f64,
    }

    impl PartialEq for Distance {
        fn eq(&self, other: &Self) -> bool {
            self.distance.eq(&other.distance)
        }
    }

    impl Eq for Distance {}

    // NOTE: distance is never NaN
    impl PartialOrd for Distance {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            other.distance.partial_cmp(&self.distance)
        }
    }
    impl Ord for Distance {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    struct Room {
        points: Vec<Point>,
        circuit_sizes: Vec<usize>,
        distances: BinaryHeap<Distance>,
    }

    impl Room {
        fn new(input: &str) -> Self {
            let points: Vec<_> = input
                .trim()
                .split("\n")
                .enumerate()
                .map(|(n, l)| {
                    let mut point: Point = l.split(",").map(|s| s.parse().unwrap()).collect();
                    point.circuit = n;
                    point
                })
                .collect();
            let circuit_sizes = vec![1usize; points.len()];
            let distances: BinaryHeap<Distance> = points
                .iter()
                .enumerate()
                .flat_map(|(index_a, a)| {
                    points.iter().enumerate().filter_map(move |(index_b, b)| {
                        if index_a >= index_b {
                            // Don't compare the same value, nor should we run the
                            // same comparison twice
                            None
                        } else {
                            Some(Distance {
                                index_a,
                                index_b,
                                distance: a.distance_to(&b),
                            })
                        }
                    })
                })
                .collect();
            Self {
                points,
                circuit_sizes,
                distances,
            }
        }

        fn consolidate(&mut self) -> Distance {
            let closest = self.distances.pop().unwrap();
            /*println!("Distance: {closest:?}");
            println!("A: {:?}", points.get(closest.index_a).unwrap());
            println!("B: {:?}", points.get(closest.index_b).unwrap());*/
            let circuit = self.points.get(closest.index_a).unwrap().circuit;
            let circuit_b = self.points.get(closest.index_b).unwrap().circuit;
            let index_b: Vec<usize> = self
                .points
                .iter()
                .enumerate()
                .filter_map(|(index, point)| {
                    if point.circuit == circuit_b {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect();
            let changed = index_b.iter().fold(0, |mut changed, index| {
                let point = self.points.get_mut(*index).unwrap();
                if point.circuit != circuit {
                    let size = self.circuit_sizes.get_mut(point.circuit).unwrap();
                    *size -= 1;
                    changed += 1;
                    point.circuit = circuit;
                }
                changed
            });
            let size = self.circuit_sizes.get_mut(circuit).unwrap();
            *size += changed;
            closest
        }

        fn into_circuit_sizes(self) -> BinaryHeap<usize> {
            self.circuit_sizes.into_iter().collect()
        }

        fn check_if_consolidated(&self) -> bool {
            let mut iter = self.circuit_sizes.iter();
            let _ = iter
                .find(|&size| size > &0)
                .expect("No nonzero circuits...where did all the points go");
            // Find the next nonzero circuit. If there isn't one, we're merged.
            iter.find(|&size| size > &0).is_none()
        }
    }

    pub fn run1(input: &str) {
        let mut room = Room::new(input);
        for _ in 0..1000 {
            room.consolidate();
        }
        let mut largest = room.into_circuit_sizes();
        let mut sum = 1;
        for i in 0..3 {
            let size = largest.pop().unwrap();
            println!("Circuit {i}: {size}");
            sum *= size;
        }
        println!("Sum: {sum}");
    }

    pub fn run2(input: &str) {
        let mut room = Room::new(input);
        let mut pair = None::<Distance>;
        while !room.check_if_consolidated() {
            pair = Some(room.consolidate());
        }
        let pair = pair.unwrap();
        let a = room.points.get(pair.index_a).unwrap();
        let b = room.points.get(pair.index_b).unwrap();
        let x = a.x * b.x;
        println!("Result: {x}");
    }
}

mod day9 {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    struct Tile {
        x: i64,
        y: i64,
    }
    impl Tile {
        fn new(x: i64, y: i64) -> Self {
            Self { x, y }
        }
        fn area(&self, other: &Self) -> i64 {
            let width = (self.x - other.x).abs() + 1;
            let height = (self.y - other.y).abs() + 1;
            width * height
        }
    }
    impl FromIterator<i64> for Tile {
        fn from_iter<I: IntoIterator<Item = i64>>(iter: I) -> Self {
            let mut iter = iter.into_iter();
            let x = iter.next().expect("No x");
            let y = iter.next().expect("No y");
            Self { x, y }
        }
    }

    struct Rectangle {
        a: Tile,
        b: Tile,
        area: i64,
    }
    impl Rectangle {
        fn new(a: &Tile, b: &Tile) -> Self {
            let area = a.area(&b);
            Self {
                a: a.clone(),
                b: b.clone(),
                area,
            }
        }

        fn corner_coordinates(&self) -> (i64, i64, i64, i64) {
            let (x1, x2) = (self.a.x, self.b.x);
            let (y1, y2) = (self.a.y, self.b.y);
            // Order the points so we can just go around
            let (x1, x2) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
            let (y1, y2) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
            (x1, x2, y1, y2)
        }

        fn corners(&self) -> impl Iterator<Item = Tile> {
            let (x1, x2, y1, y2) = self.corner_coordinates();
            vec![
                Tile::new(x1, y1),
                Tile::new(x2, y1),
                Tile::new(x2, y2),
                Tile::new(x1, y2),
            ]
            .into_iter()
        }

        fn edges(&self) -> impl Iterator<Item = Edge> {
            let corners: Vec<_> = self.corners().collect();
            let edges: Vec<_> = EdgeIter::new(corners.iter()).map(|e| e.clone()).collect();
            edges.into_iter()
        }

        fn contains(&self, edge: &Edge) -> bool {
            let (x1, x2, y1, y2) = self.corner_coordinates();
            match edge {
                Edge::Horizontal { x, y } => {
                    (x1 < x.0 && x.0 < x2) && (x1 < x.1 && x.1 < x2) && y1 < *y && *y < y2
                }
                Edge::Vertical { x, y } => {
                    (y1 < y.0 && y.0 < y2) && (y1 < y.1 && y.1 < y2) && x1 < *x && *x < x2
                }
            }
        }
    }
    impl PartialEq for Rectangle {
        fn eq(&self, other: &Self) -> bool {
            self.area.eq(&other.area)
        }
    }
    impl Eq for Rectangle {}
    impl Ord for Rectangle {
        fn cmp(&self, other: &Self) -> Ordering {
            self.area.cmp(&other.area)
        }
    }
    impl PartialOrd for Rectangle {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    #[derive(Clone, Debug)]
    enum Edge {
        Horizontal { x: (i64, i64), y: i64 },
        Vertical { x: i64, y: (i64, i64) },
    }
    impl Edge {
        fn new(a: &Tile, b: &Tile) -> Self {
            // We can assume that the tiles are co-...axial? The edge will always
            // be horizontal or vertical
            if a.x == b.x {
                Self::Vertical {
                    x: a.x,
                    y: (a.y, b.y),
                }
            } else {
                Self::Horizontal {
                    x: (a.x, b.x),
                    y: a.y,
                }
            }
        }

        fn point_in_line(&self, point: i64) -> bool {
            match self {
                Self::Horizontal { x: mx, y: _ } => {
                    if mx.0 <= mx.1 {
                        mx.0 < point && point < mx.1
                    } else {
                        mx.1 < point && point < mx.0
                    }
                }
                Self::Vertical { x: _, y: my } => {
                    if my.0 <= my.1 {
                        my.0 < point && point < my.1
                    } else {
                        my.1 < point && point < my.0
                    }
                }
            }
        }

        fn intersects(&self, other: &Self) -> bool {
            match self {
                Self::Horizontal { x: _, y: my } => match other {
                    Self::Horizontal { x: _, y: _ } => false,
                    Self::Vertical { x: ox, y: _ } => {
                        self.point_in_line(*ox) && other.point_in_line(*my)
                    }
                },
                Self::Vertical { x: mx, y: _ } => match other {
                    Self::Horizontal { x: _, y: oy } => {
                        self.point_in_line(*oy) && other.point_in_line(*mx)
                    }
                    Self::Vertical { x: _, y: _ } => false,
                },
            }
        }
    }

    struct Room {
        tiles: Vec<Tile>,
    }
    impl Room {
        fn new(input: &str) -> Self {
            let tiles: Vec<Tile> = input
                .trim()
                .split("\n")
                .map(|l| {
                    l.split(",")
                        .map(|s| s.parse().expect("Invalid number"))
                        .collect()
                })
                .collect();
            Self { tiles }
        }

        fn rectangles(&self) -> impl Iterator<Item = Rectangle> {
            self.tiles.iter().enumerate().flat_map(|(index_a, a)| {
                self.tiles
                    .iter()
                    .enumerate()
                    .filter_map(move |(index_b, b)| {
                        if index_b >= index_a {
                            return None;
                        }
                        Some(Rectangle::new(a, b))
                    })
            })
        }

        fn edges(&self) -> impl Iterator<Item = Edge> {
            EdgeIter::new(self.tiles.iter())
        }

        fn largest_area(&self) -> i64 {
            self.rectangles().fold(
                0,
                |area, rect| {
                    if rect.area > area { rect.area } else { area }
                },
            )
        }
    }

    // Generates a bunch of edges from the tiles
    struct EdgeIter<'a, T: Iterator<Item = &'a Tile>> {
        iter: T,
        first: Option<&'a Tile>,
        last: Option<&'a Tile>,
    }
    impl<'a, T> EdgeIter<'a, T>
    where
        T: Iterator<Item = &'a Tile>,
    {
        fn new(mut iter: T) -> Self {
            let first = iter.next();
            let last = first.clone();
            Self { iter, first, last }
        }

        fn next_edge(&mut self, a: &Tile) -> Option<Edge> {
            match self.iter.next().or_else(|| self.first.take()) {
                Some(b) => {
                    let edge = Some(Edge::new(&a, &b));
                    self.last = Some(b);
                    edge
                }
                _ => None,
            }
        }
    }
    impl<'a, T> Iterator for EdgeIter<'a, T>
    where
        T: Iterator<Item = &'a Tile>,
    {
        type Item = Edge;
        fn next(&mut self) -> Option<Self::Item> {
            match self.last {
                None => match self.iter.next() {
                    Some(a) => self.next_edge(a),
                    _ => None,
                },
                Some(a) => self.next_edge(a),
            }
        }
    }

    struct SvgBuilder {
        s: String,
    }
    impl SvgBuilder {
        fn new() -> Self {
            Self {
                s: "<svg height=\"100000\" width=\"100000\"  xmlns=\"http://www.w3.org/2000/svg\">\n"
                    .to_string(),
            }
        }

        fn add_tiles<T>(&mut self, iter: T)
        where
            T: IntoIterator<Item = Tile>,
        {
            let points = iter.into_iter().fold(String::new(), |mut s, t| {
                s.push_str(format!("{},{} ", t.x, t.y).as_str());
                s
            });
            self.s.push_str(
                format!(
                    "<polygon points=\"{}\"\nstyle=\"fill:green;stroke:black;stroke-width:100\" />\n",
                    points
                )
                .as_str(),
            );
        }

        fn add_rectangle(&mut self, rect: &Rectangle) {
            self.add_tiles(rect.corners());
        }

        fn svg(mut self) -> String {
            self.s.push_str("</svg>");
            self.s
        }
    }

    pub fn run1(input: &str) {
        let room = Room::new(input);
        let largest = room.largest_area();
        println!("Largest area: {largest}");
    }

    pub fn draw(input: &str) {
        // This is a frustrating problem, I need to see what I'm dealing with
        let room = Room::new(input);
        let svg_points = room.tiles.iter().fold(String::new(), |mut s, t| {
            s.push_str(format!("{},{} ", t.x, t.y).as_str());
            s
        });
        let svg = format!(
            "<svg height=\"100000\" width=\"100000\"  xmlns=\"http://www.w3.org/2000/svg\">\
            <polygon points=\"{}\" style=\"stroke:black;stroke-width:1\" /></svg>",
            svg_points
        );
        std::fs::write("day9.svg", svg).expect("Unable to write file");
    }

    pub fn run2(input: &str) {
        let room = Room::new(input);
        // Rectanges sorted by largest. We just need to find the largest
        // rectangle that is fully inside the polygon and we can quit.
        let mut rectangles: BinaryHeap<_> = room.rectangles().collect();
        println!("There are {} rectangles!", rectangles.len());
        let edges: Vec<_> = room.edges().collect();
        while let Some(rectangle) = rectangles.pop() {
            // A rectangle is fully contained in the polygon if no edges
            // intersect it. Note that our rectangles are formed from the edges
            // of the polygon, so we know the rectangle is inside the polygon:
            // The only question is if the rectangle is completely filled by
            // the polygon.
            let contained = rectangle
                .edges()
                .find(|rect_edge| {
                    edges
                        .iter()
                        .find(|room_edge| {
                            let intersects =
                                rect_edge.intersects(room_edge) || rectangle.contains(room_edge);
                            if intersects {
                                // println!("{rect_edge:?} intersected {room_edge:?}");
                            }
                            intersects
                        })
                        .is_some()
                })
                .is_none();
            if contained {
                let mut svg = SvgBuilder::new();
                svg.add_tiles(room.tiles);
                svg.add_rectangle(&rectangle);
                std::fs::write("day9.svg", svg.svg()).expect("Unable to write file");
                println!("Largest rectangle: {}", rectangle.area);
                return;
            }
        }
        println!("dang");
    }
}

mod day10 {
    use std::collections::HashSet;

    struct Machine {
        // Each bit is an indicator
        desired: u32,
        buttons: Vec<u32>,
        joltage: Vec<u32>,
    }
    impl Machine {
        fn new(line: &str) -> Self {
            let mut iter = line.trim().split_whitespace();
            let desired_str = iter.next().expect("No desired pattern");
            let desired: u32 =
                desired_str[1..desired_str.len() - 1]
                    .chars()
                    .rev()
                    .fold(0, |mut sum, ch| {
                        sum <<= 1;
                        if ch == '#' {
                            sum += 1;
                        }
                        sum
                    });
            let mut iter = iter.rev();
            let mut joltage: Vec<u32> = iter
                .next()
                .and_then(|s| Some(s[1..s.len() - 1].to_string()))
                .expect("No joltage")
                .split(",")
                .map(|n| n.parse().expect("Bad joltage number"))
                .collect();
            let buttons: Vec<u32> = iter
                .map(|s| {
                    s[1..s.len() - 1]
                        .split(",")
                        .map(|n| n.parse().expect("Bad button number"))
                        .fold(0, |mask, number| mask + 2u32.pow(number))
                })
                .collect();
            Self {
                desired,
                buttons,
                joltage,
            }
        }

        fn get_init_sequence_len(&self) -> usize {
            if self.desired == 0 {
                return 0;
            }
            // First button press...
            let mut states: HashSet<u32> = self.buttons.iter().map(|i| *i).collect();
            for i in 1.. {
                if states.contains(&self.desired) {
                    return i;
                }
                // This is a brute force algorithm, but none of the machines
                // have a particularly large control panel. This should top out
                // at like 200-500 items for the 9-ish bits I think we actually
                // use.
                let progress: Vec<_> = states.drain().collect();
                // For each in-progress sequence, press each button once and
                // add the result into the set
                for p in progress.into_iter() {
                    for b in self.buttons.iter() {
                        states.insert(p ^ b);
                    }
                }
            }
            unreachable!()
        }

        fn get_joltage_sequence_len(&self) -> usize {
            // The final joltage can be thought of as the sum of the button
            // presses. We'll create a matrix where a button press for a counter
            // places a 1 in a spot representing that counter. For example,
            // a button that hits 0,2,3,4 ends up with 1 0 1 1 1. That is then
            // rotated so that each vertical position in a column represents
            // each button. Let's say the solution was 6, 4, 10, 17, 21. We'd
            // create an augmented matrix like this:
            //
            //  1  ... 6
            //  0  ... 4
            //  1  ... 10
            //  1  ... 17
            //  1  ... 21
            //
            //  Then when we run the gaussian elimination algorithm we'll end
            //  up answering how many times each button is pressed.
            let buttons: Vec<_> = self.buttons.iter().map(|b| {
                let mut j = vec![0; self.joltage.len()];
                j.iter_mut().fold(b.clone(), |b, j| {
                    if b & 0x1 > 0 {
                        *j = 1;
                    }
                    b >> 1
                });
                j
            }).collect();

            // Create our matrix. The first index is rows. The second is the
            // columns.
            let mut matrix: Vec<Vec<f64>> = vec![Vec::new(); self.joltage.len()];
            for b in buttons {
                matrix.iter_mut().enumerate()
                    .for_each(|(i, entry)| {
                        entry.push(b[i].into());
                    });
            }
            // Pad out the array to the correct length
            for row in matrix.iter_mut() {
                row.resize(self.joltage.len(), 0f64);
            }
            // Slap the joltage on the end there
            matrix.iter_mut().enumerate()
                .for_each(|(i, entry)| {
                    entry.push(self.joltage[i].into());
                });
            println!("start");
            for r in matrix.iter() {
                println!("{r:?}");
            }

            // First reduce to triangular form by iterating the columns and
            // then scaling and subtracting the column'th row from the rows
            // below to put the matrix in row-echelon form. The lower-left
            // corner is now zeros with a diagonal line down the middle.
            for col_idx in 0..self.joltage.len() {
                for row_idx in (col_idx + 1)..self.joltage.len() {
                    let [source, dest] = matrix.get_disjoint_mut([col_idx, row_idx]).unwrap();
                    if dest[col_idx] == 0f64 {
                        continue;
                    }
                    // Create a scalar so that when we subtract source from
                    // dest, dest will become 0 in col_idx.
                    let scalar = -dest[col_idx] / source[col_idx];
                    dest.iter_mut().enumerate().for_each(|(i, d)| {
                        *d += scalar * source[i];
                    });
                }
            }
            println!("row echelon");
            for r in matrix.iter() {
                println!("{r:?}");
            }

            // Continue reducing to reduced row-echelon form by now performing
            // the same operation, but instead subtracting the column'th row
            // from the row above.
            for col_idx in 1..self.joltage.len() {
                for row_idx in 0..col_idx {
                    let [source, dest] = matrix.get_disjoint_mut([col_idx, row_idx]).unwrap();
                    if dest[col_idx] == 0f64 {
                        continue;
                    }
                    let scalar = -dest[col_idx] / source[col_idx];
                    dest.iter_mut().enumerate().for_each(|(i, d)| {
                        *d += scalar * source[i];
                    });
                }
            }
            // Finally, scale each row so that its leading number is 1.0 (unless it's zero)
            for row_idx in 0..self.joltage.len() {
                let row = &mut matrix[row_idx];
                let scalar = row[row_idx];
                if scalar == 0f64 {
                    continue
                }
                row.iter_mut().for_each(|d| {
                    *d /= scalar;
                });
            }
            println!("reduced row echelon");
            for r in matrix.iter() {
                println!("{r:?}");
            }

            // Sum up the final column values to get the joltage sequence length
            let len = matrix.iter().fold(0, |sum, row| {
                sum + (*row.last().unwrap() as usize)
            });
            println!("len: {len}");
            len
        }
    }

    /*fn apply_joltage<'a, T>(button: u32, counters: T)
    where
        T: IntoIterator<Item = &'a mut u32>,
    {
        counters.into_iter().fold(button, |button, value| {
            if button & 0x1 != 0 {
                *value += 1;
            }
            button >> 1
        });
    }*/

    pub fn run1(input: &str) {
        let machines: Vec<_> = input.trim().split("\n").map(|l| Machine::new(l)).collect();
        let presses = machines
            .iter()
            .map(|m| m.get_init_sequence_len())
            .fold(0, |sum, l| sum + l);
        println!("Total presses: {presses}");
    }

    pub fn run2(input: &str) {
        let machines: Vec<_> = input.trim().split("\n").map(|l| Machine::new(l)).collect();
        let presses = machines
            .iter()
            .map(|m| m.get_joltage_sequence_len())
            .fold(0, |sum, l| sum + l);
        println!("Total presses: {presses}");
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
    "7.1" => day7::run1,
    "7.2" => day7::run2,
    "8.1" => day8::run1,
    "8.2" => day8::run2,
    "9-draw" => day9::draw,
    "9.1" => day9::run1,
    "9.2" => day9::run2,
    "10.1" => day10::run1,
    "10.2" => day10::run2,
};

fn main() {
    let mut args = std::env::args();
    let day = args.nth(1).expect("Please supply a day");

    let filename = args.next().expect("Please supply a filename");
    let input = std::fs::read_to_string(filename).expect("Unable to open file");
    DAYS.get(&day).expect("That day isn't implemented")(&input);
}
